#!/usr/bin/env python3
"""把一次训练 run 的“真实生效参数”固化到 ``quantize/trained_params.py``。

一条命令完成：

1. 从 run 目录读 ``run_config.json``（sim_dt / ctrl_dt / PD / 动作 scale /
   VMC / 观测噪声 / 无力期 / 动作延迟）；
2. 从 float checkpoint 读网络结构 + EmpiricalNormalization 统计量；
3. 从 ``balance_infantry.xml`` 读真实关节限位；
4. 生成/覆盖 ``quantize/trained_params.py``，供
   ``quantize_model.py`` / ``test_quantized.py`` / ``deploy_obs.py`` 共同 import。

这样测试、校准、导出不再各自硬编码一份常量，训练后只需重新跑一次本命令。

用法::

    # 最新 run + 最新 checkpoint
    uv run python src/unilab/envs/locomotion/balance_infantry/quantize/sync_trained_params.py

    # 指定 run / checkpoint
    uv run python .../sync_trained_params.py \\
        --run logs/rsl_rl_sequence/BalanceInfantryJoystickFlat/2026-10-24_17-04-44_mujoco \\
        --checkpoint model_3000.pt

    # 只检查当前 trained_params.py 是否与训练产物一致（CI / 上线前）
    uv run python .../sync_trained_params.py --check

    # 只打印，不写文件
    uv run python .../sync_trained_params.py --dry-run
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[5]  # quantize → ... → repo root (UniLab/)
_UNILAB_SRC = _REPO_ROOT / "src"
_LOG_ROOT = _REPO_ROOT / "logs" / "rsl_rl_sequence"
_ASSETS_DIR = _UNILAB_SRC / "unilab" / "assets" / "robots" / "balance_infantry"
_ROBOT_XML = _ASSETS_DIR / "balance_infantry.xml"

for _p in (str(_SCRIPT_DIR), str(_UNILAB_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ckpt_meta import extract_norm_stats, infer_dims_from_ckpt  # noqa: E402

# 部署侧 obs / action 顺序的契约；quantize 目录下所有脚本都依赖这两个值。
_EXPECTED_NUM_OBS = 29
_EXPECTED_NUM_ACTIONS = 6

# balance_infantry.xml 中的关节顺序 = [LF, RF, LR, RR, LW, RW]
_JOINT_ORDER = (
    "left_front_joint",
    "right_front_joint",
    "left_rear_joint",
    "right_rear_joint",
    "left_wheel_joint",
    "right_wheel_joint",
)

_MISSING = object()


# ═══════════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════════


def _die(message):
    print(f"[sync] ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def _rel(path):
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _get(mapping, dotted, default=_MISSING):
    cur = mapping
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if default is _MISSING:
                raise KeyError(dotted)
            return default
        cur = cur[part]
    return cur


def _req(mapping, dotted, context):
    try:
        return _get(mapping, dotted)
    except KeyError:
        _die(f"run_config.json 缺少 {context}.{dotted}")


def _resolve_run(run_arg, task):
    """run_arg 可以是 run 名、run 目录、或 run_config.json 路径。"""
    if run_arg:
        p = Path(run_arg).expanduser()
        if p.is_file() and p.name == "run_config.json":
            p = p.parent
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates += [_REPO_ROOT / p, Path.cwd() / p]
            if task:
                candidates.append(_LOG_ROOT / task / p)
            candidates += sorted(_LOG_ROOT.glob(f"*/{run_arg}"))
        for c in candidates:
            if (c / "run_config.json").is_file():
                return c
        _die(f"找不到含 run_config.json 的 run: {run_arg}")

    pattern = f"{task}/*/run_config.json" if task else "*/*/run_config.json"
    configs = list(_LOG_ROOT.glob(pattern))
    if not configs:
        _die(f"{_LOG_ROOT} 下没有匹配 {pattern} 的训练产物")
    return max(configs, key=lambda f: f.stat().st_mtime).parent


def _resolve_checkpoint(run_dir, ckpt_arg):
    if ckpt_arg and str(ckpt_arg).lower() not in ("-1", "latest"):
        p = Path(ckpt_arg).expanduser()
        if p.is_file():
            return p
        if (run_dir / p).is_file():
            return run_dir / p
        if str(ckpt_arg).isdigit() and (run_dir / f"model_{ckpt_arg}.pt").is_file():
            return run_dir / f"model_{ckpt_arg}.pt"
        _die(f"checkpoint 不存在: {ckpt_arg} (run={run_dir})")

    candidates = []
    for f in run_dir.glob("model_*.pt"):
        m = re.fullmatch(r"model_(\d+)\.pt", f.name)
        if m:
            candidates.append((int(m.group(1)), f))
    if not candidates:
        _die(f"{run_dir} 下没有 model_XXXX.pt")
    return max(candidates, key=lambda item: item[0])[1]


def _joint_limits_from_xml():
    if not _ROBOT_XML.is_file():
        _die(f"找不到机器人 XML: {_ROBOT_XML}")
    joints = {}
    for joint in ET.parse(_ROBOT_XML).iter("joint"):
        name = joint.get("name")
        if name in _JOINT_ORDER:
            joints[name] = joint

    limits = []
    for name in _JOINT_ORDER:
        joint = joints.get(name)
        if joint is None:
            _die(f"{_ROBOT_XML.name} 缺少关节 {name}")
        rng = joint.get("range")
        if not rng:
            limits.append((-np.inf, np.inf))
            continue
        parts = rng.split()
        if len(parts) != 2:
            _die(f"关节 {name} 的 range 格式非法: {rng!r}")
        limits.append((float(parts[0]), float(parts[1])))
    return np.asarray(limits, dtype=np.float64)


def _pick(mapping, key, default, cast=float):
    value = mapping.get(key, default)
    if value is None:
        value = default
    return cast(value)


def _build_params(run_dir, config, ckpt_path):
    cfg = config.get("config")
    run_meta = config.get("run", {})
    if not isinstance(cfg, dict):
        _die(f"{run_dir / 'run_config.json'} 没有 config 段")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("quantized", False):
        _die(f"{ckpt_path} 是量化 checkpoint；请用训练输出的 float32 model_XXXX.pt 同步参数")

    dims = infer_dims_from_ckpt(ckpt)
    if dims["num_obs"] != _EXPECTED_NUM_OBS or dims["num_actions"] != _EXPECTED_NUM_ACTIONS:
        _die(
            f"模型维度 ({dims['num_obs']} obs, {dims['num_actions']} act) 与部署契约 "
            f"({_EXPECTED_NUM_OBS} obs, {_EXPECTED_NUM_ACTIONS} act) 不一致；"
            "deploy_obs / test_quantized 需要同步修改"
        )

    stats = extract_norm_stats(ckpt, dims)

    env = _req(cfg, "env", "config")
    ctrl = _get(env, "control_config", {}) or {}
    noise = _get(env, "noise_config", {}) or {}
    reward = _get(cfg, "reward", {}) or {}
    algo_policy = _get(cfg, "algo.policy", {}) or {}

    sim_dt = float(_req(env, "sim_dt", "config.env"))
    ctrl_dt = float(_req(env, "ctrl_dt", "config.env"))
    ctrl_steps = int(round(ctrl_dt / sim_dt))
    if ctrl_steps < 1:
        _die(f"ctrl_dt={ctrl_dt} < sim_dt={sim_dt}，无法得到有效控制步数")

    clip_actions = float(_pick(ctrl, "clip_actions", 100.0))

    action_delay_range = ctrl.get("action_delay_range", [1, 3]) or [1, 3]
    action_delay_range = [int(v) for v in action_delay_range]
    # test_quantized 固定用单个 delay；取训练范围的中间值，保持与旧行为 (2) 一致。
    action_delay = int(round(sum(action_delay_range) / len(action_delay_range)))

    obs_delay_range = ctrl.get("obs_delay_range", action_delay_range)
    obs_delay_range = [int(v) for v in (obs_delay_range or action_delay_range)]

    vmc = ctrl.get("vmc", {}) or {}
    vmc_enabled = bool(vmc.get("enabled", False))
    # 以下默认值是 PD run 里 test_quantized 的历史值；VMC run 会被配置覆盖。
    vmc_defaults = {
        "L1": 0.215,
        "L2": 0.254,
        "L3": 0.254,
        "L4": 0.215,
        "L5": 0.0,
        "feedforward_force": 80.0,
        "ff_anneal_target": None,
        "kp_theta": 50.0,
        "kd_theta": 3.0,
        "kp_l0": 500.0,
        "kd_l0": 50.0,
        "action_scale_theta": 0.7,
        "action_scale_l0": 0.15,
        "default_L0": 0.28,
        "theta_clip": 0.5,
    }
    vmc_values = {k: _pick(vmc, k, default) for k, default in vmc_defaults.items()}
    # 前馈会从 feedforward_force 退火到 ff_anneal_target，训练结束时的稳态值是后者。
    # 导出必须用稳态值：否则固件会施加一个策略在训练末期从未见过的前馈力。
    ff_steady = vmc_values["feedforward_force"]
    if vmc_values.get("ff_anneal_target") is not None:
        ff_steady = vmc_values["ff_anneal_target"]
    # 固件里 cmd_jump 的保持时长与仿真侧 auto_jump_hold_time 是同一个量，共用配置。
    jump_duration = _pick(env.get("commands", {}) or {}, "auto_jump_hold_time", 0.6)
    l0_clip = vmc.get("L0_clip", [0.15, 0.45])
    if not isinstance(l0_clip, (list, tuple)) or len(l0_clip) != 2:
        _die(f"vmc.L0_clip 必须是两个数: {l0_clip!r}")

    noise_level = float(noise.get("level", 1.0))
    # 与 deploy_obs.add_observation_noise 的语义一致：训练噪声 = level * scale。
    # 这里把 level 乘进 scale，部署侧默认 level=1.0 即复现训练分布。
    noise_defaults = {
        "gyro": 0.2,
        "gravity": 0.05,
        "joint_angle": 0.02,
        "joint_vel": 1.5,
        "vmc_phi0": 0.02,
        "vmc_L0": 0.004,
    }
    noise_scales = {
        key: float(noise.get(f"scale_{key}", default)) * noise_level
        for key, default in noise_defaults.items()
    }

    params = {
        # 结构
        "NUM_OBS": int(dims["num_obs"]),
        "NUM_ACTIONS": int(dims["num_actions"]),
        "OBS_HISTORY_LENGTH": int(dims["obs_history_length"]),
        "NUM_ENCODER_OBS": int(dims["num_encoder_obs"]),
        "LATENT_DIM": int(dims["latent_dim"]),
        "NUM_CRITIC_OBS": int(dims["num_critic_obs"]),
        "ENCODER_HIDDEN_DIMS": list(dims["encoder_hidden_dims"]),
        "ACTOR_HIDDEN_DIMS": list(dims["actor_hidden_dims"]),
        "CRITIC_HIDDEN_DIMS": list(dims["critic_hidden_dims"]),
        "ACTIVATION": str(algo_policy.get("activation", dims["activation"])),
        # 归一化
        "OBS_MEAN": stats["obs_mean"],
        "OBS_STD": stats["obs_std"],
        "LATENT_MEAN": stats["latent_mean"],
        "LATENT_STD": stats["latent_std"],
        "ENC_MEAN": stats["enc_mean"],
        "ENC_STD": stats["enc_std"],
        "ACTOR_MEAN": stats["actor_mean"],
        "ACTOR_STD": stats["actor_std"],
        "NORM_EPS": 1e-2,
        # 控制
        "SIM_DT": sim_dt,
        "CTRL_DT": ctrl_dt,
        "CTRL_STEPS": ctrl_steps,
        "OBS_CLIP": clip_actions,
        "LEG_CLIP": clip_actions,
        "WHEEL_CLIP": clip_actions,
        "LEG_CLIP_INDICES": [0, 1, 2, 3],
        "WHEEL_CLIP_INDICES": [4, 5],
        "LEG_KP": [float(v) for v in _req(ctrl, "leg_kp", "config.env.control_config")],
        "LEG_KD": [float(v) for v in _req(ctrl, "leg_kd", "config.env.control_config")],
        "WHEEL_KD": [float(v) for v in _req(ctrl, "wheel_kd", "config.env.control_config")],
        "ACTION_SCALE_JOINT": float(_req(ctrl, "action_scale_joint", "config.env.control_config")),
        "ACTION_SCALE_WHEEL": float(_req(ctrl, "action_scale_wheel", "config.env.control_config")),
        "ACTION_DELAY": action_delay,
        "ACTION_DELAY_RANGE": action_delay_range,
        "OBS_DELAY_RANGE": obs_delay_range,
        "ACTION_SMOOTHING": float(_pick(ctrl, "action_smoothing", 1.0)),
        "JOINT_LIMITS": _joint_limits_from_xml(),
        # VMC
        "VMC_ENABLED": vmc_enabled,
        "VMC_L1": float(vmc_values["L1"]),
        "VMC_L2": float(vmc_values["L2"]),
        "VMC_L3": float(vmc_values["L3"]),
        "VMC_L4": float(vmc_values["L4"]),
        "VMC_L5": float(vmc_values["L5"]),
        "VMC_FEEDFORWARD_FORCE": float(ff_steady),
        "VMC_KP_THETA": float(vmc_values["kp_theta"]),
        "VMC_KD_THETA": float(vmc_values["kd_theta"]),
        "VMC_KP_L0": float(vmc_values["kp_l0"]),
        "VMC_KD_L0": float(vmc_values["kd_l0"]),
        "VMC_ACTION_SCALE_THETA": float(vmc_values["action_scale_theta"]),
        "VMC_ACTION_SCALE_L0": float(vmc_values["action_scale_l0"]),
        "VMC_DEFAULT_L0": float(vmc_values["default_L0"]),
        "VMC_THETA_CLIP": float(vmc_values["theta_clip"]),
        "VMC_L0_CLIP": (float(l0_clip[0]), float(l0_clip[1])),
        # 以下两项只在测试 / C header 里用，训练配置没有对应字段。
        # VMC_JUMP_FORCE 在仿真里从未施加过（训练侧无此力），导出必须为 0：
        # 否则固件会额外给一个策略没见过的起跳冲量（旧值 400 N 是历史遗留）。
        "VMC_JUMP_FORCE": 0.0,
        "JUMP_DURATION": jump_duration,
        # 观测噪声
        "TRAINING_OBS_NOISE": noise_scales,
        "OBS_NOISE_LEVEL": 1.0,
        # 无力期
        "POWERLESS_DURATION": float(_pick(reward, "powerless_duration", 2.0)),
        "POWERLESS_PROBABILITY": float(_pick(reward, "powerless_probability", 0.15)),
    }

    source = {
        "run": run_dir.name,
        "checkpoint": ckpt_path.name,
        "task": str(run_meta.get("task", run_dir.parent.name)),
        "backend": str(run_meta.get("sim_backend", "mujoco")),
        "git_commit": str(_get(run_meta, "git.commit", "unknown")),
        "git_branch": str(_get(run_meta, "git.branch", "unknown")),
        "git_dirty": bool(_get(run_meta, "git.dirty", False)),
        "config": _rel(run_dir / "run_config.json"),
    }
    return params, source


# ═══════════════════════════════════════════════════════════════
# 生成 trained_params.py
# ═══════════════════════════════════════════════════════════════


def _fmt_float(value):
    value = float(value)
    if np.isnan(value):
        return "np.nan"
    if np.isposinf(value):
        return "np.inf"
    if np.isneginf(value):
        return "-np.inf"
    return repr(value)


def _fmt_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return _fmt_float(value)


def _fmt_list(values):
    return "[" + ", ".join(_fmt_scalar(v) for v in values) + "]"


def _fmt_array(values, dtype="np.float32"):
    arr = np.asarray(values, dtype=np.float32)
    body = np.array2string(
        arr,
        precision=8,
        floatmode="unique",
        separator=", ",
        max_line_width=110,
    )
    return f"np.array({body}, dtype={dtype})"


def _fmt_joint_limits(limits):
    rows = []
    for lo, hi in np.asarray(limits, dtype=np.float64):
        rows.append(f"    [{_fmt_float(lo)}, {_fmt_float(hi)}]")
    return "np.array([\n" + ",\n".join(rows) + ",\n], dtype=np.float64)"


def _fmt_noise(scales):
    lines = ["{"]
    for key, value in scales.items():
        lines.append(f'    "{key}": {_fmt_float(value)},')
    lines.append("}")
    return "\n".join(lines)


def _render(params, source):
    p = params
    lines = []
    lines.append('"""AUTO-GENERATED by quantize/sync_trained_params.py — do not edit.')
    lines.append("")
    lines.append("训练 run 的实际生效参数：网络结构 / 归一化统计 / PD / 动作 scale /")
    lines.append("VMC / 观测噪声 / 无力期。测试、校准、导出统一 import 本文件。")
    lines.append("")
    lines.append("Provenance:")
    lines.append(f"  run        : {source['run']}")
    lines.append(f"  checkpoint : {source['checkpoint']}")
    lines.append(f"  task       : {source['task']}")
    lines.append(f"  backend    : {source['backend']}")
    lines.append(f"  git        : {source['git_commit']} ({source['git_branch']}"
                 f"{', dirty' if source['git_dirty'] else ''})")
    lines.append(f"  config     : {source['config']}")
    lines.append("")
    lines.append("重新训练后重新生成:")
    lines.append("  uv run python src/unilab/envs/locomotion/balance_infantry/quantize/"
                 "sync_trained_params.py")
    lines.append('"""')
    lines.append("")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("")
    lines.append("SOURCE = {")
    for key, value in source.items():
        if isinstance(value, bool):
            rendered = "True" if value else "False"
        else:
            rendered = repr(str(value))
        lines.append(f"    {key!r}: {rendered},")
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("# ── 网络结构 (from checkpoint state_dict) ──")
    for key in (
        "NUM_OBS", "NUM_ACTIONS", "OBS_HISTORY_LENGTH", "NUM_ENCODER_OBS",
        "LATENT_DIM", "NUM_CRITIC_OBS", "ENCODER_HIDDEN_DIMS",
        "ACTOR_HIDDEN_DIMS", "CRITIC_HIDDEN_DIMS", "ACTIVATION",
    ):
        value = p[key]
        if isinstance(value, list):
            lines.append(f"{key} = {_fmt_list(value)}")
        elif isinstance(value, str):
            lines.append(f"{key} = {value!r}")
        else:
            lines.append(f"{key} = {_fmt_scalar(value)}")
    lines.append("")
    lines.append("")
    lines.append("# ── 观测归一化 (from EmpiricalNormalization buffers) ──")
    lines.append("# 公式: y = (x - mean) / (std + NORM_EPS)")
    for key in ("OBS_MEAN", "OBS_STD", "LATENT_MEAN", "LATENT_STD",
                "ENC_MEAN", "ENC_STD", "ACTOR_MEAN", "ACTOR_STD"):
        lines.append(f"{key} = {_fmt_array(p[key])}")
    lines.append("VEL_MEAN = LATENT_MEAN")
    lines.append("VEL_STD = LATENT_STD")
    lines.append(f"NORM_EPS = {_fmt_float(p['NORM_EPS'])}")
    lines.append("")
    lines.append("")
    lines.append("# ── 控制 (from run_config.json env.control_config) ──")
    for key in ("SIM_DT", "CTRL_DT", "CTRL_STEPS"):
        lines.append(f"{key} = {_fmt_scalar(p[key])}")
    for key in ("OBS_CLIP", "LEG_CLIP", "WHEEL_CLIP"):
        lines.append(f"{key} = {_fmt_float(p[key])}")
    for key in ("LEG_CLIP_INDICES", "WHEEL_CLIP_INDICES"):
        lines.append(f"{key} = {_fmt_list(p[key])}")
    for key in ("LEG_KP", "LEG_KD", "WHEEL_KD"):
        lines.append(f"{key} = {_fmt_list(p[key])}")
    for key in ("ACTION_SCALE_JOINT", "ACTION_SCALE_WHEEL", "ACTION_SMOOTHING"):
        lines.append(f"{key} = {_fmt_float(p[key])}")
    lines.append(f"ACTION_DELAY = {_fmt_scalar(p['ACTION_DELAY'])}")
    lines.append(f"ACTION_DELAY_RANGE = {_fmt_list(p['ACTION_DELAY_RANGE'])}")
    lines.append(f"OBS_DELAY_RANGE = {_fmt_list(p['OBS_DELAY_RANGE'])}")
    lines.append("JOINT_LIMITS = " + _fmt_joint_limits(p["JOINT_LIMITS"]))
    lines.append("")
    lines.append("")
    lines.append("# ── VMC (from run_config.json control_config.vmc) ──")
    lines.append(f"VMC_ENABLED = {_fmt_scalar(p['VMC_ENABLED'])}")
    for key in ("VMC_L1", "VMC_L2", "VMC_L3", "VMC_L4", "VMC_L5",
                "VMC_FEEDFORWARD_FORCE", "VMC_KP_THETA", "VMC_KD_THETA",
                "VMC_KP_L0", "VMC_KD_L0", "VMC_ACTION_SCALE_THETA",
                "VMC_ACTION_SCALE_L0", "VMC_DEFAULT_L0", "VMC_THETA_CLIP"):
        lines.append(f"{key} = {_fmt_float(p[key])}")
    lines.append(f"VMC_L0_CLIP = ({_fmt_float(p['VMC_L0_CLIP'][0])}, "
                 f"{_fmt_float(p['VMC_L0_CLIP'][1])})")
    lines.append("# 以下两项训练配置没有对应字段，保持 test_quantized / C header 的历史值。")
    lines.append(f"VMC_JUMP_FORCE = {_fmt_float(p['VMC_JUMP_FORCE'])}")
    lines.append(f"JUMP_DURATION = {_fmt_float(p['JUMP_DURATION'])}")
    lines.append("")
    lines.append("")
    lines.append("# ── 观测噪声 (from env.noise_config；level 已乘进 scale) ──")
    lines.append(f"OBS_NOISE_LEVEL = {_fmt_float(p['OBS_NOISE_LEVEL'])}")
    lines.append("TRAINING_OBS_NOISE = " + _fmt_noise(p["TRAINING_OBS_NOISE"]))
    lines.append("")
    lines.append("")
    lines.append("# ── 无力期 (from config.reward) ──")
    lines.append(f"POWERLESS_DURATION = {_fmt_float(p['POWERLESS_DURATION'])}")
    lines.append(f"POWERLESS_PROBABILITY = {_fmt_float(p['POWERLESS_PROBABILITY'])}")
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="把训练 run 的真实参数导出到 quantize/trained_params.py")
    parser.add_argument("--run", default=None,
                        help="run 目录 / run 名 / run_config.json；默认取最新 run")
    parser.add_argument("--task", default=None,
                        help="任务名 (默认从 run 路径或 run_config 推断)")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint 路径或文件名；默认取 run 下最新 model_XXXX.pt")
    parser.add_argument("--out", default=str(_SCRIPT_DIR / "trained_params.py"),
                        help="输出文件 (默认 quantize/trained_params.py)")
    parser.add_argument("--check", action="store_true",
                        help="只检查 out 是否与训练产物一致，不一致返回 1")
    parser.add_argument("--dry-run", action="store_true",
                        help="打印生成内容，不写文件")
    args = parser.parse_args()

    run_dir = _resolve_run(args.run, args.task)
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task = args.task or str(_get(config.get("run", {}), "task", run_dir.parent.name))

    ckpt_path = _resolve_checkpoint(run_dir, args.checkpoint)
    params, source = _build_params(run_dir, config, ckpt_path)
    text = _render(params, source)

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    if args.dry_run:
        print(text)
        return 0

    if args.check:
        if not out_path.is_file():
            print(f"[sync] MISSING: {out_path}")
            return 1
        current = out_path.read_text(encoding="utf-8")
        if current == text:
            print(f"[sync] OK: {_rel(out_path)} 与 run={source['run']} / "
                  f"ckpt={source['checkpoint']} 一致")
            return 0
        diff = difflib.unified_diff(
            current.splitlines(), text.splitlines(),
            fromfile=str(out_path), tofile="expected", lineterm="",
        )
        print("\n".join(list(diff)[:400]))
        print(f"\n[sync] DRIFT: {_rel(out_path)} 与训练产物不一致；"
              "运行 sync_trained_params.py 重新生成")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"[sync] run={source['run']}  ckpt={source['checkpoint']}  "
          f"task={source['task']}  backend={source['backend']}")
    print(f"[sync] wrote {_rel(out_path)}")
    print(f"        dims: obs={params['NUM_OBS']} act={params['NUM_ACTIONS']} "
          f"history={params['OBS_HISTORY_LENGTH']} latent={params['LATENT_DIM']}")
    print(f"        control: kp={params['LEG_KP']} kd={params['LEG_KD']} "
          f"act_scale={params['ACTION_SCALE_JOINT']}/{params['ACTION_SCALE_WHEEL']} "
          f"dt={params['SIM_DT']}/{params['CTRL_DT']} delay={params['ACTION_DELAY']}")
    print(f"        latent norm: mean={np.round(params['LATENT_MEAN'], 4).tolist()} "
          f"std={np.round(params['LATENT_STD'], 4).tolist()}")
    print("        下一步: test_quantized.py / quantize_model.py / export_onnx.py "
          "会自动 import 这些值")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
