#!/usr/bin/env python3
"""加载 int8 量化 PPO 策略，在 MuJoCo 仿真中测试 Sim-to-Sim 效果。

参数与 mujoco.yaml 配置对齐。

观测/执行约定:
  - 观测噪声由 ``--obs_noise_level`` 控制，默认 ``1.0`` = 与训练同分布
    (gyro / dof_vel / dof_pos_err / pitch / roll / theta / L0)；
    ``--obs_noise_level 0`` 关闭。compare 模式下两个模型吃同一份噪声，
    差值仍然只反映量化误差。
  - 观测延迟: 不加 (训练侧有 obs delay，这里刻意省略)。
  - 动作延迟固定 ``ACTION_DELAY`` 个 sim step (见常量)，不做随机化。
  - 腿部质量缩放见 ``MASS_SCALE`` (默认 1.0 = 不缩放)。

用法:
    cd /path/to/balance_infantry

    # int8 量化模型测试 (默认 rmuc 场景)
    python quantize/test_quantized.py --model quantize/ppo_int8.pt

    # float32 原始模型测试
    python quantize/test_quantized.py --model /home/xxx/ubuntu_software/unilab/UniLab/logs/rsl_rl_sequence/BalanceInfantryJoystickFlat/2026-09-05_21-08-28_mujoco/model_4000.pt --no_quantize

    # 对比模式: 同时跑 float32 和 int8, 统计差异
    # float32 baseline 默认取同目录的 model_XXXX.pt (可用 --float_model 覆盖)
    python quantize/test_quantized.py --model quantize/ppo_int8.pt --compare

    # 指定场景
    python quantize/test_quantized.py --model quantize/ppo_int8.pt --scene flat
"""

import argparse
import ctypes
import os
import sys
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.join(_SCRIPT_DIR, "..")
_LQR_DIR = os.path.join(_PARENT_DIR, "lqr")
_SL_DIR = os.path.join(_PARENT_DIR, "SL")
_UNILAB_SRC = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "..")

sys.path.insert(0, _PARENT_DIR)
sys.path.insert(0, _LQR_DIR)
sys.path.insert(0, _SL_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_UNILAB_SRC, "src"))

import deploy_obs
from deploy_obs import build_observation as _build_deploy_observation
from test_vmc_lqr import compute_jacobian_partials, fk_fivebar

from unilab.algos.torch.ppo.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence

# 从 quantize_model 导入自动检测的模型参数（向后兼容硬编码 fallback）
try:
    from quantize_model import DETECTED, infer_dims_from_ckpt
    from quantize_model import _replace_linear_with_int8_calib as _replace_linear_with_int8_calib
    from ckpt_meta import warn_if_trained_params_mismatch
except ImportError:
    DETECTED = {
        "num_obs": 29, "num_actions": 6, "obs_history_length": 5,
        "num_encoder_obs": 145, "latent_dim": 3,
        "encoder_hidden_dims": [128, 64], "actor_hidden_dims": [128, 64, 32],
        "critic_hidden_dims": [256, 128, 64], "activation": "elu",
    }

    def warn_if_trained_params_mismatch(_path):
        pass

# 控制 / VMC / 噪声 / 无力期全部来自训练产物。
# 训练后运行 quantize/sync_trained_params.py 覆盖 quantize/trained_params.py。
from trained_params import (  # noqa: E402
    ACTION_DELAY,
    ACTION_SCALE_JOINT,
    ACTION_SCALE_WHEEL,
    CTRL_DT,
    CTRL_STEPS,
    JOINT_LIMITS,
    JUMP_DURATION,
    LEG_CLIP,
    LEG_CLIP_INDICES,
    LEG_KD,
    LEG_KP,
    POWERLESS_DURATION,
    POWERLESS_PROBABILITY,
    SIM_DT,
    VMC_ACTION_SCALE_L0,
    VMC_ACTION_SCALE_THETA,
    VMC_DEFAULT_L0,
    VMC_FEEDFORWARD_FORCE,
    VMC_KD_L0,
    VMC_KD_THETA,
    VMC_KP_L0,
    VMC_KP_THETA,
    VMC_L0_CLIP,
    VMC_L1,
    VMC_L4,
    VMC_THETA_CLIP,
    WHEEL_CLIP,
    WHEEL_CLIP_INDICES,
    WHEEL_KD,
)

# ═══════════════════════════════════════════════════════════════
# PD 关节目标 & 实物注入 (直接修改此处即可)
# ═══════════════════════════════════════════════════════════════
PD_TARGET = np.array([-0.833, -0.897, -0.204, -0.101], dtype=np.float64)
# PD_TARGET (°):     LF=-47.7°  RF=-51.4°  LR=-11.7°  RR=-5.8°
# 实物 dof_err:       LF=-0.893  RF=-0.907  LR=-0.104  RR=-0.101  (rad)

INJECT_ACT = np.array([-1.722, -1.586, 1.726, 0.919, -2.373, -1.841], dtype=np.float32)
# 实物 last_actions:  LF         RF         LR         RR         LW         RW
INJECT_CMD_YAW_RATE = 0.3912  # 实物 cmd_yaw (rad/s)，按P注入

# H 键：只发 cmd_jump=1 命令，由策略自己决定怎么跳（不再外加大 VMC 前馈）。
# 实测一次跳跃：vz 峰值 ~3 步、开始下落 ~6 步、落地中位 8 步 / p90 17 步 / 最大 29 步；
# 物理上限 vz=1.43 → 2vz/g ≈ 29 步滞空。
# 保持时长 = 下蹲(5~10 步) + 滞空(中位 8 / 最大 29 步)，即 0.4~0.5 s 才够。
# 0.35 s 会在刚离地时就清零 cmd_jump，策略随即退出收腿（leg_tuck 只在 cmd_jump=1 时被奖励）。
# 数值来自配置 commands.auto_jump_hold_time（经 sync_trained_params 写入 JUMP_DURATION）。
JUMP_HOLD_SEC = JUMP_DURATION

# 方向键：与 play_interactive 的 KeyboardCommander 一致 —— 每按一次"增量微调"，
# 不是切换。vx ±0.5 m/s、vyaw ±0.2 rad/s，clamp 到训练的命令范围。
CMD_VX_STEP, CMD_YAW_STEP = 0.5, 0.2
CMD_VX_LIMIT = (-2.5, 2.5)       # 训练 commands.vel_limit
CMD_YAW_LIMIT = (-8.0, 8.0)

# base_height_rangefinder 的射线起点在 base 前方 0.30 m（robot.xml 的 site pos="0.30 0 0"），
# 所以"机体中心到障碍"的距离 = 传感器读数 + RF_SITE_OFFSET。
RF_SITE_OFFSET = 0.30

# 实物 tau (带VMC):   LF=+20.6  RF=+21.1  LR=+37.4  RR=+17.0  LW=-3.5  RW=-3.5  (Nm)

# ═══════════════════════════════════════════════════════════════
# 模型超参数 —— 从 DETECTED 获取，不再硬编码。
# ═══════════════════════════════════════════════════════════════
NUM_OBS = DETECTED["num_obs"]
NUM_ACTIONS = DETECTED["num_actions"]
OBS_HISTORY_LENGTH = DETECTED["obs_history_length"]
NUM_ENCODER_OBS = DETECTED["num_encoder_obs"]
LATENT_DIM = DETECTED["latent_dim"]
NUM_CRITIC_OBS = DETECTED["num_critic_obs"]
ENCODER_HIDDEN_DIMS = DETECTED["encoder_hidden_dims"]
ACTOR_HIDDEN_DIMS = DETECTED["actor_hidden_dims"]
CRITIC_HIDDEN_DIMS = DETECTED["critic_hidden_dims"]
ACTIVATION = DETECTED["activation"]

# PD / 动作裁剪 / VMC / ACTION_DELAY 均 import 自 trained_params。

# ═══════════════════════════════════════════════════════════════
# 模式检测: 根据模型路径自动选择 Joint PD / VMC
# ═══════════════════════════════════════════════════════════════
_CONTROL_MODE = "pd"  # 默认，main() 中根据路径覆盖


def detect_control_mode(model_path):
    """根据模型路径名自动选择控制模式: "VMC" 或 "pd"."""
    path_upper = model_path.upper()
    if "VMC" in path_upper:
        return "vmc"
    return "pd"

# 无力期 / 控制步长均 import 自 trained_params。


# ═══════════════════════════════════════════════════════════════
# 观测构建 (与 joystick.py _compute_obs 一致)
# ═══════════════════════════════════════════════════════════════

def build_observation(model, data, dof_pos, dof_vel, last_actions,
                      cmd_vel, cmd_yaw_rate, cmd_height,
                      yaw_error, default_dof_pos, cmd_jump=0.0):
    """29 维原始观测 —— 委托给 deploy_obs (与训练 env 同一套公式)。

    ``model`` / ``yaw_error`` 保留在签名里只为兼容旧调用点。
    ``cmd_jump`` 是 H 键发出的跳跃命令（0/1），跳任务的策略会读这一维。
    """
    return _build_deploy_observation(
        gyro=data.qvel[3:6],
        quat=data.qpos[3:7],
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        last_actions=last_actions,
        cmd_vel=cmd_vel,
        cmd_yaw_rate=cmd_yaw_rate,
        cmd_height=cmd_height,
        cmd_jump=cmd_jump,
        default_dof_pos=default_dof_pos,
    )


OBS_LABELS = [
    ( 0, "gyro_x           ", "ωx, roll rate  (rad/s), 右滚>0"),
    ( 1, "gyro_y           ", "ωy, pitch rate (rad/s), 低头>0"),
    ( 2, "gyro_z           ", "ωz, yaw rate   (rad/s), 左转>0"),
    ( 3, "pitch            ", "asin(gx) (rad), 前倾>0, 后仰<0"),
    ( 4, "roll             ", "asin(-gy)(rad), 左倾>0, 右倾<0"),
    ( 5, "cmd_vel          ", "前进速度命令 (m/s), 前>0"),
    ( 6, "cmd_yaw_rate     ", "偏航角速度命令 (rad/s), 左转>0"),
    ( 7, "cmd_height       ", "目标高度 (m)"),
    ( 8, "cmd_standing     ", "站立标志"),
    ( 9, "last_action_LF   ", "动作 LF"),
    (10, "last_action_RF   ", "动作 RF"),
    (11, "last_action_LR   ", "动作 LR"),
    (12, "last_action_RR   ", "动作 RR"),
    (13, "last_action_LW   ", "动作 LW"),
    (14, "last_action_RW   ", "动作 RW"),
    (15, "dof_pos_err_LF   ", "LF 关节偏差 (rad)"),
    (16, "dof_pos_err_RF   ", "RF 关节偏差 (rad)"),
    (17, "dof_pos_err_LR   ", "LR 关节偏差 (rad)"),
    (18, "dof_pos_err_RR   ", "RR 关节偏差 (rad)"),
    (19, "dof_vel_LF       ", "LF 关节速度 (rad/s)"),
    (20, "dof_vel_RF       ", "RF 关节速度 (rad/s)"),
    (21, "dof_vel_LR       ", "LR 关节速度 (rad/s)"),
    (22, "dof_vel_RR       ", "RR 关节速度 (rad/s)"),
    (23, "dof_vel_LW       ", "LW 轮子速度 (rad/s)"),
    (24, "dof_vel_RW       ", "RW 轮子速度 (rad/s)"),
    (25, "theta_l          ", "左腿 VMC 摆角 (rad), 前倾增大"),
    (26, "theta_r          ", "右腿 VMC 摆角 (rad), 前倾增大"),
    (27, "L0_l             ", "左腿虚拟腿长 (m), ~0.28"),
    (28, "L0_r             ", "右腿虚拟腿长 (m), ~0.28"),
]


OBS_NAMES = [
    "gyro_x", "gyro_y", "gyro_z",
    "pitch", "roll",
    "cmd_vel", "cmd_yaw", "cmd_height", "cmd_jump",
    "act_LF", "act_RF", "act_LR", "act_RR", "act_LW", "act_RW",
    "dof_err_LF", "dof_err_RF", "dof_err_LR", "dof_err_RR",
    "dof_vel_LF", "dof_vel_RF", "dof_vel_LR", "dof_vel_RR",
    "dof_vel_LW", "dof_vel_RW",
    "theta_l", "theta_r", "L0_l", "L0_r",
]


def debug_print_obs(obs, step):
    print(f"\n{'='*70}")
    print(f" 观测详情  step={step}")
    print(f"{'='*70}")
    print(f" {'idx':>3s}  {'名称':<18s} {'值':>10s}   方向说明")
    print(f" {'-'*3}  {'-'*18} {'-'*10}   {'-'*40}")
    for idx, name, desc in OBS_LABELS:
        val = obs[idx]
        if idx in (3, 4):
            print(f" [{idx:>2d}]  {name} {val:>+10.4f} rad ({np.rad2deg(val):>+7.2f}°)   {desc}")
        elif idx in (27, 28):
            print(f" [{idx:>2d}]  {name} {val:>+10.4f} m     {desc}")
        else:
            print(f" [{idx:>2d}]  {name} {val:>+10.4f}    {desc}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════
# 推理模型
# ═══════════════════════════════════════════════════════════════

class InferenceModel(nn.Module):
    """encoder + actor 推理包装，接口: forward(obs_history, obs) → action.

    支持两种模式:
      - latent_norm=False: encoder 输出直接 cat obs (新模型, obs_normalization=False)
      - latent_norm=True:  encoder 输出先归一化再 cat obs (旧模型, obs_normalization=True)
    """

    def __init__(self, encoder, actor, vel_scale=3.0,
                 latent_mean=None, latent_std=None, eps=1e-2):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.vel_scale = vel_scale
        # checkpoint 里 latent_mean/std 是 numpy 数组；这里一次性转成 float32 tensor。
        # 若留成 numpy，前向中的张量运算会走 PyTorch 的 numpy 隐式转换路径，
        # 在本机（int8 模型 + cpu）会直接段错误。
        self.latent_mean = (
            None if latent_mean is None else torch.as_tensor(latent_mean, dtype=torch.float32)
        )
        self.latent_std = (
            None if latent_std is None else torch.as_tensor(latent_std, dtype=torch.float32)
        )
        self.eps = eps

    @property
    def use_latent_norm(self):
        return self.latent_mean is not None and self.latent_std is not None

    def forward(self, obs_history, obs):
        latent_raw = self.encoder(obs_history) * self.vel_scale
        if self.use_latent_norm:
            latent = (latent_raw - self.latent_mean) / (self.latent_std + self.eps)
        else:
            latent = latent_raw
        actor_input = torch.cat([obs, latent], dim=-1)
        return self.actor(actor_input), latent_raw


def load_quantized_model(checkpoint_path, device="cpu", ckpt=None):
    """加载 int8 量化 checkpoint 或 float32 PPO checkpoint。

    Returns:
        (InferenceModel, norm_dict)
    """
    if ckpt is None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if ckpt.get("quantized"):
        model = ActorCriticSequence(
            num_obs=NUM_OBS,
            num_critic_obs=NUM_CRITIC_OBS,
            num_actions=NUM_ACTIONS,
            num_encoder_obs=NUM_ENCODER_OBS,
            latent_dim=LATENT_DIM,
            encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
            actor_hidden_dims=ACTOR_HIDDEN_DIMS,
            critic_hidden_dims=CRITIC_HIDDEN_DIMS,
            activation=ACTIVATION,
            obs_normalization=False,
            use_real_vel=False,
        ).to(device)

        if ckpt.get("quantize_dtype") == "qint8_static":
            # 静态量化: 直接加载整个模块
            model.encoder = ckpt["encoder"].to(device)
            model.actor = ckpt["actor"].to(device)
        elif ckpt.get("quantize_dtype") == "qint8_per_channel":
            # 手动 per-channel 量化: 替换 Linear 为 Int8Linear 后加载 state_dict
            _replace_linear_with_int8_calib(model.encoder)
            _replace_linear_with_int8_calib(model.actor)
            model.encoder.load_state_dict(ckpt["encoder_state_dict"])
            model.actor.load_state_dict(ckpt["actor_state_dict"])
        else:
            # 动态量化: 重建并量化后加载 state_dict
            model.encoder = torch.ao.quantization.quantize_dynamic(
                model.encoder, {nn.Linear}, dtype=torch.qint8,
            )
            model.actor = torch.ao.quantization.quantize_dynamic(
                model.actor, {nn.Linear}, dtype=torch.qint8,
            )
            model.encoder.load_state_dict(ckpt["encoder_state_dict"])
            model.actor.load_state_dict(ckpt["actor_state_dict"])

        model.eval()

        inf = InferenceModel(model.encoder, model.actor,
                             vel_scale=ckpt.get("vel_scale", 3.0),
                             latent_mean=ckpt.get("latent_mean"),
                             latent_std=ckpt.get("latent_std"))
        norm = {
            "enc_mean": ckpt["enc_mean"],
            "enc_std": ckpt["enc_std"],
            "obs_mean": ckpt["obs_mean"],
            "obs_std": ckpt["obs_std"],
        }
        return inf, norm
    else:
        # float32 原始 PPO checkpoint
        model_dict = ckpt.get("model_state_dict", ckpt)
        num_critic_obs = model_dict["critic.0.weight"].shape[1]
        model = ActorCriticSequence(
            num_obs=NUM_OBS,
            num_critic_obs=num_critic_obs,
            num_actions=NUM_ACTIONS,
            num_encoder_obs=NUM_ENCODER_OBS,
            latent_dim=LATENT_DIM,
            encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
            actor_hidden_dims=ACTOR_HIDDEN_DIMS,
            critic_hidden_dims=CRITIC_HIDDEN_DIMS,
            activation=ACTIVATION,
            obs_normalization=False,
            use_real_vel=False,
        ).to(device)

        model.load_state_dict(model_dict, strict=False)
        model.eval()

        inf = InferenceModel(model.encoder, model.actor,
                             vel_scale=getattr(model, "vel_scale", 3.0))

        # 从检查点提取训练好的归一化系数
        sd = model_dict
        has_enc_norm = "encoder_obs_normalizer._mean" in sd
        has_act_norm = "actor_obs_normalizer._mean" in sd

        if has_enc_norm and has_act_norm:
            # 旧模型: obs_normalization=True, 有 EmpiricalNormalization
            enc_mean_raw = sd["encoder_obs_normalizer._mean"]
            print(f"[CHECK] encoder_obs_normalizer._mean shape: {enc_mean_raw.shape}")
            if enc_mean_raw.shape[-1] == NUM_ENCODER_OBS:
                enc_mean_per_frame = enc_mean_raw.reshape(1, OBS_HISTORY_LENGTH, NUM_OBS).mean(dim=1).flatten().cpu().numpy()
                enc_std_per_frame = sd["encoder_obs_normalizer._std"].reshape(1, OBS_HISTORY_LENGTH, NUM_OBS).mean(dim=1).flatten().cpu().numpy()
                print(f"[CHECK] 旧格式 ({NUM_ENCODER_OBS}) → 已转换为 {NUM_OBS} ({OBS_HISTORY_LENGTH}帧平均)")
            else:
                enc_mean_per_frame = enc_mean_raw.flatten().cpu().numpy()
                enc_std_per_frame = sd["encoder_obs_normalizer._std"].flatten().cpu().numpy()
                print(f"[CHECK] 新格式 ({NUM_OBS})")
            obs_mean_full = sd["actor_obs_normalizer._mean"].flatten().cpu().numpy()
            obs_std_full = sd["actor_obs_normalizer._std"].flatten().cpu().numpy()

            enc_mean = np.tile(enc_mean_per_frame, OBS_HISTORY_LENGTH)
            enc_std = np.tile(enc_std_per_frame, OBS_HISTORY_LENGTH)
            latent_mean = obs_mean_full[NUM_OBS:].astype(np.float32)
            latent_std = obs_std_full[NUM_OBS:].astype(np.float32)

            inf = InferenceModel(model.encoder, model.actor,
                                 vel_scale=getattr(model, "vel_scale", 3.0),
                                 latent_mean=torch.tensor(latent_mean),
                                 latent_std=torch.tensor(latent_std))
            norm = {
                "enc_mean": enc_mean.astype(np.float32),
                "enc_std": enc_std.astype(np.float32),
                "obs_mean": obs_mean_full[:NUM_OBS].astype(np.float32),
                "obs_std": obs_std_full[:NUM_OBS].astype(np.float32),
            }
            print("[CHECK] 旧模型: 使用 EmpiricalNormalization 参数 + latent 归一化")
        else:
            # 新模型: obs_normalization=False, 无 EmpiricalNormalization
            # 使用 joystick.py 手动归一化参数
            from quantize_model import ENC_MEAN, ENC_STD, OBS_MEAN, OBS_STD
            print("[CHECK] 无 EmpiricalNormalization: 回退到 env 手动归一化参数 "
                  "(要求训练时 obs_manual_norm=true)")
            inf = InferenceModel(model.encoder, model.actor,
                                 vel_scale=getattr(model, "vel_scale", 3.0))
            norm = {
                "enc_mean": ENC_MEAN,
                "enc_std": ENC_STD,
                "obs_mean": OBS_MEAN,
                "obs_std": OBS_STD,
            }
        return inf, norm


def _apply_ckpt_dims(ckpt):
    """按 checkpoint 真实维度刷新模块级常量，并对不支持的布局给出明确错误。"""
    global NUM_OBS, NUM_ACTIONS, OBS_HISTORY_LENGTH, NUM_ENCODER_OBS, LATENT_DIM
    global NUM_CRITIC_OBS, ENCODER_HIDDEN_DIMS, ACTOR_HIDDEN_DIMS, CRITIC_HIDDEN_DIMS

    try:
        dims = infer_dims_from_ckpt(ckpt)
    except Exception as exc:  # noqa: BLE001 - 转成面向用户的错误
        raise SystemExit(f"[ERROR] 无法从 checkpoint 推断模型维度: {exc}") from exc

    NUM_OBS = dims["num_obs"]
    NUM_ACTIONS = dims["num_actions"]
    OBS_HISTORY_LENGTH = dims["obs_history_length"]
    NUM_ENCODER_OBS = dims["num_encoder_obs"]
    LATENT_DIM = dims["latent_dim"]
    ENCODER_HIDDEN_DIMS = dims["encoder_hidden_dims"]
    ACTOR_HIDDEN_DIMS = dims["actor_hidden_dims"]
    CRITIC_HIDDEN_DIMS = dims["critic_hidden_dims"]
    # ACTIVATION 无法从 state_dict 推断，保留 trained_params 中训练 config 的值。
    if dims.get("num_critic_obs"):
        NUM_CRITIC_OBS = dims["num_critic_obs"]

    print(f"[INFER] ckpt: obs={NUM_OBS}, encoder_obs={NUM_ENCODER_OBS}, "
          f"history={OBS_HISTORY_LENGTH}, latent={LATENT_DIM}, actions={NUM_ACTIONS}")

    if NUM_OBS != deploy_obs.NUM_OBS:
        raise SystemExit(
            f"[ERROR] 该 checkpoint 是 {NUM_OBS} 维观测, 本测试脚本只支持 "
            f"{deploy_obs.NUM_OBS} 维布局 (见 deploy_obs.py)。\n"
            "obs_vel / 观测布局不同的模型需要先扩展 deploy_obs.build_observation。"
        )


def _resolve_float_ckpt(model_file, ckpt, explicit=None):
    """为 --compare 找到真正的 float32 baseline。

    优先显式 ``--float_model``；否则按量化器复制到同目录的
    ``model_XXXX.pt`` 查找（``model_XXXX_int8.pt`` → ``model_XXXX.pt``）。
    """
    if explicit:
        path = explicit if os.path.isabs(explicit) else os.path.join(_PARENT_DIR, explicit)
        return os.path.normpath(path)
    if not ckpt.get("quantized"):
        return model_file
    folder = os.path.dirname(model_file)
    stem, ext = os.path.splitext(os.path.basename(model_file))
    for suffix in ("_int8_static", "_int8"):
        if stem.endswith(suffix):
            candidate = os.path.join(folder, stem[: -len(suffix)] + ext)
            if os.path.isfile(candidate):
                return candidate
    return None


def normalize_obs(obs_history, obs, norm, device):
    """归一化观测并转为 tensor (与 env + EmpiricalNormalization 一致)。

    env 在归一化前会 ``clip(obs, -100, 100)``，这里保持一致。
    """
    eps = deploy_obs.NORM_EPS
    oh_t = torch.tensor(obs_history, dtype=torch.float32).unsqueeze(0).to(device)
    o_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    oh_t = torch.clamp(oh_t, -100.0, 100.0)
    o_t = torch.clamp(o_t, -100.0, 100.0)
    enc_mean_t = torch.tensor(norm["enc_mean"], dtype=torch.float32).to(device)
    enc_std_t = torch.tensor(norm["enc_std"], dtype=torch.float32).to(device)
    obs_mean_t = torch.tensor(norm["obs_mean"], dtype=torch.float32).to(device)
    obs_std_t = torch.tensor(norm["obs_std"], dtype=torch.float32).to(device)

    oh_n = (oh_t - enc_mean_t) / (enc_std_t + eps)
    o_n = (o_t - obs_mean_t) / (obs_std_t + eps)
    return oh_n, o_n


def compute_vmc_feedforward(dof_pos, pitch, default_dof_pos, force=None):
    """VMC 重力前馈力矩 (与 joystick.py _compute_vmc_torques 一致).

    纯前馈: F = feedforward_force / cos(theta) * ramp, T = 0
    通过雅可比映射到关节力矩: tau = J^T * [F, T]

    Args:
        force: 前馈力大小, 默认使用 VMC_FEEDFORWARD_FORCE (80N)
    """
    phi1_l = (dof_pos[2] - default_dof_pos[2]) + 5 * np.pi / 6
    phi1_r = (dof_pos[3] - default_dof_pos[3]) + 5 * np.pi / 6
    phi4_l = (dof_pos[0] - default_dof_pos[0]) + np.pi / 6
    phi4_r = (dof_pos[1] - default_dof_pos[1]) + np.pi / 6

    L0_l, phi0_l, phi2_l, phi3_l, sigma1_l = fk_fivebar(phi1_l, phi4_l)
    L0_r, phi0_r, phi2_r, phi3_r, sigma1_r = fk_fivebar(phi1_r, phi4_r)

    theta_l = (np.pi / 2 - phi0_l + pitch + np.pi) % (2.0 * np.pi) - np.pi
    theta_r = (np.pi / 2 - phi0_r + pitch + np.pi) % (2.0 * np.pi) - np.pi

    theta_abs_l = abs(theta_l)
    theta_abs_r = abs(theta_r)
    ramp_l = np.clip((np.pi / 3 - theta_abs_l) / (np.pi / 6), 0.0, 1.0)
    ramp_r = np.clip((np.pi / 3 - theta_abs_r) / (np.pi / 6), 0.0, 1.0)

    _force = force if force is not None else VMC_FEEDFORWARD_FORCE
    ff_l = _force / np.cos(np.clip(theta_l, -1.5, 1.5)) * ramp_l
    ff_r = _force / np.cos(np.clip(theta_r, -1.5, 1.5)) * ramp_r

    sigma2_l = np.sin(phi3_l - phi4_l)
    sigma3_l = np.sin(phi1_l - phi2_l)
    sigma2_r = np.sin(phi3_r - phi4_r)
    sigma3_r = np.sin(phi1_r - phi2_r)

    eps = 1e-8
    inv_sigma1_l = 1.0 / (sigma1_l + eps)
    inv_sigma1_r = 1.0 / (sigma1_r + eps)

    J00_l = VMC_L1 * np.sin(phi0_l - phi3_l) * sigma3_l * inv_sigma1_l
    J10_l = VMC_L4 * np.sin(phi0_l - phi2_l) * sigma2_l * inv_sigma1_l
    J00_r = VMC_L1 * np.sin(phi0_r - phi3_r) * sigma3_r * inv_sigma1_r
    J10_r = VMC_L4 * np.sin(phi0_r - phi2_r) * sigma2_r * inv_sigma1_r

    tau1_l = J00_l * ff_l
    tau4_l = J10_l * ff_l
    tau1_r = J00_r * ff_r
    tau4_r = J10_r * ff_r

    vmc = np.zeros(6, dtype=np.float64)
    vmc[0] = tau4_l
    vmc[1] = tau4_r
    vmc[2] = tau1_l
    vmc[3] = tau1_r
    vmc = np.nan_to_num(vmc, nan=0.0, posinf=40.0, neginf=-40.0)
    return vmc


def vmc_action_to_torque(raw_action, dof_pos, dof_vel, pitch, default_dof_pos):
    """VMC 动作 → 关节力矩 (与 joystick_vmc.py apply_action + _pre_step_motor_control 一致).

    动作布局: [ΔL0_L, Δθ_L, ΔL0_R, Δθ_R, ω_LW, ω_RW]
    1. 动作 → 虚拟参考量: L0_ref, θ_ref, wheel_vel_ref
    2. FK 计算当前 L0, θ
    3. 虚拟空间 PD → 五连杆雅可比 → 关节力矩
    4. 轮子速度 PD
    """
    a = np.asarray(raw_action, dtype=np.float64)

    # 1. 动作 → 虚拟参考量
    l0_ref = np.clip(VMC_DEFAULT_L0 + a[[0, 2]] * VMC_ACTION_SCALE_L0,
                     VMC_L0_CLIP[0], VMC_L0_CLIP[1])
    theta_ref = np.clip(a[[1, 3]] * VMC_ACTION_SCALE_THETA,
                        -VMC_THETA_CLIP, VMC_THETA_CLIP)

    # 2. FK 计算当前虚拟状态
    cal_lf = dof_pos[0] - default_dof_pos[0]
    cal_rf = dof_pos[1] - default_dof_pos[1]
    cal_lr = dof_pos[2] - default_dof_pos[2]
    cal_rr = dof_pos[3] - default_dof_pos[3]

    phi1_l = cal_lr + 5 * np.pi / 6
    phi4_l = cal_lf + np.pi / 6
    phi1_r = cal_rr + 5 * np.pi / 6
    phi4_r = cal_rf + np.pi / 6

    L0_l, phi0_l, phi2_l, phi3_l, sigma1_l = fk_fivebar(phi1_l, phi4_l)
    L0_r, phi0_r, phi2_r, phi3_r, sigma1_r = fk_fivebar(phi1_r, phi4_r)

    theta_l = (np.pi / 2 - phi0_l + pitch + np.pi) % (2.0 * np.pi) - np.pi
    theta_r = (np.pi / 2 - phi0_r + pitch + np.pi) % (2.0 * np.pi) - np.pi

    # 虚拟腿长变化率 (用雅可比)
    J00_l, J01_l, _, _ = compute_jacobian_partials(phi1_l, phi4_l, phi0_l, phi2_l, phi3_l, sigma1_l, L0_l)
    J00_r, J01_r, _, _ = compute_jacobian_partials(phi1_r, phi4_r, phi0_r, phi2_r, phi3_r, sigma1_r, L0_r)
    L0_dot_l = J00_l * dof_vel[2] + J01_l * dof_vel[0]  # φ₁=LR, φ₄=LF
    L0_dot_r = J00_r * dof_vel[3] + J01_r * dof_vel[1]  # φ₁=RR, φ₄=RF

    # φ₀ 变化率同理
    _, _, J10_l, J11_l = compute_jacobian_partials(phi1_l, phi4_l, phi0_l, phi2_l, phi3_l, sigma1_l, L0_l)
    _, _, J10_r, J11_r = compute_jacobian_partials(phi1_r, phi4_r, phi0_r, phi2_r, phi3_r, sigma1_r, L0_r)
    phi0_dot_l = J10_l * dof_vel[2] + J11_l * dof_vel[0]
    phi0_dot_r = J10_r * dof_vel[3] + J11_r * dof_vel[1]
    theta_dot_l = -phi0_dot_l
    theta_dot_r = -phi0_dot_r

    # 3. 虚拟空间 PD → 力/力矩
    theta_world = np.array([theta_l, theta_r])
    theta_dot = np.array([theta_dot_l, theta_dot_r])
    L0 = np.array([L0_l, L0_r])

    Tp = VMC_KP_THETA * (theta_world - theta_ref) + VMC_KD_THETA * theta_dot
    F = (VMC_KP_L0 * (l0_ref - L0) - VMC_KD_L0 * np.array([L0_dot_l, L0_dot_r]))

    # 重力前馈: F_ff = force / cos(θ) * ramp (与 joystick_vmc.py 一致;
    # 常量 80N 在 |θ|>0.5 时偏差可达 -44N)
    theta_abs = np.abs(theta_world)
    ramp = np.clip((np.pi / 3.0 - theta_abs) / (np.pi / 6.0), 0.0, 1.0)
    feedforward = VMC_FEEDFORWARD_FORCE / np.cos(np.clip(theta_world, -1.5, 1.5)) * ramp
    F = F + feedforward

    # 五连杆雅可比 → 关节力矩
    sigma2_l = np.sin(phi3_l - phi4_l)
    sigma3_l = np.sin(phi1_l - phi2_l)
    sigma2_r = np.sin(phi3_r - phi4_r)
    sigma3_r = np.sin(phi1_r - phi2_r)

    eps = 1e-8
    inv_sigma1_l = 1.0 / (sigma1_l + eps)
    inv_sigma1_r = 1.0 / (sigma1_r + eps)

    tau1_l = VMC_L1 * np.sin(phi0_l - phi3_l) * sigma3_l * inv_sigma1_l * F[0]
    tau4_l = VMC_L4 * np.sin(phi0_l - phi2_l) * sigma2_l * inv_sigma1_l * F[0]
    tau1_r = VMC_L1 * np.sin(phi0_r - phi3_r) * sigma3_r * inv_sigma1_r * F[1]
    tau4_r = VMC_L4 * np.sin(phi0_r - phi2_r) * sigma2_r * inv_sigma1_r * F[1]

    # Tp 映射同理
    tau1_l += VMC_L1 * np.cos(phi0_l - phi3_l) * sigma3_l * inv_sigma1_l * (1.0 / (L0_l + eps)) * Tp[0]
    tau4_l += VMC_L4 * np.cos(phi0_l - phi2_l) * sigma2_l * inv_sigma1_l * (1.0 / (L0_l + eps)) * Tp[0]
    tau1_r += VMC_L1 * np.cos(phi0_r - phi3_r) * sigma3_r * inv_sigma1_r * (1.0 / (L0_r + eps)) * Tp[1]
    tau4_r += VMC_L4 * np.cos(phi0_r - phi2_r) * sigma2_r * inv_sigma1_r * (1.0 / (L0_r + eps)) * Tp[1]

    tau = np.zeros(6, dtype=np.float64)
    tau[0] = tau4_l  # τ₄ → LF
    tau[1] = tau4_r  # τ₄ → RF
    tau[2] = tau1_l  # τ₁ → LR
    tau[3] = tau1_r  # τ₁ → RR

    # 4. 轮子速度 PD
    lw_target = a[4] * ACTION_SCALE_WHEEL
    rw_target = a[5] * ACTION_SCALE_WHEEL
    tau[4] = WHEEL_KD[0] * (lw_target - dof_vel[4])
    tau[5] = WHEEL_KD[1] * (rw_target - dof_vel[5])

    tau = np.nan_to_num(tau, nan=0.0, posinf=40.0, neginf=-40.0)
    return tau


def raw_action_to_torque(raw_action, dof_pos, dof_vel, default_dof_pos, vmc_torques=None):
    """RL 原始动作 → 关节力矩 (与 joystick.py apply_action + _pre_step_motor_control 一致).

    腿关节: target = action * action_scale_joint + default_dof_pos, clip to limits
             torque = kp * (target - pos) - kd * vel
    轮关节: target_vel = action * action_scale_wheel
             torque = wheel_kd * (target_vel - vel)
    VMC 前馈: 叠加重力补偿力矩 (与训练环境一致)
    """
    lf_target = raw_action[0] * ACTION_SCALE_JOINT + default_dof_pos[0]
    rf_target = raw_action[1] * ACTION_SCALE_JOINT + default_dof_pos[1]
    lr_target = raw_action[2] * ACTION_SCALE_JOINT + default_dof_pos[2]
    rr_target = raw_action[3] * ACTION_SCALE_JOINT + default_dof_pos[3]
    lw_target = raw_action[4] * ACTION_SCALE_WHEEL
    rw_target = raw_action[5] * ACTION_SCALE_WHEEL

    lf_target = np.clip(lf_target, JOINT_LIMITS[0][0], JOINT_LIMITS[0][1])
    rf_target = np.clip(rf_target, JOINT_LIMITS[1][0], JOINT_LIMITS[1][1])
    lr_target = np.clip(lr_target, JOINT_LIMITS[2][0], JOINT_LIMITS[2][1])
    rr_target = np.clip(rr_target, JOINT_LIMITS[3][0], JOINT_LIMITS[3][1])

    tau_lf = LEG_KP[0] * (lf_target - dof_pos[0]) - LEG_KD[0] * dof_vel[0]
    tau_rf = LEG_KP[1] * (rf_target - dof_pos[1]) - LEG_KD[1] * dof_vel[1]
    tau_lr = LEG_KP[2] * (lr_target - dof_pos[2]) - LEG_KD[2] * dof_vel[2]
    tau_rr = LEG_KP[3] * (rr_target - dof_pos[3]) - LEG_KD[3] * dof_vel[3]
    tau_lw = WHEEL_KD[0] * (lw_target - dof_vel[4])
    tau_rw = WHEEL_KD[1] * (rw_target - dof_vel[5])

    tau = np.array([tau_lf, tau_rf, tau_lr, tau_rr, tau_lw, tau_rw], dtype=np.float64)

    if vmc_torques is not None:
        tau[:4] += vmc_torques[:4]

    return tau


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="加载量化 PPO 策略，仿真测试")
    parser.add_argument("--model", type=str, required=True,
                        help="模型 checkpoint 路径 (.pt, 支持 int8 量化或 float32 PPO)")
    parser.add_argument("--scene", type=str, default="rmuc",
                        help="场景 (rmuc/flat/step/rough)")
    parser.add_argument("--cmd_vel", type=float, default=0.0,
                        help="前进速度命令 m/s (默认 0=原地站; 测台阶/斜坡需要 >0)")
    parser.add_argument("--cmd_height", type=float, default=0.29,
                        help="目标高度命令 m (默认 0.29)")
    parser.add_argument("--spawn_x", type=float, default=None,
                        help="起始 x 覆盖 (scene_step 的台阶在 x=6, 用 3.0 可直接站到台阶前)")
    parser.add_argument("--spawn_y", type=float, default=None,
                        help="起始 y 覆盖")
    parser.add_argument("--auto_jump", action="store_true",
                        help="自动跳跃: 前方测距 < --auto_jump_dist 时自动发 cmd_jump=1")
    parser.add_argument("--auto_jump_dist", type=float, default=0.40,
                        help="自动跳跃触发距离 m (默认 0.40)")
    parser.add_argument("--auto_jump_cooldown", type=float, default=1.5,
                        help="两次自动跳跃的最小间隔 s (默认 1.5)")
    parser.add_argument("--headless", action="store_true",
                        help="不开查看器，跑固定步数后直接输出对比统计（无显示环境用）")
    parser.add_argument("--headless_steps", type=int, default=1000,
                        help="--headless 下运行的物理步数 (默认 1000)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="计算设备 (cpu/cuda)")
    parser.add_argument("--obs_noise_level", type=float, default=1.0,
                        help="观测噪声倍率: 1.0=训练同分布, 0=关闭")
    parser.add_argument("--obs_noise_seed", type=int, default=None,
                        help="观测噪声随机种子 (默认不固定)")
    parser.add_argument("--compare", action="store_true",
                        help="对比模式: 同时推理 float32 和 int8, 统计差异")
    parser.add_argument("--float_model", type=str, default=None,
                        help="对比模式的 float32 baseline (默认: 同目录 model_XXXX.pt)")
    parser.add_argument("--no_quantize", action="store_true",
                        help="加载 float32 PPO checkpoint, 不量化直接测试")
    parser.add_argument("--compare_obs", type=str, default=None,
                        help=f"对比模式: 输入 Play 打印的 raw obs ({NUM_OBS}个逗号分隔的数值), 可选 ';' 后跟 play action")
    parser.add_argument("--action_smooth", type=float, default=0.0,
                        help="动作滤波 (旧值权重, 0=不过滤, 越大越平滑, 建议 0.5~0.8)")
    parser.add_argument("--debug_obs", type=int, default=0,
                        help="每隔 N 个控制步打印观测详情 (含名称/数值/方向说明), 0=关闭")
    parser.add_argument("--init_dof_pos", type=float, nargs=4, default=None,
                        help="覆盖初始 4 个腿关节角度 (deg): LF RF LR RR")
    parser.add_argument("--hw_default_offset", type=float, default=0.0,
                        help="实物标定: LR/RR 的 default_dof_pos 偏移 (rad, 训练默认为 0)")
    args = parser.parse_args()

    global _CONTROL_MODE
    _CONTROL_MODE = detect_control_mode(args.model)
    print(f"[控制模式] {_CONTROL_MODE.upper()} ({'虚拟模型控制' if _CONTROL_MODE == 'vmc' else '关节位置PD'})")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用, 回退到 CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    # ── 加载场景 ──
    scene_dir = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..",
                             "assets", "robots", "balance_infantry")
    scene_map = {
        "rmuc": os.path.join(scene_dir, "scene_rmuc.xml"),
        "flat": os.path.join(scene_dir, "scene_flat.xml"),
        # 这两个场景的 home keyframe 带五连杆站立姿态；rmuc/flat 的 qpos 关节全 0，
        # 直接拿来测试会让机器人从直腿姿态起步并摔倒（与 quantize_model 保持一致）。
        "step": os.path.join(scene_dir, "scene_step.xml"),
        "rough": os.path.join(scene_dir, "scene_rough.xml"),
    }
    scene_path = scene_map.get(args.scene, scene_map["rmuc"])
    print(f"加载场景: {scene_path}")

    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(mj_model)

    # ── 关节索引 ──
    joint_names = [
        "left_front_joint", "right_front_joint",
        "left_rear_joint", "right_rear_joint",
        "left_wheel_joint", "right_wheel_joint",
    ]
    joint_qpos = [mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                  for n in joint_names]
    joint_dofadr = [mj_model.jnt_dofadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                    for n in joint_names]
    act_names = [n + "_actuator" for n in joint_names]
    act_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]

    # ── 加载模型 ──
    model_file = args.model
    if not os.path.isabs(model_file):
        model_file = os.path.join(_PARENT_DIR, model_file)
    warn_if_trained_params_mismatch(model_file)
    print(f"加载模型: {model_file}")

    ckpt = torch.load(model_file, map_location=device, weights_only=False)
    _apply_ckpt_dims(ckpt)
    inf_model, norm = load_quantized_model(model_file, device, ckpt=ckpt)
    is_quantized = bool(ckpt.get("quantized"))
    if args.no_quantize and is_quantized:
        print("[WARN] 传入的是量化 checkpoint, --no_quantize 已无作用")

    print("  actor_obs_normalizer (VMC相关维度):")
    for i, name in [(25, "theta_l"), (26, "theta_r"), (27, "L0_l"), (28, "L0_r")]:
        print(f"    {name}: mean={norm['obs_mean'][i]:.6f}, std={norm['obs_std'][i]:.6f}")
    print(f"    theta_l-theta_r mean diff: {norm['obs_mean'][25] - norm['obs_mean'][26]:.6f}")
    print(f"    L0_l-L0_r mean diff: {norm['obs_mean'][27] - norm['obs_mean'][28]:.6f}")

    # ── DEBUG: 打印 actor 输出层 bias ──
    print("\n  Actor 输出层 bias (最后一层 Linear bias):")
    actor = inf_model.actor
    last_linear = None
    for module in actor.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        bias = last_linear.bias.data.cpu().numpy()
        act_bias_names = ["LF", "RF", "LR", "RR", "LW", "RW"]
        for i, name in enumerate(act_bias_names):
            print(f"    {name}: {bias[i]:+.6f}")
        print(f"    LF-RF bias diff: {bias[0] - bias[1]:+.6f}")
        print(f"    LR-RR bias diff: {bias[2] - bias[3]:+.6f}")

    # ── DEBUG: 打印所有 obs 维度的归一化参数 ──
    print("\n  所有 obs 维度归一化参数 (mean, std):")
    obs_label_names = list(OBS_NAMES)
    for i in range(NUM_OBS):
        print(f"    [{i:2d}] {obs_label_names[i]:15s}  mean={norm['obs_mean'][i]:+.6f}  std={norm['obs_std'][i]:+.6f}")

    # 对比模式下额外加载 float32 模型（必须真的是 float32，不能是同一个 int8 文件）
    float_inf = None
    if args.compare:
        if not is_quantized:
            print("[WARN] 传入的不是量化模型, --compare 只是同一权重的自比较")
        float_path = _resolve_float_ckpt(model_file, ckpt, args.float_model)
        if float_path is None:
            raise SystemExit(
                "[ERROR] --compare 找不到 float32 baseline。\n"
                "量化器会把原始 checkpoint 复制到输出目录；请确认同目录存在 "
                "model_XXXX.pt，或显式传 --float_model <path>。"
            )
        if not os.path.isfile(float_path):
            raise SystemExit(f"[ERROR] float baseline 不存在: {float_path}")
        float_ckpt = torch.load(float_path, map_location=device, weights_only=False)
        if float_ckpt.get("quantized"):
            raise SystemExit(
                f"[ERROR] --compare 的 baseline 仍是量化模型 ({float_path})，"
                "这样的对比没有意义"
            )
        print(f"对比模式: float32 baseline = {float_path}")
        float_inf, _ = load_quantized_model(float_path, device, ckpt=float_ckpt)
        print("  int8 模型已加载, float32 对比模型已加载 ✓")

    tag = "int8" if is_quantized else "f32"
    print(f"  模型类型: {tag}, 设备: {args.device}")
    print(f"  观测噪声: level={args.obs_noise_level:g} "
          f"({'训练同分布' if args.obs_noise_level > 0 else '关闭'}), "
          f"seed={args.obs_noise_seed}")

    # ── 初始状态 (优先使用场景自带 keyframe, 不再无脑覆盖高度) ──
    has_keyframe = False
    try:
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
            has_keyframe = True
    except Exception:
        pass
    if not has_keyframe:
        mj_data.qpos[2] = 0.8

    # ── 起始位置覆盖：方便直接把机器人放到台阶/斜坡前 ──
    if args.spawn_x is not None:
        mj_data.qpos[0] = float(args.spawn_x)
    if args.spawn_y is not None:
        mj_data.qpos[1] = float(args.spawn_y)
    if args.spawn_x is not None or args.spawn_y is not None:
        mujoco.mj_forward(mj_model, mj_data)
        print(f"  ⚡ 起始位置: x={mj_data.qpos[0]:.2f} y={mj_data.qpos[1]:.2f} z={mj_data.qpos[2]:.2f}")

    mj_data.qvel[0:6] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    mj_model.qpos0 = mj_data.qpos.copy()

    # 保存初始姿态，reset 时恢复
    init_qpos = mj_data.qpos.copy()
    init_qvel = mj_data.qvel.copy()

    # ── 覆盖初始关节角度 (--init_dof_pos LF RF LR RR, 单位: deg) ──
    custom_dof_pos = None
    if args.init_dof_pos is not None:
        init_deg = np.array(args.init_dof_pos, dtype=np.float64)
        custom_dof_pos = np.deg2rad(init_deg)
        for i in range(4):
            mj_data.qpos[joint_qpos[i]] = custom_dof_pos[i]
        mujoco.mj_forward(mj_model, mj_data)
        mj_model.qpos0 = mj_data.qpos.copy()
        init_qpos = mj_data.qpos.copy()
        print(f"  ⚡ 覆盖初始关节角度: {init_deg}°")

    # 从仿真动态读取默认关节角度 (与 joystick.py _default_dof_pos 一致)
    default_dof_pos = np.array([mj_data.qpos[joint_qpos[i]] for i in range(4)], dtype=np.float32)
    if args.hw_default_offset != 0.0:
        # 实物后电机标定偏移; 训练侧 default 是 0, 默认不要打开
        default_dof_pos[2] -= args.hw_default_offset  # LR
        default_dof_pos[3] -= args.hw_default_offset  # RR
        print(f"  [WARN] 应用实物 default 偏移 {args.hw_default_offset:+.3f} rad (LR/RR), "
              "仅在真机对齐时使用")
    print(f"  default_dof_pos = {np.rad2deg(default_dof_pos)}°")

    # ── 自动跳跃: 前方测距传感器 ──
    # robot.xml 里已有 base_height_rangefinder (site 在 base 前方 0.30m, zaxis=+x),
    # 沿机体 +x 打一条射线, 命中前方障碍就返回距离, 没命中返回负值。
    rf_adr = None
    _rf_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "base_height_rangefinder")
    if _rf_sid >= 0:
        rf_adr = int(mj_model.sensor_adr[_rf_sid])
    else:
        print("  ⚠ 该场景没有 base_height_rangefinder，H 键自动跳跃扫描不可用")
    # H 键切换的自动跳跃扫描开关（--auto_jump 只决定初始状态）
    auto_jump_on = [bool(args.auto_jump)]
    if rf_adr is not None:
        print(f"  自动跳跃扫描: {'ON' if auto_jump_on[0] else 'OFF'} (H 键切换)  "
              f"阈值 {args.auto_jump_dist:.2f} m  冷却 {args.auto_jump_cooldown:.1f}s")

    # ── 状态变量 ──
    obs_history = deque(maxlen=OBS_HISTORY_LENGTH)
    last_actions = np.zeros(6, dtype=np.float32)
    raw_action = np.zeros(6, dtype=np.float64)
    raw_action_f32 = np.zeros(6, dtype=np.float64)
    filtered_action = np.zeros(6, dtype=np.float64)
    filtered_action_f32 = np.zeros(6, dtype=np.float64)
    action_fifo = np.zeros((ACTION_DELAY + 1, 6), dtype=np.float64)
    action_fifo_f32 = np.zeros((ACTION_DELAY + 1, 6), dtype=np.float64)
    pred_vel = 0.0
    cmd_vel = [float(args.cmd_vel)]
    cmd_yaw_rate = [0.0]
    cmd_height = [float(args.cmd_height)]
    target_yaw = [float(np.arctan2(2.0 * (mj_data.qpos[3] * mj_data.qpos[6] + mj_data.qpos[4] * mj_data.qpos[5]),
                                   1.0 - 2.0 * (mj_data.qpos[5] * mj_data.qpos[5] + mj_data.qpos[6] * mj_data.qpos[6])))]
    step_count = [0]
    ctrl_step_count = [0]
    paused = [False]
    debug_flag = [False]
    use_nn = [True]
    need_reset = [False]
    jump_timer = [0.0]
    cmd_jump = [0.0]
    auto_jump_cd = [0.0]
    rf_log_timer = [0.0]
    max_base_z = [0.0]
    episode_steps = [0]
    powerless_active = [False]
    powerless_steps = int(POWERLESS_DURATION / CTRL_DT)

    # ── PD 控制实物位姿 ──
    pd_active = [False]
    pd_target = PD_TARGET.copy()
    pd_kp = 80.0
    pd_kd = 1.5
    _x11 = None
    _x11_display = None
    _x11_root = None
    _x11_screen_w = 0
    _x11_screen_h = 0
    _xtst = None
    try:
        _x11 = ctypes.CDLL("libX11.so.6")
        _x11.XOpenDisplay.restype = ctypes.c_void_p
        _x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        _x11_display = _x11.XOpenDisplay(None)
        if _x11_display:
            _x11.XDefaultRootWindow.restype = ctypes.c_void_p
            _x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
            _x11_root = _x11.XDefaultRootWindow(_x11_display)
            _x11.XDefaultScreen.restype = ctypes.c_int
            _x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
            _screen = _x11.XDefaultScreen(_x11_display)
            _x11.XDisplayWidth.restype = ctypes.c_int
            _x11.XDisplayWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
            _x11_screen_w = _x11.XDisplayWidth(_x11_display, _screen)
            _x11.XDisplayHeight.restype = ctypes.c_int
            _x11.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
            _x11_screen_h = _x11.XDisplayHeight(_x11_display, _screen)
            _x11.XQueryPointer.restype = ctypes.c_int
            _x11.XQueryPointer.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_uint),
            ]
            _x11.XFlush.restype = ctypes.c_int
            _x11.XFlush.argtypes = [ctypes.c_void_p]
        _xtst = ctypes.CDLL("libXtst.so.6")
        _xtst.XTestFakeMotionEvent.restype = ctypes.c_int
        _xtst.XTestFakeMotionEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong,
        ]
    except Exception:
        _x11_display = None

    mouse_yaw_active = [False]
    mouse_last_x = [0.0]
    mouse_last_y = [0.0]
    mouse_yaw_sensitivity = 0.015
    mouse_yaw_decay = 0.85

    # ── 对比统计 ──
    compare_stats = {
        "max_action_diff": 0.0,
        "max_torque_diff": 0.0,
        "sum_action_diff": 0.0,
        "sum_torque_diff": 0.0,
        "count": 0,
    }

    # ── 对比 Play 观测模式 ──
    if args.compare_obs is not None:
        print("\n" + "=" * 60)
        print("[COMPARE_OBS] 对比 Play 观测与 Test 动作")
        print("=" * 60)
        parts = args.compare_obs.split(";")
        obs_raw = np.array([float(x) for x in parts[0].split(",")], dtype=np.float32)
        if len(obs_raw) == NUM_OBS:
            print(f"Play obs_raw (1帧, {NUM_OBS}d): {obs_raw}")
            obs_hist = np.tile(obs_raw, OBS_HISTORY_LENGTH).astype(np.float32)
        elif len(obs_raw) == NUM_ENCODER_OBS:
            print(f"Play obs_hist ({OBS_HISTORY_LENGTH}帧, {NUM_ENCODER_OBS}d), 最后1帧: {obs_raw[-NUM_OBS:]}")
            obs_hist = obs_raw.astype(np.float32)
        else:
            print(f"ERROR: obs 维度 {len(obs_raw)}，期望 {NUM_OBS} (1帧) 或 {NUM_ENCODER_OBS} ({OBS_HISTORY_LENGTH}帧)")
            sys.exit(1)

        play_act = None
        if len(parts) >= 2:
            play_act = np.array([float(x) for x in parts[1].split(",")], dtype=np.float64)
            print(f"Play action: {play_act}")

        oh_n, o_n = normalize_obs(obs_hist, obs_hist[-NUM_OBS:], norm, device)
        print(f"Test enc_obs_norm (last {NUM_OBS}d): {oh_n[0, -NUM_OBS:].cpu().numpy()}")
        print(f"Test obs_norm ({NUM_OBS}d): {o_n[0].cpu().numpy()}")

        with torch.no_grad():
            raw_action_test, latent = inf_model(oh_n, o_n)
            raw_action_test = raw_action_test.cpu().numpy()[0]
            if raw_action_test.ndim == 2:
                raw_action_test = raw_action_test[0]
            latent_np = latent.cpu().numpy()[0] if latent.ndim == 2 else latent.cpu().numpy()

        print(f"\nTest action (raw, no clip): {raw_action_test}")
        if play_act is not None:
            print(f"Play action:                 {play_act}")
            diff = raw_action_test - play_act
            print(f"Diff (Test - Play):          {diff}")
            print(f"Max abs diff: {np.max(np.abs(diff)):.6f}")
        print(f"Latent: {latent_np}")
        print("=" * 60)
        sys.exit(0)

    mj_model.opt.timestep = SIM_DT
    print(f"  sim_dt={mj_model.opt.timestep:.4f}s, ctrl_dt={CTRL_DT:.3f}s, ctrl_steps={CTRL_STEPS}")

    # 增加腿连杆质量 (模拟真机腿部惯量偏大)
    leg_body_names = [
        "left_front_link", "left_front_child1_link", "left_front_child2_link", "left_front_child3_link",
        "left_rear_link", "left_rear_child1_link",
        "right_front_link", "right_front_child1_link", "right_front_child2_link", "right_front_child3_link",
        "right_rear_link", "right_rear_child1_link",
    ]
    MASS_SCALE = 1.0
    for name in leg_body_names:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            old = mj_model.body_mass[bid]
            mj_model.body_mass[bid] *= MASS_SCALE
            print(f"    {name}: mass {old:.4f} → {mj_model.body_mass[bid]:.4f} (×{MASS_SCALE})")

    print(f"\n  {'='*60}")
    print(f"  PPO 策略测试 ({tag})")
    print(f"  场景: {args.scene}  |  设备: {args.device}")
    if args.compare:
        print("  对比模式: float32 vs int8")
    print("  N=切换NN  Space=暂停  ↑↓=vx±0.5  ←→=vyaw±0.2  Enter=归零  M=鼠标偏航  H=自动跳扫描  J=手动跳  R/T=高度  P=PD实物位姿  Backspace=重置  Esc=退出")
    print(f"  {'='*60}\n")

    obs_noise_rng = np.random.default_rng(args.obs_noise_seed)

    def reset_sim():
        """重置仿真到初始状态"""
        nonlocal pred_vel
        mj_data.qpos[:] = init_qpos
        mj_data.qvel[:] = init_qvel
        mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        step_count[0] = 0
        ctrl_step_count[0] = 0
        last_actions[:] = 0.0
        raw_action[:] = 0.0
        raw_action_f32[:] = 0.0
        filtered_action[:] = 0.0
        filtered_action_f32[:] = 0.0
        action_fifo[:] = 0.0
        action_fifo_f32[:] = 0.0
        pred_vel = 0.0
        obs_history.clear()
        cmd_vel[0] = float(args.cmd_vel)
        cmd_yaw_rate[0] = 0.0
        cmd_height[0] = float(args.cmd_height)
        jump_timer[0] = 0.0
        cmd_jump[0] = 0.0
        auto_jump_cd[0] = 0.0
        rf_log_timer[0] = 0.0
        episode_steps[0] = 0
        powerless_active[0] = np.random.random() < POWERLESS_PROBABILITY
        qw, qx, qy, qz = mj_data.qpos[3:7]
        target_yaw[0] = float(np.arctan2(2.0 * (qw * qz + qx * qy),
                                         1.0 - 2.0 * (qy * qy + qz * qz)))
        status = "[无力期 ON]" if powerless_active[0] else ""
        print(f">>> 重置 <<< {status}")

    def controller(model, data):
        nonlocal last_actions, raw_action, raw_action_f32, filtered_action, filtered_action_f32, action_fifo, action_fifo_f32, pred_vel, jump_timer, cmd_jump, episode_steps, powerless_active
        dt = model.opt.timestep
        phys_step = step_count[0]
        step_count[0] += 1

        # ── 鼠标偏航控制: 每帧轮询鼠标水平位移, 映射到 cmd_yaw_rate ──
        if mouse_yaw_active[0] and _x11_display is not None:
            try:
                root_ret = ctypes.c_void_p()
                child_ret = ctypes.c_void_p()
                rx = ctypes.c_int()
                ry = ctypes.c_int()
                wx = ctypes.c_int()
                wy = ctypes.c_int()
                mask = ctypes.c_uint()
                ret = _x11.XQueryPointer(
                    _x11_display, _x11_root,
                    ctypes.byref(root_ret), ctypes.byref(child_ret),
                    ctypes.byref(rx), ctypes.byref(ry),
                    ctypes.byref(wx), ctypes.byref(wy),
                    ctypes.byref(mask),
                )
                if ret:
                    mx, my = rx.value, ry.value
                    if mouse_last_x[0] != 0.0 or mouse_last_y[0] != 0.0:
                        dx = mx - mouse_last_x[0]
                        if abs(dx) > 0.5:
                            new_yaw = cmd_yaw_rate[0] - dx * mouse_yaw_sensitivity
                            cmd_yaw_rate[0] = float(np.clip(new_yaw, -5.0, 5.0))
                            mouse_last_x[0] = mx
                            mouse_last_y[0] = my
                        else:
                            cmd_yaw_rate[0] *= mouse_yaw_decay
                            cx = _x11_screen_w // 2
                            cy = _x11_screen_h // 2
                            if _xtst is not None and (abs(mx - cx) > 2 or abs(my - cy) > 2):
                                new_x = int(mx + (cx - mx) * (1.0 - mouse_yaw_decay))
                                new_y = int(my + (cy - my) * (1.0 - mouse_yaw_decay))
                                _xtst.XTestFakeMotionEvent(_x11_display, -1, new_x, new_y, 0)
                                _x11.XFlush(_x11_display)
                                mouse_last_x[0] = new_x
                                mouse_last_y[0] = new_y
                            else:
                                mouse_last_x[0] = mx
                                mouse_last_y[0] = my
                    else:
                        mouse_last_x[0] = mx
                        mouse_last_y[0] = my
            except Exception:
                pass

        q = np.array([data.qpos[idx] for idx in joint_qpos])
        qvel = np.array([data.qvel[joint_dofadr[i]] for i in range(6)])

        is_ctrl_step = (phys_step % CTRL_STEPS == 0)
        smooth = args.action_smooth

        qw, qx, qy, qz = data.qpos[3:7]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))
        if is_ctrl_step:
            ctrl_step_count[0] += 1
            target_yaw[0] += cmd_yaw_rate[0] * CTRL_DT
            target_yaw[0] = np.arctan2(np.sin(target_yaw[0]), np.cos(target_yaw[0]))
        yaw_error = np.arctan2(np.sin(target_yaw[0] - yaw), np.cos(target_yaw[0] - yaw))

        gx = 2 * (qx * qz - qw * qy)
        pitch = np.arcsin(np.clip(gx, -1.0, 1.0))

        if is_ctrl_step:
            if pd_active[0]:
                last_actions = INJECT_ACT.copy()
                cmd_yaw_rate[0] = INJECT_CMD_YAW_RATE
            obs = build_observation(
                model, data, q, qvel, last_actions,
                cmd_vel[0], cmd_yaw_rate[0], cmd_height[0],
                yaw_error, default_dof_pos, cmd_jump[0],
            )
            obs = deploy_obs.add_observation_noise(
                obs, args.obs_noise_level, rng=obs_noise_rng,
            )

            if args.debug_obs > 0 and ctrl_step_count[0] % args.debug_obs == 0:
                debug_print_obs(obs, ctrl_step_count[0])

            obs_history.append(obs)
            if len(obs_history) == OBS_HISTORY_LENGTH:
                obs_hist = np.concatenate(list(obs_history)).astype(np.float32)
            else:
                obs_hist = np.tile(obs, OBS_HISTORY_LENGTH).astype(np.float32)

            if use_nn[0]:
                oh_n, o_n = normalize_obs(obs_hist, obs, norm, device)

                with torch.no_grad():
                    if args.compare:
                        raw_action_f32, latent_f32 = float_inf(oh_n, o_n)
                        raw_action_f32 = raw_action_f32.cpu().numpy()[0]
                        raw_action_i8, latent_i8 = inf_model(oh_n, o_n)
                        raw_action_i8 = raw_action_i8.cpu().numpy()[0]
                        raw_action = raw_action_i8
                        latent = latent_i8

                        diff = np.abs(raw_action_f32 - raw_action_i8)
                        if diff.max() > compare_stats["max_action_diff"]:
                            compare_stats["max_action_diff"] = diff.max()
                        compare_stats["sum_action_diff"] += diff.sum()
                        compare_stats["count"] += 1
                    else:
                        raw_action, latent = inf_model(oh_n, o_n)
                        raw_action = raw_action.cpu().numpy()[0]
                        pred_vel = float(latent[0, 0].cpu().numpy())

                raw_action[LEG_CLIP_INDICES] = np.clip(raw_action[LEG_CLIP_INDICES], -LEG_CLIP, LEG_CLIP)
                raw_action[WHEEL_CLIP_INDICES] = np.clip(raw_action[WHEEL_CLIP_INDICES], -WHEEL_CLIP, WHEEL_CLIP)

                # EMA 低通滤波: filtered = (1-smooth)*raw + smooth*old
                filtered_action[:] = (1 - smooth) * raw_action + smooth * filtered_action

                # env 的 last_actions 存的是**执行后（平滑后）**的动作，保持一致
                last_actions = filtered_action.copy()
            else:
                raw_action = np.zeros(6, dtype=np.float64)
                filtered_action[:] = 0.0

            # ── DEBUG: 每400步打印观测（无论use_nn是否生效都打印） ──
            debug_flag[0] = False
            if step_count[0] == 1 or step_count[0] % 400 == 1:
                debug_flag[0] = True
                nn_active = use_nn[0]
                print(f"\n{'='*60}")
                mark = "●" if nn_active else "○"
                rf_dbg = ""
                if rf_adr is not None:
                    _drf = float(mj_data.sensordata[rf_adr])
                    _flag = "触发窗口内" if 0.0 < _drf < args.auto_jump_dist else "未到阈值"
                    rf_dbg = f"  rf={_drf:+.3f}m(阈值{args.auto_jump_dist:.2f} {_flag})"
                print(f"[DEBUG step {step_count[0]}] cmd_height={cmd_height[0]:.3f}  nn={mark}{rf_dbg}  原始观测 (raw obs):")
                obs_names = list(OBS_NAMES)
                for i in range(NUM_OBS):
                    print(f"  [{i:2d}] {obs_names[i]:15s} = {obs[i]:+.6f}")
                if nn_active:
                    print("\n  归一化后观测 (norm obs):")
                    o_np = o_n.cpu().numpy()[0]
                    for i in range(NUM_OBS):
                        print(f"  [{i:2d}] {obs_names[i]:15s} = {o_np[i]:+.6f}")
                    print("\n  左右对称维度对比:")
                    pairs = [
                        (9, 10, "last_act LF/RF"),
                        (11, 12, "last_act LR/RR"),
                        (15, 16, "dof_err LF/RF"),
                        (17, 18, "dof_err LR/RR"),
                        (19, 20, "dof_vel LF/RF"),
                        (21, 22, "dof_vel LR/RR"),
                        (25, 26, "theta_l/r"),
                        (27, 28, "L0_l/r"),
                    ]
                    for li, ri, name in pairs:
                        diff_raw = obs[li] - obs[ri]
                        diff_norm = o_np[li] - o_np[ri]
                        print(f"  {name:20s}: raw=({obs[li]:+.6f},{obs[ri]:+.6f}) diff={diff_raw:+.6f}  "
                              f"norm=({o_np[li]:+.6f},{o_np[ri]:+.6f}) diff={diff_norm:+.6f}")
                    print(f"  latent (raw)           = {latent.cpu().numpy()[0]}")
                    print(f"  action (raw, no clip)  = {raw_action}")

        action_fifo[1:] = action_fifo[:-1]
        action_fifo[0] = filtered_action.copy()
        delayed_action = action_fifo[ACTION_DELAY].copy()

        max_base_z[0] = max(max_base_z[0], float(mj_data.qpos[2]))

        # 自动跳跃：前方测距 < 阈值就发 cmd_jump=1（和手动 H 走同一条通路）
        if rf_adr is not None and auto_jump_on[0]:
            # 每 0.5s 打一次当前测距，方便看"为什么不跳"
            rf_log_timer[0] -= dt
            if rf_log_timer[0] <= 0.0:
                rf_log_timer[0] = 0.5
                _d = float(mj_data.sensordata[rf_adr])
                if 0.0 < _d < args.auto_jump_dist:
                    _mark = "  ← 在触发窗口内"
                elif _d < 0.0:
                    _mark = "  ← 没命中(前方无遮挡)"
                elif _d > 2.0:
                    _mark = "  ← 贴脸盲区(射线已进入台阶内部)"
                else:
                    _mark = "  ← 太远, 未到阈值"
                print(f"  [rf] 传感器 {_d:+.3f} m → 机体到障碍 {_d + RF_SITE_OFFSET:.3f} m{_mark}")

            if auto_jump_cd[0] > 0.0:
                auto_jump_cd[0] -= dt
            elif jump_timer[0] <= 0.0:
                _dist = float(mj_data.sensordata[rf_adr])
                # 负值 = 射线没命中任何几何体
                if 0.0 < _dist < args.auto_jump_dist:
                    cmd_jump[0] = 1.0
                    jump_timer[0] = JUMP_HOLD_SEC
                    auto_jump_cd[0] = args.auto_jump_cooldown
                    print(f"  [auto-jump] 传感器 {_dist:.2f} m → 机体到障碍 "
                          f"{_dist + RF_SITE_OFFSET:.2f} m (阈值 {args.auto_jump_dist:.2f}) → cmd_jump=1")

        # H 键：cmd_jump=1 只保持 JUMP_HOLD_SEC 秒，到时自动清零。
        # 这里不再叠加任何额外 VMC 前馈 —— 跳跃完全由策略根据 cmd_jump 自己决定。
        if jump_timer[0] > 0.0:
            jump_timer[0] -= dt
            if jump_timer[0] <= 0.0:
                jump_timer[0] = 0.0
                cmd_jump[0] = 0.0
                print("  cmd_jump → 0（保持时长到）")

        if use_nn[0]:
            if _CONTROL_MODE == "vmc":
                tau = vmc_action_to_torque(delayed_action, q, qvel, pitch, default_dof_pos)
            else:
                vmc = compute_vmc_feedforward(q, pitch, default_dof_pos)
                tau = raw_action_to_torque(delayed_action, q, qvel, default_dof_pos, vmc)
        else:
            tau = compute_vmc_feedforward(q, pitch, default_dof_pos)
            tau[4:] = 0.0

        tau_raw = tau.copy()

        # ── 无力期: 只用重力前馈 (关 PD + 关轮驱), 与 joystick.py _pre_step_motor_control 一致 ──
        episode_steps[0] += 1
        if powerless_active[0] and episode_steps[0] < powerless_steps:
            vmc_ff = compute_vmc_feedforward(q, pitch, default_dof_pos)
            tau[:4] = vmc_ff[:4]
            tau[4:] = 0.0
        elif episode_steps[0] >= powerless_steps:
            powerless_active[0] = False

        if pd_active[0]:
            cur_pd = np.array([q[0], q[1], q[2], q[3]], dtype=np.float64)
            vel_pd = np.array([qvel[0], qvel[1], qvel[2], qvel[3]], dtype=np.float64)
            tau_pd = pd_kp * (pd_target - cur_pd) - pd_kd * vel_pd
            tau[:4] = tau_pd
            tau[4:] = 0.0
            if debug_flag[0]:
                err_max = np.max(np.abs(pd_target - cur_pd))
                print(f"  [PD] err_max={np.rad2deg(err_max):+.2f}°  "
                      f"tau=[{tau_pd[0]:+.2f},{tau_pd[1]:+.2f},{tau_pd[2]:+.2f},{tau_pd[3]:+.2f}]Nm")
        else:
            tau[:4] = np.clip(1 * tau[:4], -40.0, 40.0)
            tau[4:] *= 1

        if debug_flag[0]:
            mode = "PD" if pd_active[0] else ("NN" if use_nn[0] else "无力期(前馈)")
            if use_nn[0]:
                dof_err = q[:4] - default_dof_pos
                print(f"  dof_err(°): [{np.rad2deg(dof_err[0]):+.2f}, {np.rad2deg(dof_err[1]):+.2f}, {np.rad2deg(dof_err[2]):+.2f}, {np.rad2deg(dof_err[3]):+.2f}]")
                if _CONTROL_MODE == "pd":
                    print(f"  VMC前馈: [{vmc[0]:+.2f}, {vmc[1]:+.2f}, {vmc[2]:+.2f}, {vmc[3]:+.2f}, {vmc[4]:+.2f}, {vmc[5]:+.2f}]Nm")
            print(f"  原始力矩(raw): [{tau_raw[0]:+.4f}, {tau_raw[1]:+.4f}, {tau_raw[2]:+.4f}, "
                  f"{tau_raw[3]:+.4f}, {tau_raw[4]:+.4f}, {tau_raw[5]:+.4f}]Nm")
            print(f"  力矩[{mode}] (tau): [{tau[0]:+.4f}, {tau[1]:+.4f}, {tau[2]:+.4f}, "
                  f"{tau[3]:+.4f}, {tau[4]:+.4f}, {tau[5]:+.4f}]Nm")
            print(f"{'='*60}\n")

        if args.compare and use_nn[0]:
            raw_action_f32[LEG_CLIP_INDICES] = np.clip(raw_action_f32[LEG_CLIP_INDICES], -LEG_CLIP, LEG_CLIP)
            raw_action_f32[WHEEL_CLIP_INDICES] = np.clip(raw_action_f32[WHEEL_CLIP_INDICES], -WHEEL_CLIP, WHEEL_CLIP)
            filtered_action_f32[:] = (1 - smooth) * raw_action_f32 + smooth * filtered_action_f32
            action_fifo_f32[1:] = action_fifo_f32[:-1]
            action_fifo_f32[0] = filtered_action_f32.copy()
            delayed_action_f32 = action_fifo_f32[ACTION_DELAY].copy()
            if _CONTROL_MODE == "vmc":
                tau_f32 = vmc_action_to_torque(delayed_action_f32, q, qvel, pitch, default_dof_pos)
            else:
                vmc_f32 = compute_vmc_feedforward(q, pitch, default_dof_pos)
                tau_f32 = raw_action_to_torque(delayed_action_f32, q, qvel, default_dof_pos, vmc_f32)
            tau_diff = np.abs(tau_f32 - tau)
            if tau_diff.max() > compare_stats["max_torque_diff"]:
                compare_stats["max_torque_diff"] = tau_diff.max()
            compare_stats["sum_torque_diff"] += tau_diff.sum()

        for i in range(6):
            data.ctrl[act_ids[i]] = tau[i]

        if is_ctrl_step and step_count[0] % (100 * CTRL_STEPS) == 1:
            pitch, roll = deploy_obs.quat_pitch_roll(data.qpos[3:7])
            q_lf, q_rf, q_lr, q_rr = q[0], q[1], q[2], q[3]
            clf = q_lf - default_dof_pos[0]
            crf = q_rf - default_dof_pos[1]
            clr = q_lr - default_dof_pos[2]
            crr = q_rr - default_dof_pos[3]
            phi1_l = clr + 5 * np.pi / 6
            phi4_l = clf + np.pi / 6
            phi1_r = crr + 5 * np.pi / 6
            phi4_r = crf + np.pi / 6
            L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
            L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)

            rf_txt = (f"rf={float(data.sensordata[rf_adr]):+.3f}m "
                      if rf_adr is not None else "")
            print(f"  [{step_count[0]:5d}] ({tag}) "
                  f"pitch={np.rad2deg(pitch):+6.2f}° roll={np.rad2deg(roll):+6.2f}° "
                  f"L0=({L0_l:.3f},{L0_r:.3f})m "
                  f"z={data.qpos[2]:.3f}m "
                  f"{rf_txt}"
                  f"vx_real={data.qvel[0]:+.2f} pred_vx={pred_vel:+.2f} "
                  f"wyaw_real={data.qvel[5]:+.2f} "
                  f"yaw_err={np.rad2deg(yaw_error):+.1f}° "
                  f"act=[{raw_action[0]:+.3f},{raw_action[1]:+.3f},"
                  f"{raw_action[2]:+.3f},{raw_action[3]:+.3f},"
                  f"{raw_action[4]:+.3f},{raw_action[5]:+.3f}]  "
                  f"tau=[{tau[0]:+.2f},{tau[1]:+.2f},{tau[2]:+.2f},{tau[3]:+.2f},{tau[4]:+.2f},{tau[5]:+.2f}]")

            if args.compare and compare_stats["count"] > 0:
                avg_act = compare_stats["sum_action_diff"] / (compare_stats["count"] * 6)
                avg_tor = compare_stats["sum_torque_diff"] / (compare_stats["count"] * 6)
                print(f"         [对比] max_act_diff={compare_stats['max_action_diff']:.6f} "
                      f"avg_act_diff={avg_act:.6f} "
                      f"max_tor_diff={compare_stats['max_torque_diff']:.4f} "
                      f"avg_tor_diff={avg_tor:.4f}")

    def key_callback(keycode):
        if keycode == 32:
            paused[0] = not paused[0]
            print(f"  {'⏸ 暂停' if paused[0] else '▶ 继续'}")
        elif keycode == 77:
            mouse_yaw_active[0] = not mouse_yaw_active[0]
            mouse_last_x[0] = 0.0
            mouse_last_y[0] = 0.0
            status = "ON (鼠标左右划控制 vyaw)" if mouse_yaw_active[0] else "OFF"
            print(f"  鼠标偏航控制: {status}")
        elif keycode == 265:  # ↑ 前进 (参考 play: 每按一次 +0.1 m/s)
            cmd_vel[0] = float(np.clip(cmd_vel[0] + CMD_VX_STEP, *CMD_VX_LIMIT))
            print(f"  cmd_vel={cmd_vel[0]:+.2f} m/s")
        elif keycode == 264:  # ↓ 后退
            cmd_vel[0] = float(np.clip(cmd_vel[0] - CMD_VX_STEP, *CMD_VX_LIMIT))
            print(f"  cmd_vel={cmd_vel[0]:+.2f} m/s")
        elif keycode == 263:  # ← 左转 (参考 play: 每按一次 +0.2 rad/s)
            cmd_yaw_rate[0] = float(np.clip(cmd_yaw_rate[0] + CMD_YAW_STEP, *CMD_YAW_LIMIT))
            print(f"  cmd_yaw_rate={cmd_yaw_rate[0]:+.2f} rad/s")
        elif keycode == 262:  # → 右转
            cmd_yaw_rate[0] = float(np.clip(cmd_yaw_rate[0] - CMD_YAW_STEP, *CMD_YAW_LIMIT))
            print(f"  cmd_yaw_rate={cmd_yaw_rate[0]:+.2f} rad/s")
        elif keycode in (257, 335):  # Enter: 命令归零 (参考 play 的 commander.zero())
            cmd_vel[0] = 0.0
            cmd_yaw_rate[0] = 0.0
            print("  命令归零  vx=0  vyaw=0")
        elif keycode == 72:  # H: 自动跳跃扫描开关
            if rf_adr is None:
                print("  ⚠ 该场景没有前方测距传感器，无法开启自动跳跃扫描")
            else:
                auto_jump_on[0] = not auto_jump_on[0]
                print(f"  [H] 自动跳跃扫描 {'ON' if auto_jump_on[0] else 'OFF'}"
                      f"（读数 < {args.auto_jump_dist:.2f} m 时自动发 cmd_jump=1）")
        elif keycode == 259:
            need_reset[0] = True
        elif keycode == 82:
            cmd_height[0] = min(cmd_height[0] + 0.02, 0.42)
            print(f"  cmd_height={cmd_height[0]:.3f} m")
        elif keycode == 84:
            cmd_height[0] = max(cmd_height[0] - 0.02, 0.16)
            print(f"  cmd_height={cmd_height[0]:.3f} m")
        elif keycode == 78:
            use_nn[0] = True
            print("  → 神经网络模式")
        elif keycode == 76:
            use_nn[0] = not use_nn[0]
            print(f"  → NN控制 {'ON' if use_nn[0] else 'OFF (力矩归零)'}")
        elif keycode == 74:  # J: 手动跳一次
            cmd_jump[0] = 1.0
            jump_timer[0] = JUMP_HOLD_SEC
            print(f"  [J] cmd_jump=1，保持 {JUMP_HOLD_SEC:.2f}s"
                  f"（{int(round(JUMP_HOLD_SEC / CTRL_DT))} 个控制步），由策略自主起跳")
        elif keycode == 75:  # K: 注入自定义关节角度
            if custom_dof_pos is not None:
                for i in range(4):
                    mj_data.qpos[joint_qpos[i]] = custom_dof_pos[i]
                mujoco.mj_forward(mj_model, mj_data)
                dpos = np.array([mj_data.qpos[joint_qpos[i]] for i in range(6)])
                print(f"  🔧 注入关节角度: LF={np.rad2deg(dpos[0]):.1f}° RF={np.rad2deg(dpos[1]):.1f}° LR={np.rad2deg(dpos[2]):.1f}° RR={np.rad2deg(dpos[3]):.1f}°")
            else:
                print("  ⚠ 未设置 --init_dof_pos，请重启时指定")
        elif keycode == 80:  # P: 切换 PD 控制到实物位姿
            pd_active[0] = not pd_active[0]
            if pd_active[0]:
                print(f"  🤖 PD控制 ON → 目标: LF={np.rad2deg(pd_target[0]):+.1f}° RF={np.rad2deg(pd_target[1]):+.1f}° "
                      f"LR={np.rad2deg(pd_target[2]):+.1f}° RR={np.rad2deg(pd_target[3]):+.1f}°")
                print(f"     Kp={pd_kp} Nm/rad  Kd={pd_kd} Nm/(rad·s)")
            else:
                print("  🤖 PD控制 OFF")

    class _NullCam:
        def __init__(self):
            self.lookat = np.zeros(3)

    class _NullViewer:
        """--headless 的替身：不建窗口，跑够 headless_steps 就结束。"""

        def __init__(self):
            self.cam = _NullCam()

        def is_running(self):
            return step_count[0] < args.headless_steps

        def sync(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    if args.headless:
        print(f"  [HEADLESS] 不开查看器, 运行 {args.headless_steps} 步")
        viewer_ctx = _NullViewer()
    else:
        viewer_ctx = mujoco.viewer.launch_passive(
            mj_model, mj_data, key_callback=key_callback,
            show_left_ui=False, show_right_ui=False,
        )

    with viewer_ctx as viewer:
        while viewer.is_running():
            if need_reset[0]:
                reset_sim()
                need_reset[0] = False
                continue
            if not paused[0]:
                controller(mj_model, mj_data)
                mujoco.mj_step(mj_model, mj_data)
                viewer.cam.lookat[:] = mj_data.qpos[0:3]
                viewer.sync()
            else:
                time.sleep(0.01)

    if args.compare and compare_stats["count"] > 0:
        avg_act = compare_stats["sum_action_diff"] / (compare_stats["count"] * 6)
        avg_tor = compare_stats["sum_torque_diff"] / (compare_stats["count"] * 6)
        print(f"\n  {'='*60}")
        print(f"  Sim-to-Sim 对比总结 (f32 vs int8), {compare_stats['count']} 步")
        print(f"  动作差异: max={compare_stats['max_action_diff']:.6f}  avg={avg_act:.6f}")
        print(f"  力矩差异: max={compare_stats['max_torque_diff']:.4f}  avg={avg_tor:.4f}")
        print(f"  {'='*60}")

    print(f"  最高 base z = {max_base_z[0]:.3f} m   end x={mj_data.qpos[0]:.2f} z={mj_data.qpos[2]:.3f} m"
          f"   (站立≈0.14, 台阶1顶面≈0.19, 台阶2顶面≈0.34, 台阶3顶面≈0.56)")
    print("退出.")


if __name__ == "__main__":
    main()