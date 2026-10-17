#!/usr/bin/env python3
"""导出 Cube.AI 所需文件: float32 ONNX + 归一化 C 头文件 + 校准数据.

用法:
  python export_onnx.py \
    --input /path/to/model_4000.pt \
    --output quantize/model_f32.onnx \
    --calib_from_sim --calib_samples 2000
"""

import os
import sys
import argparse
import copy
import shutil
from collections import deque
import numpy as np

import torch
import torch.nn as nn

_PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PARENT_DIR)
sys.path.insert(0, os.path.join(_PARENT_DIR, "lqr"))

from quantize_model import (
    load_float_model, Int8Linear,
    _replace_linear_with_int8_calib, ActorCriticSequence,
)

OBS_HISTORY_LENGTH = 5


# ═══════════════════════════════════════════════════════════════
# 导出模型 (纯 encoder+actor, 不含归一化)
# ═══════════════════════════════════════════════════════════════

class ExportModel(nn.Module):
    """纯网络: encoder + actor (不含任何归一化).

    MCU 上完成 obs 归一化后喂入:
      输入: obs_history_norm(165) + obs_norm(33)  [MCU 已归一化]
      输出: action(6)
    ONNX 内部只含 Linear + ELU, Cube.AI 量化精度最高.
    """

    def __init__(self, encoder, actor, vel_scale):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.vel_scale = vel_scale

    def forward(self, obs_history_norm, obs_norm):
        latent = self.encoder(obs_history_norm) * self.vel_scale
        actor_input = torch.cat([obs_norm, latent], dim=-1)
        return self.actor(actor_input)


# ═══════════════════════════════════════════════════════════════
# 反量化 int8 → float32
# ═══════════════════════════════════════════════════════════════

def dequantize_to_float32(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not ckpt.get("quantized") or ckpt.get("quantize_dtype") != "qint8_per_channel":
        raise ValueError("checkpoint 不是 qint8_per_channel 量化模型")

    model = ActorCriticSequence(
        num_obs=33, num_critic_obs=150, num_actions=6,
        num_encoder_obs=165, latent_dim=3,
        encoder_hidden_dims=[128, 64, 3],
        actor_hidden_dims=[128, 64, 32, 6],
        critic_hidden_dims=[128, 128, 64, 1],
        activation="elu", obs_normalization=False, use_real_vel=False,
    )
    _replace_linear_with_int8_calib(model.encoder)
    _replace_linear_with_int8_calib(model.actor)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.actor.load_state_dict(ckpt["actor_state_dict"])

    def dequantize_module(module):
        for name, child in module.named_children():
            if isinstance(child, Int8Linear):
                w = child.weight_q.float() * child.scale
                new_linear = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                new_linear.weight.data = w
                if child.bias is not None:
                    new_linear.bias.data = child.bias
                setattr(module, name, new_linear)
            elif isinstance(child, nn.Sequential):
                dequantize_module(child)

    dequantize_module(model.encoder)
    dequantize_module(model.actor)
    print("  反量化: Int8Linear → nn.Linear (float32)")
    return model, ckpt


# ═══════════════════════════════════════════════════════════════
# 收集归一化后的校准数据
# ═══════════════════════════════════════════════════════════════

def collect_norm_calib(num_steps, norm_dict, scene="rmuc"):
    """从仿真收集归一化后的观测数据，匹配 ONNX 模型输入。

    ONNX 模型输入是已归一化的 165 + 33 维向量，校准数据也要归一化。
    """
    try:
        import mujoco
        from test_vmc_lqr import fk_fivebar
    except ImportError as e:
        print(f"[ERROR] 导入失败: {e}")
        sys.exit(1)

    scene_dir = os.path.join(_PARENT_DIR, "..", "..", "..", "..",
                             "assets", "robots", "balance_infantry")
    scene_path = os.path.join(scene_dir, f"scene_{scene}.xml")
    print(f"  校准场景: {scene_path}")

    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(mj_model)

    joint_names = [
        "left_front_joint", "right_front_joint",
        "left_rear_joint", "right_rear_joint",
        "left_wheel_joint", "right_wheel_joint",
    ]
    joint_qpos = [mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                  for n in joint_names]
    joint_dofadr = [mj_model.jnt_dofadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                    for n in joint_names]

    try:
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    except Exception:
        pass
    mj_data.qpos[2] = 0.8
    mujoco.mj_forward(mj_model, mj_data)

    default_dof_pos = np.array([mj_data.qpos[joint_qpos[i]] for i in range(4)], dtype=np.float32)

    CTRL_DT = 0.02
    SIM_DT = 0.005
    CTRL_STEPS = int(CTRL_DT / SIM_DT)
    mj_model.opt.timestep = SIM_DT

    enc_mean = norm_dict["enc_mean"]
    enc_std = norm_dict["enc_std"]
    obs_mean = norm_dict["obs_mean"]
    obs_std = norm_dict["obs_std"]
    eps = 1e-2

    obs_history = deque(maxlen=OBS_HISTORY_LENGTH)
    last_actions = np.zeros(6, dtype=np.float32)
    wheel_pos_err = np.zeros(2, dtype=np.float32)
    wheel_err_decay_count = 0
    cmd_vel = 0.0
    cmd_yaw_rate = 0.0
    cmd_height = 0.29
    qw, qx, qy, qz = mj_data.qpos[3:7]
    target_yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz)))

    oh_list = []
    o_list = []
    phys_step = 0

    print(f"  收集归一化校准数据 (目标 {num_steps} 步)...")
    while len(oh_list) < num_steps:
        q = np.array([mj_data.qpos[idx] for idx in joint_qpos])
        qvel = np.array([mj_data.qvel[joint_dofadr[i]] for i in range(6)])

        is_ctrl_step = (phys_step % CTRL_STEPS == 0)
        phys_step += 1

        qw, qx, qy, qz = mj_data.qpos[3:7]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))
        if is_ctrl_step:
            target_yaw += cmd_yaw_rate * CTRL_DT
            target_yaw = np.arctan2(np.sin(target_yaw), np.cos(target_yaw))
        yaw_error = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))

        WHEEL_RADIUS = 0.06
        WHEELBASE_val = 0.380
        vx = mj_data.qvel[0]
        target_lw = (cmd_vel - 0.5 * cmd_yaw_rate * WHEELBASE_val) / WHEEL_RADIUS
        target_rw = (cmd_vel + 0.5 * cmd_yaw_rate * WHEELBASE_val) / WHEEL_RADIUS
        err_dot = np.array([target_lw - qvel[4], target_rw - qvel[5]], dtype=np.float32)
        if abs(vx) < 0.3:
            wheel_pos_err += err_dot * SIM_DT
            wheel_err_decay_count = 0
        else:
            if wheel_err_decay_count == 0:
                wheel_err_decay_count = 4
        if is_ctrl_step and wheel_err_decay_count > 0:
            remain = float(wheel_err_decay_count)
            wheel_pos_err *= (remain - 1.0) / remain
            wheel_err_decay_count -= 1

        if is_ctrl_step:
            gyro = mj_data.qvel[3:6].astype(np.float32)
            gx_ = 2 * (qx * qz - qw * qy)
            pitch = float(np.arcsin(np.clip(gx_, -1.0, 1.0)))
            gx, gy_val = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx)
            roll = float(np.arctan2(gy_val, np.sqrt(1.0 - gx * gx)))
            dof_pos_err = (default_dof_pos - q[:4]).astype(np.float32)
            dof_vel = qvel[:6].astype(np.float32)
            phi1_l = q[2] + 5 * np.pi / 6
            phi4_l = q[0] + np.pi / 6
            phi1_r = q[3] + 5 * np.pi / 6
            phi4_r = q[1] + np.pi / 6
            L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
            L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)
            theta_l = np.pi / 2 - phi0_l - pitch
            theta_r = np.pi / 2 - phi0_r - pitch

            obs = np.concatenate([
                gyro,
                np.array([pitch], dtype=np.float32),
                np.array([roll], dtype=np.float32),
                np.array([cmd_vel, cmd_yaw_rate, cmd_height, 0.05], dtype=np.float32),
                last_actions,
                np.array([wheel_pos_err[0]], dtype=np.float32),
                np.array([wheel_pos_err[1]], dtype=np.float32),
                np.array([np.sin(yaw_error)], dtype=np.float32),
                np.array([np.cos(yaw_error)], dtype=np.float32),
                dof_pos_err,
                dof_vel,
                np.array([theta_l], dtype=np.float32),
                np.array([theta_r], dtype=np.float32),
                np.array([L0_l], dtype=np.float32),
                np.array([L0_r], dtype=np.float32),
            ]).astype(np.float32)

            obs_history.append(obs)
            if len(obs_history) == OBS_HISTORY_LENGTH:
                obs_hist = np.concatenate(list(obs_history)).astype(np.float32)
            else:
                obs_hist = np.tile(obs, OBS_HISTORY_LENGTH).astype(np.float32)

            # 归一化后存储
            oh_norm = (obs_hist - enc_mean) / (enc_std + eps)
            o_norm = (obs - obs_mean) / (obs_std + eps)
            oh_list.append(oh_norm.astype(np.float32))
            o_list.append(o_norm.astype(np.float32))

            last_actions = np.random.uniform(-1.0, 1.0, 6).astype(np.float32)
            if len(oh_list) % 50 == 0:
                cmd_vel = np.random.uniform(-1.0, 1.0)
                cmd_yaw_rate = np.random.uniform(-1.0, 1.0)
                cmd_height = np.random.uniform(0.25, 0.35)

        tau = np.zeros(6, dtype=np.float64)
        for i in range(6):
            mj_data.ctrl[joint_dofadr[i]] = tau[i]
        mujoco.mj_step(mj_model, mj_data)

    print(f"  收集完成: {len(oh_list)} 步")
    return oh_list, o_list


# ═══════════════════════════════════════════════════════════════
# 导出归一化参数 C 头文件
# ═══════════════════════════════════════════════════════════════

def export_norm_header(norm_dict, vel_scale, output_dir):
    """导出归一化参数为 C 头文件，MCU 上做 float32 归一化。"""
    path = os.path.join(output_dir, "norm_params.h")
    with open(path, "w") as f:
        f.write("// 自动生成: 归一化参数 (MCU 上 float32 计算)\n")
        f.write("// 公式: y = (x - mean) / (std + 1e-2)\n\n")
        f.write("#pragma once\n\n")

        for name in ["enc_mean", "enc_std", "obs_mean", "obs_std"]:
            arr = norm_dict[name]
            n = len(arr)
            f.write(f"static const float {name}[{n}] = {{\n    ")
            f.write(", ".join(f"{v:.8f}f" for v in arr))
            f.write(f"\n}};\n\n")

        f.write(f"#define ENC_DIM 165\n")
        f.write(f"#define OBS_DIM 33\n")
        f.write(f"#define OBS_HISTORY_LENGTH 5\n")
        f.write(f"#define NORM_EPS 1e-2f\n\n")

        f.write("// MCU 推理流程:\n")
        f.write("// 1. oh = (obs_history - enc_mean) / (enc_std + NORM_EPS)\n")
        f.write("// 2. o  = (obs - obs_mean) / (obs_std + NORM_EPS)\n")
        f.write("// 3. 喂入 ONNX (encoder+actor) → action[6]\n")
        f.write("// (latent 不归一化, encoder 输出直接 cat obs 入 actor)\n")

    print(f"  归一化头文件: {path}")


# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="导出 Cube.AI 所需文件")
    parser.add_argument("--input", type=str, required=True,
                        help="模型路径 (float32 PPO 或 qint8_per_channel)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 ONNX 路径 (默认自动派生: model_XXXX.pt → model_XXXX_deploy.onnx)")
    parser.add_argument("--calib_from_sim", action="store_true",
                        help="从仿真收集校准数据")
    parser.add_argument("--calib_samples", type=int, default=2000,
                        help="校准样本数")
    args = parser.parse_args()

    device = torch.device("cpu")

    # 输出路径处理: 自动派生 model_XXXX.pt → quantize/<项目>/model_XXXX_deploy.onnx
    input_abs = os.path.abspath(args.input)
    input_name = os.path.splitext(os.path.basename(input_abs))[0]
    project_name = os.path.basename(os.path.dirname(input_abs))
    if args.output is None:
        output_dir = os.path.join(_PARENT_DIR, project_name)
        output_path = os.path.join(output_dir, f"{input_name}_deploy.onnx")
    else:
        output_path = os.path.abspath(args.output)
        output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 复制原始 .pt 到导出目录
    dst_pt = os.path.join(output_dir, os.path.basename(input_abs))
    if os.path.abspath(input_abs) != os.path.abspath(dst_pt):
        shutil.copy2(input_abs, dst_pt)
        print(f"复制原始模型: {dst_pt}")

    print(f"输出 ONNX: {output_path}")

    # ── 1. 加载 ──
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    if ckpt.get("quantized") and ckpt.get("quantize_dtype") == "qint8_per_channel":
        print("加载 int8 模型, 反量化到 float32...")
        model, ckpt = dequantize_to_float32(args.input)
        norm_dict = {
            "enc_mean": ckpt["enc_mean"], "enc_std": ckpt["enc_std"],
            "obs_mean": ckpt["obs_mean"], "obs_std": ckpt["obs_std"],
            "latent_mean": ckpt["latent_mean"], "latent_std": ckpt["latent_std"],
        }
        vel_scale = ckpt.get("vel_scale", 3.0)
    else:
        print("加载 float32 PPO 模型...")
        model, norm_dict = load_float_model(args.input, device)
        vel_scale = model.vel_scale

    # ── 2. 导出归一化 C 头文件 ──
    export_norm_header(norm_dict, vel_scale, output_dir)

    # ── 3. 构建导出模型 (纯 encoder+actor, 无归一化) ──
    export_model = ExportModel(
        model.encoder, model.actor, vel_scale,
    )
    export_model.eval()
    export_model.to(device)

    # ── 4. 导出 ONNX ──
    oh = torch.randn(1, 165, device=device)
    o = torch.randn(1, 33, device=device)

    torch.onnx.export(
        export_model, (oh, o), output_path,
        input_names=["obs_history_norm", "obs_norm"],
        output_names=["action"],
        opset_version=13,
        dynamic_axes={
            "obs_history_norm": {0: "batch"},
            "obs_norm": {0: "batch"},
            "action": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"  ONNX 已保存: {os.path.abspath(output_path)}")

    # ── 5. 导出校准数据 (归一化后) ──
    if args.calib_from_sim:
        oh_list, o_list = collect_norm_calib(args.calib_samples, norm_dict, "rmuc")
        oh_arr = np.stack(oh_list, axis=0).astype(np.float32)
        o_arr = np.stack(o_list, axis=0).astype(np.float32)
        np.save(os.path.join(output_dir, "calib_obs_history_norm.npy"), oh_arr)
        np.save(os.path.join(output_dir, "calib_obs_norm.npy"), o_arr)
        print(f"  校准数据: calib_obs_history_norm.npy ({oh_arr.shape})")
        print(f"  校准数据: calib_obs_norm.npy ({o_arr.shape})")
        print(f"  范围: oh=[{oh_arr.min():.4f}, {oh_arr.max():.4f}], "
              f"o=[{o_arr.min():.4f}, {o_arr.max():.4f}]")
    else:
        print("  跳过校准数据 (--calib_from_sim 启用)")

    # ── 6. 总结 ──
    print(f"\n{'='*60}")
    print(f"Cube.AI 部署文件 (在 {output_dir}/):")
    print(f"  model_f32.onnx              → Cube.AI import, int8 量化")
    print(f"  norm_params.h               → MCU 归一化 (C 头文件)")
    print(f"  calib_obs_history_norm.npy  → 校准输入 (obs_history, 归一化后)")
    print(f"  calib_obs_norm.npy          → 校准输入 (obs, 归一化后)")
    print(f"\nMCU 推理流程:")
    print(f"  1. 构建 obs_history(165) + obs(33) 原始观测")
    print(f"  2. float32 归一化: oh = (oh - enc_mean) / (enc_std + 1e-2)")
    print(f"                     o  = (o  - obs_mean) / (obs_std + 1e-2)")
    print(f"  3. 喂入 ONNX (encoder+actor, 纯 Linear+ELU) → action(6)")
    print(f"  (latent 不归一化, encoder 输出直接 cat obs 入 actor)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()