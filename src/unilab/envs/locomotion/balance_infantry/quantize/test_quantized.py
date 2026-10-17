#!/usr/bin/env python3
"""加载 int8 量化 PPO 策略，在 MuJoCo 仿真中测试 Sim-to-Sim 效果。

参数与 mujoco.yaml 配置对齐。

用法:
    cd /path/to/balance_infantry

    # int8 量化模型测试 (默认 rmuc 场景)
    python quantize/test_quantized.py --model quantize/ppo_int8.pt

    # float32 原始模型测试
    python quantize/test_quantized.py --model /home/xxx/ubuntu_software/unilab/UniLab/logs/rsl_rl_sequence/BalanceInfantryJoystickFlat/2026-09-05_21-08-28_mujoco/model_4000.pt --no_quantize

    # 对比模式: 同时跑 float32 和 int8, 统计差异
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

import numpy as np
import mujoco
import mujoco.viewer
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
sys.path.insert(0, os.path.join(_UNILAB_SRC, "src"))

from unilab.algos.torch.ppo.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence
from test_vmc_lqr import fk_fivebar

# ═══════════════════════════════════════════════════════════════
# PD 关节目标 & 实物注入 (直接修改此处即可)
# ═══════════════════════════════════════════════════════════════
PD_TARGET = np.array([-0.833, -0.897, -0.204, -0.101], dtype=np.float64)
# PD_TARGET (°):     LF=-47.7°  RF=-51.4°  LR=-11.7°  RR=-5.8°
# 实物 dof_err:       LF=-0.893  RF=-0.907  LR=-0.104  RR=-0.101  (rad)

INJECT_ACT = np.array([-1.722, -1.586, 1.726, 0.919, -2.373, -1.841], dtype=np.float32)
# 实物 last_actions:  LF         RF         LR         RR         LW         RW
INJECT_CMD_YAW_RATE = 0.3912  # 实物 cmd_yaw (rad/s)，按P注入

# 实物 tau (带VMC):   LF=+20.6  RF=+21.1  LR=+37.4  RR=+17.0  LW=-3.5  RW=-3.5  (Nm)

# ═══════════════════════════════════════════════════════════════
# 模型超参数 (与 mujoco.yaml 一致)
# ═══════════════════════════════════════════════════════════════
NUM_OBS = 31
NUM_ACTIONS = 6
OBS_HISTORY_LENGTH = 5
NUM_ENCODER_OBS = OBS_HISTORY_LENGTH * NUM_OBS
LATENT_DIM = 3
ENCODER_HIDDEN_DIMS = [128, 64]
ACTOR_HIDDEN_DIMS = [128, 64, 32]
CRITIC_HIDDEN_DIMS = [256, 128, 64]
NUM_CRITIC_OBS = 150  # privileged_obs(147) + latent_dim(3)
ACTIVATION = "elu"

# ═══════════════════════════════════════════════════════════════
# PD 参数 (与 mujoco.yaml control_config 一致)
# ═══════════════════════════════════════════════════════════════
LEG_KP = (50.0, 50.0, 50.0, 50.0)
LEG_KD = (3.2, 3.2, 3.2, 3.2)
WHEEL_KD = (0.1, 0.1)
ACTION_SCALE_JOINT = 0.5
ACTION_SCALE_WHEEL = 50.0
JOINT_LIMITS = np.array([
    [-1.57, 2.40], [-1.57, 2.40], [-1.57, 1.57], [-1.57, 1.57],
    [-np.inf, np.inf], [-np.inf, np.inf],
], dtype=np.float64)

# 动作裁剪 (与 joystick.py apply_action 一致)
# 训练时: clip_actions=100.0, 腿和轮都几乎不裁剪, 模型输出无 tanh
LEG_CLIP = 100.0
WHEEL_CLIP = 100.0
# 裁剪索引: 与 base.py LEG_JOINT_INDICES / WHEEL_JOINT_INDICES 一致
LEG_CLIP_INDICES = [0, 1, 2, 3]
WHEEL_CLIP_INDICES = [4, 5]

# ═══════════════════════════════════════════════════════════════
# VMC 参数 (与 mujoco.yaml control_config.vmc 一致)
# ═══════════════════════════════════════════════════════════════
VMC_L1 = 0.215
VMC_L4 = 0.215
VMC_FEEDFORWARD_FORCE = 80.0
VMC_JUMP_FORCE = 400.0
JUMP_DURATION = 0.3

ACTION_DELAY = 2


# ═══════════════════════════════════════════════════════════════
# 观测构建 (与 joystick.py _compute_obs 一致)
# ═══════════════════════════════════════════════════════════════

def build_observation(model, data, dof_pos, dof_vel, last_actions,
                      cmd_vel, cmd_yaw_rate, cmd_height,
                      yaw_error, default_dof_pos):
    gyro = data.qvel[3:6].copy()

    qw, qx, qy, qz = data.qpos[3:7]
    gx = 2 * (qx * qz - qw * qy)
    gy = 2 * (qy * qz + qw * qx)
    gz = 1 - 2 * (qx * qx + qy * qy)
    pitch = np.arcsin(np.clip(gx, -1.0, 1.0))
    roll = np.arcsin(np.clip(-gy, -1.0, 1.0))

    commands = np.array([cmd_vel, cmd_yaw_rate, cmd_height, 0.0])

    q_lr, q_rr, q_lf, q_rf = dof_pos[2], dof_pos[3], dof_pos[0], dof_pos[1]
    cal_lr, cal_rr, cal_lf, cal_rf = q_lr - default_dof_pos[2], q_rr - default_dof_pos[3], q_lf - default_dof_pos[0], q_rf - default_dof_pos[1]
    phi1_l = cal_lr + 5 * np.pi / 6
    phi1_r = cal_rr + 5 * np.pi / 6
    phi4_l = cal_lf + np.pi / 6
    phi4_r = cal_rf + np.pi / 6
    L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
    L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)
    theta_l = np.pi / 2 - phi0_l - pitch
    theta_r = np.pi / 2 - phi0_r - pitch

    dof_pos_err = dof_pos[:4] - default_dof_pos[:4]

    obs = np.concatenate([
        gyro,
        np.array([pitch]),
        np.array([roll]),
        commands,
        last_actions,
        np.array([np.sin(yaw_error)]),
        np.array([np.cos(yaw_error)]),
        dof_pos_err,
        dof_vel,
        np.array([theta_l]),
        np.array([theta_r]),
        np.array([L0_l]),
        np.array([L0_r]),
    ])
    return obs.astype(np.float32)


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
    (15, "sin(yaw_error)   ", "sin(target_yaw - yaw)"),
    (16, "cos(yaw_error)   ", "cos(target_yaw - yaw)"),
    (17, "dof_pos_err_LF   ", "LF 关节偏差 (rad)"),
    (18, "dof_pos_err_RF   ", "RF 关节偏差 (rad)"),
    (19, "dof_pos_err_LR   ", "LR 关节偏差 (rad)"),
    (20, "dof_pos_err_RR   ", "RR 关节偏差 (rad)"),
    (21, "dof_vel_LF       ", "LF 关节速度 (rad/s)"),
    (22, "dof_vel_RF       ", "RF 关节速度 (rad/s)"),
    (23, "dof_vel_LR       ", "LR 关节速度 (rad/s)"),
    (24, "dof_vel_RR       ", "RR 关节速度 (rad/s)"),
    (25, "dof_vel_LW       ", "LW 轮子速度 (rad/s)"),
    (26, "dof_vel_RW       ", "RW 轮子速度 (rad/s)"),
    (27, "theta_l          ", "左腿 VMC 摆角 (rad), 前倾增大"),
    (28, "theta_r          ", "右腿 VMC 摆角 (rad), 前倾增大"),
    (29, "L0_l             ", "左腿虚拟腿长 (m), ~0.28"),
    (30, "L0_r             ", "右腿虚拟腿长 (m), ~0.28"),
]


def debug_print_obs(obs, step):
    print(f"\n{'='*70}")
    print(f" 观测详情  step={step}")
    print(f"{'='*70}")
    print(f" {'idx':>3s}  {'名称':<18s} {'值':>10s}   方向说明")
    print(f" {'-'*3}  {'-'*18} {'-'*10}   {'-'*40}")
    for idx, name, desc in OBS_LABELS:
        val = obs[idx]
        if idx in (3, 4, 29, 30):
            print(f" [{idx:>2d}]  {name} {val:>+10.4f} rad ({np.rad2deg(val):>+7.2f}°)   {desc}")
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
        self.latent_mean = latent_mean
        self.latent_std = latent_std
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


def load_quantized_model(checkpoint_path, device="cpu"):
    """加载 quantize_model.py 导出的 int8 量化 checkpoint。

    返回 (encoder, actor, vel_scale, norm_params) 用于构建 InferenceModel。
    """
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
            from quantize.quantize_model import _replace_linear_with_int8_calib
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
            from quantize.quantize_model import OBS_MEAN, OBS_STD, ENC_MEAN, ENC_STD
            print("[CHECK] 新模型: 使用手动归一化 (joystick.py 参数), 无 latent 归一化")
            inf = InferenceModel(model.encoder, model.actor,
                                 vel_scale=getattr(model, "vel_scale", 3.0))
            norm = {
                "enc_mean": ENC_MEAN,
                "enc_std": ENC_STD,
                "obs_mean": OBS_MEAN,
                "obs_std": OBS_STD,
            }
        return inf, norm


def normalize_obs(obs_history, obs, norm, device):
    """归一化观测并转为 tensor (与 EmpiricalNormalization 一致: eps=1e-2)。"""
    eps = 1e-2
    oh_t = torch.tensor(obs_history, dtype=torch.float32).unsqueeze(0).to(device)
    o_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
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

    theta_l = np.pi / 2 - phi0_l - pitch
    theta_r = np.pi / 2 - phi0_r - pitch

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
                        help="场景 (rmuc/flat)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="计算设备 (cpu/cuda)")
    parser.add_argument("--compare", action="store_true",
                        help="对比模式: 同时推理 float32 和 int8, 统计差异")
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
    args = parser.parse_args()

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
    print(f"加载模型: {model_file}")

    inf_model, norm = load_quantized_model(model_file, device)
    is_quantized = not args.no_quantize

    print("  actor_obs_normalizer (VMC相关维度):")
    for i, name in [(27, "theta_l"), (28, "theta_r"), (29, "L0_l"), (30, "L0_r")]:
        print(f"    {name}: mean={norm['obs_mean'][i]:.6f}, std={norm['obs_std'][i]:.6f}")
    print(f"    theta_l-theta_r mean diff: {norm['obs_mean'][27] - norm['obs_mean'][28]:.6f}")
    print(f"    L0_l-L0_r mean diff: {norm['obs_mean'][29] - norm['obs_mean'][30]:.6f}")

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
    obs_label_names = [
        "gyro_x", "gyro_y", "gyro_z",
        "pitch", "roll",
        "cmd_vel", "cmd_yaw", "cmd_height", "cmd_0",
        "act0", "act1", "act2", "act3", "act4", "act5",
        "sin_yaw", "cos_yaw",
        "dof_err_LF", "dof_err_RF", "dof_err_LR", "dof_err_RR",
        "dof_vel_LF", "dof_vel_RF", "dof_vel_LR", "dof_vel_RR", "dof_vel_LW", "dof_vel_RW",
        "theta_l", "theta_r", "L0_l", "L0_r",
    ]
    for i in range(NUM_OBS):
        print(f"    [{i:2d}] {obs_label_names[i]:15s}  mean={norm['obs_mean'][i]:+.6f}  std={norm['obs_std'][i]:+.6f}")

    # 对比模式下额外加载 float32 模型
    float_inf = None
    if args.compare:
        print("对比模式: 额外加载 float32 baseline...")
        float_inf, _ = load_quantized_model(model_file, device)
        print(f"  int8 模型已加载, float32 对比模型已加载 ✓")

    tag = "int8" if is_quantized else "f32"
    print(f"  模型类型: {tag}, 设备: {args.device}")

    # ── 初始状态 ──
    try:
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    except Exception:
        pass
    mj_data.qpos[2] = 0.8
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
    # 实物后电机标定偏移 -0.05 rad
    default_dof_pos[2] -= 0.05  # LR
    default_dof_pos[3] -= 0.05  # RR
    print(f"  default_dof_pos = {np.rad2deg(default_dof_pos)}° (LR/RR 偏移 -2.86°)")

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
    cmd_vel = [0.0]
    cmd_yaw_rate = [0.0]
    cmd_height = [0.29]
    target_yaw = [float(np.arctan2(2.0 * (mj_data.qpos[3] * mj_data.qpos[6] + mj_data.qpos[4] * mj_data.qpos[5]),
                                   1.0 - 2.0 * (mj_data.qpos[5] * mj_data.qpos[5] + mj_data.qpos[6] * mj_data.qpos[6])))]
    step_count = [0]
    ctrl_step_count = [0]
    paused = [False]
    debug_flag = [False]
    use_nn = [True]
    need_reset = [False]
    jump_active = [False]
    jump_timer = [0.0]

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

    # ── 控制参数 (与训练同步: sim_dt=0.005, ctrl_dt=0.02) ──
    CTRL_DT = 0.01
    SIM_DT = 0.005
    CTRL_STEPS = int(CTRL_DT / SIM_DT)  # 4 步仿真 = 1 次控制
    mj_model.opt.timestep = SIM_DT
    print(f"  sim_dt={mj_model.opt.timestep:.4f}s, ctrl_dt={CTRL_DT:.3f}s, ctrl_steps={CTRL_STEPS}")

    # 增加腿连杆质量 (模拟真机腿部惯量偏大)
    leg_body_names = [
        "left_front_link", "left_front_child1_link", "left_front_child2_link", "left_front_child3_link",
        "left_rear_link", "left_rear_child1_link",
        "right_front_link", "right_front_child1_link", "right_front_child2_link", "right_front_child3_link",
        "right_rear_link", "right_rear_child1_link",
    ]
    MASS_SCALE = 1.5
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
        print(f"  对比模式: float32 vs int8")
    print(f"  N=切换NN  Space=暂停  ↑↓←→=速度/转向  M=鼠标偏航  H=跳跃  P=PD实物位姿  Backspace=重置  Esc=退出")
    print(f"  {'='*60}\n")

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
        cmd_vel[0] = 0.0
        cmd_yaw_rate[0] = 0.0
        cmd_height[0] = 0.29
        jump_active[0] = False
        jump_timer[0] = 0.0
        qw, qx, qy, qz = mj_data.qpos[3:7]
        target_yaw[0] = float(np.arctan2(2.0 * (qw * qz + qx * qy),
                                         1.0 - 2.0 * (qy * qy + qz * qz)))
        print(">>> 重置 <<<")

    def controller(model, data):
        nonlocal last_actions, raw_action, raw_action_f32, filtered_action, filtered_action_f32, action_fifo, action_fifo_f32, pred_vel, jump_active, jump_timer
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
        gy = 2 * (qy * qz + qw * qx)
        pitch = np.arcsin(np.clip(gx, -1.0, 1.0))

        WHEEL_RADIUS = 0.06
        WHEELBASE_val = 0.380

        if is_ctrl_step:
            if pd_active[0]:
                last_actions = INJECT_ACT.copy()
                cmd_yaw_rate[0] = INJECT_CMD_YAW_RATE
            obs = build_observation(
                model, data, q, qvel, last_actions,
                cmd_vel[0], cmd_yaw_rate[0], cmd_height[0],
                yaw_error, default_dof_pos,
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

                last_actions = raw_action.copy()
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
                print(f"[DEBUG step {step_count[0]}] cmd_height={cmd_height[0]:.3f}  nn={mark}  原始观测 (raw obs):")
                obs_names = [
                    "gyro_x", "gyro_y", "gyro_z",
                    "pitch", "roll",
                    "cmd_vel", "cmd_yaw", "cmd_height", "cmd_0",
                    "act0", "act1", "act2", "act3", "act4", "act5",
                    "sin_yaw", "cos_yaw",
                    "dof_err_LF", "dof_err_RF", "dof_err_LR", "dof_err_RR",
                    "dof_vel_LF", "dof_vel_RF", "dof_vel_LR", "dof_vel_RR", "dof_vel_LW", "dof_vel_RW",
                    "theta_l", "theta_r", "L0_l", "L0_r",
                ]
                for i in range(NUM_OBS):
                    print(f"  [{i:2d}] {obs_names[i]:15s} = {obs[i]:+.6f}")
                if nn_active:
                    print(f"\n  归一化后观测 (norm obs):")
                    o_np = o_n.cpu().numpy()[0]
                    for i in range(NUM_OBS):
                        print(f"  [{i:2d}] {obs_names[i]:15s} = {o_np[i]:+.6f}")
                    print(f"\n  左右对称维度对比:")
                    pairs = [
                        (9, 10, "last_act LF/RF"),
                        (11, 12, "last_act LR/RR"),
                        (17, 18, "dof_err LF/RF"),
                        (19, 20, "dof_err LR/RR"),
                        (21, 22, "dof_vel LF/RF"),
                        (23, 24, "dof_vel LR/RR"),
                        (27, 28, "theta_l/r"),
                        (29, 30, "L0_l/r"),
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

        if use_nn[0]:
            if jump_active[0]:
                jump_timer[0] -= dt
                if jump_timer[0] <= 0.0:
                    jump_active[0] = False
                    jump_timer[0] = 0.0
                    print("  跳跃结束")
                    vmc = compute_vmc_feedforward(q, pitch, default_dof_pos)
                else:
                    vmc = compute_vmc_feedforward(q, pitch, default_dof_pos, force=VMC_JUMP_FORCE)
            else:
                vmc = compute_vmc_feedforward(q, pitch, default_dof_pos)
            tau = raw_action_to_torque(delayed_action, q, qvel, default_dof_pos, vmc)
        else:
            tau = np.zeros(6, dtype=np.float64)

        tau_raw = tau.copy()

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
            tau[:4] *= 1.0
            tau[4:] *= 1.0

        if debug_flag[0]:
            mode = "PD" if pd_active[0] else ("NN" if use_nn[0] else "零力矩")
            if use_nn[0]:
                dof_err = q[:4] - default_dof_pos
                print(f"  dof_err(°): [{np.rad2deg(dof_err[0]):+.2f}, {np.rad2deg(dof_err[1]):+.2f}, {np.rad2deg(dof_err[2]):+.2f}, {np.rad2deg(dof_err[3]):+.2f}]")
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
            vmc_f32 = compute_vmc_feedforward(q, pitch, default_dof_pos)
            tau_f32 = raw_action_to_torque(delayed_action_f32, q, qvel, default_dof_pos, vmc_f32)
            tau_diff = np.abs(tau_f32 - tau)
            if tau_diff.max() > compare_stats["max_torque_diff"]:
                compare_stats["max_torque_diff"] = tau_diff.max()
            compare_stats["sum_torque_diff"] += tau_diff.sum()

        for i in range(6):
            data.ctrl[act_ids[i]] = tau[i]

        if is_ctrl_step and step_count[0] % (100 * CTRL_STEPS) == 1:
            qw, qx, qy, qz = data.qpos[3:7]
            pitch = np.arcsin(np.clip(2.0*(qw*qy - qx*qz), -1.0, 1.0))
            roll = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
            q_lf, q_rf, q_lr, q_rr = q[0], q[1], q[2], q[3]
            clf, crf, clr, crr = q_lf - default_dof_pos[0], q_rf - default_dof_pos[1], q_lr - default_dof_pos[2], q_rr - default_dof_pos[3]
            phi1_l = clr + 5*np.pi/6; phi4_l = clf + np.pi/6
            phi1_r = crr + 5*np.pi/6; phi4_r = crf + np.pi/6
            L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
            L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)

            print(f"  [{step_count[0]:5d}] ({tag}) "
                  f"pitch={np.rad2deg(pitch):+6.2f}° roll={np.rad2deg(roll):+6.2f}° "
                  f"L0=({L0_l:.3f},{L0_r:.3f})m "
                  f"z={data.qpos[2]:.3f}m "
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
        elif keycode == 265:  # ↑
            if cmd_vel[0] >= 0:
                cmd_vel[0] = 0.0 if cmd_vel[0] > 0.0 else 2.2
            else:
                cmd_vel[0] = 0.0
            print(f"  前进/停止 cmd_vel={cmd_vel[0]:.1f} m/s")
        elif keycode == 264:  # ↓
            if cmd_vel[0] <= 0:
                cmd_vel[0] = 0.0 if cmd_vel[0] < 0.0 else -2.2
            else:
                cmd_vel[0] = 0.0
            print(f"  后退/停止 cmd_vel={cmd_vel[0]:.1f} m/s")
        elif keycode == 263:  # ←
            if cmd_yaw_rate[0] >= 0:
                cmd_yaw_rate[0] = 0.0 if cmd_yaw_rate[0] > 0.0 else 1.5
            else:
                cmd_yaw_rate[0] = 0.0
            print(f"  左转/直行 cmd_yaw_rate={cmd_yaw_rate[0]:.1f} rad/s")
        elif keycode == 262:  # →
            if cmd_yaw_rate[0] <= 0:
                cmd_yaw_rate[0] = 0.0 if cmd_yaw_rate[0] < 0.0 else -1.5
            else:
                cmd_yaw_rate[0] = 0.0
            print(f"  右转/直行 cmd_yaw_rate={cmd_yaw_rate[0]:.1f} rad/s")
        elif keycode == 72:  # H
            jump_active[0] = True
            jump_timer[0] = JUMP_DURATION
            print(f"  进入跳跃状态，给予{JUMP_DURATION:.1f}秒大VMC前馈 ({VMC_JUMP_FORCE:.0f}N)")
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
        elif keycode == 74:  # J: 注入自定义关节角度
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

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_callback,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
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

    print("退出.")


if __name__ == "__main__":
    main()