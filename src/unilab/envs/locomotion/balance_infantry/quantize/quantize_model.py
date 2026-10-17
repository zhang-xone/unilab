#!/usr/bin/env python3
"""PPO 策略 int8 量化 —— 支持动态量化与静态量化。

对 encoder 和 actor 的 Linear 层做 int8 量化，保存为可直接加载的 checkpoint。

用法:
    cd /path/to/balance_infantry

    # 动态量化 (默认)
    python quantize/quantize_model.py --input /home/xxx/ubuntu_software/unilab/UniLab/logs/rsl_rl_sequence/BalanceInfantryJoystickFlat/2026-09-05_21-08-28_mujoco/model_4000.pt --output quantize/ppo_int8.pt

    # 静态量化 (更稳定，需要校准)
    python quantize/quantize_model.py --input /home/xxx/ubuntu_software/unilab/UniLab/logs/rsl_rl_sequence/BalanceInfantryJoystickFlat/2026-09-05_21-08-28_mujoco/model_4000.pt --output quantize/ppo_int8_static.pt --static

    # 静态量化 + 仿真数据校准 (最准确)
    python quantize/quantize_model.py --input /home/xxx/ubuntu_software/unilab/UniLab/logs/rsl_rl_sequence/BalanceInfantryJoystickFlat/2026-09-05_21-08-28_mujoco/model_4000.pt --output quantize/ppo_int8_static.pt --static --calib_from_sim

    # 静态量化 + 更多校准样本
    python quantize/quantize_model.py --input /home/xxx/ubuntu_software/unilab/UniLab/logs/rsl_rl_sequence/BalanceInfantryJoystickFlat/2026-09-05_21-08-28_mujoco/model_4000.pt --output quantize/ppo_int8_static.pt --static --calib_samples 5000

工作原理:
    - 动态量化: 权重 int8, 激活值 float32, 运行时逐 batch 估算 scale
    - 静态量化: 权重 int8, 激活值 int8, 通过校准数据预先确定 scale/zero_point
      静态量化激活值范围更稳定, 抖动通常更小
"""

import argparse
import copy
import os
import shutil
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.join(_SCRIPT_DIR, "..")
_UNILAB_SRC = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "..")

sys.path.insert(0, os.path.join(_UNILAB_SRC, "src"))
sys.path.insert(0, _PARENT_DIR)
sys.path.insert(0, os.path.join(_PARENT_DIR, "lqr"))

from unilab.algos.torch.ppo.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence
from test_vmc_lqr import fk_fivebar

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
# 手动归一化系数 (与 export_for_ppo.py 一致)
# ═══════════════════════════════════════════════════════════════
OBS_MEAN = np.array([
    0.0, 0.0, 0.0,     # gyro
    0.0,                # pitch
    0.0,                # roll
    0.0, 0.0, 0.29, 0.05,  # cmd
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # last_actions
    0.0, 1.0,           # sin/cos yaw
    0.0, 0.0, 0.0, 0.0, # dof_pos_err
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # dof_vel
    0.0, 0.0,           # vmc_theta
    0.28, 0.28,         # vmc_L0
], dtype=np.float32)

OBS_STD = np.array([
    0.34, 0.62, 2.0,    # gyro
    0.15,               # pitch
    0.10,               # roll
    1.5, 2.0, 0.08, 0.22,  # cmd
    0.6, 0.6, 0.6, 0.6, 0.6, 0.6,  # last_actions
    0.6, 0.6,           # sin/cos yaw
    0.3, 0.3, 0.3, 0.3, # dof_pos_err
    1.5, 1.5, 1.5, 1.5, 22.0, 22.0,  # dof_vel
    0.4, 0.4,           # vmc_theta
    0.05, 0.05,         # vmc_L0
], dtype=np.float32)

VEL_MEAN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
VEL_STD = np.array([1.5, 0.5, 0.5], dtype=np.float32)

ENC_MEAN = np.tile(OBS_MEAN, OBS_HISTORY_LENGTH).astype(np.float32)
ENC_STD = np.tile(OBS_STD, OBS_HISTORY_LENGTH).astype(np.float32)
ACTOR_MEAN = np.concatenate([OBS_MEAN, VEL_MEAN]).astype(np.float32)
ACTOR_STD = np.concatenate([OBS_STD, VEL_STD]).astype(np.float32)


def collect_calibration_from_sim(float_model, norm_dict, scene="rmuc",
                                  num_steps=500, device="cpu"):
    """从仿真收集校准数据，替代随机数据校准。

    在 MuJoCo 中运行 float32 模型 N 步，收集每步的归一化观测历史和
    观测+latent，作为静态量化的校准输入。

    Args:
        float_model: float32 ActorCriticSequence (obs_normalization=False)
        norm_dict: 归一化系数 {"enc_mean", "enc_std", "obs_mean", "obs_std", ...}
        scene: 场景 ("rmuc" / "flat")
        num_steps: 收集的控制步数
        device: 计算设备

    Returns:
        (enc_inputs, act_inputs): 两个 numpy 数组, shape (N, 165) 和 (N, 36)
    """
    import mujoco

    # ── 加载场景 ──
    scene_dir = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..",
                             "assets", "robots", "balance_infantry")
    scene_map = {
        "rmuc": os.path.join(scene_dir, "scene_rmuc.xml"),
        "flat": os.path.join(scene_dir, "scene_flat.xml"),
    }
    scene_path = scene_map.get(scene, scene_map["rmuc"])
    print(f"  校准场景: {scene_path}")

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

    # ── 复位 ──
    try:
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    except Exception:
        pass
    mj_data.qpos[2] = 0.8
    mujoco.mj_forward(mj_model, mj_data)

    default_dof_pos = np.array([mj_data.qpos[joint_qpos[i]] for i in range(4)], dtype=np.float32)

    # ── 归一化系数 ──
    enc_mean = torch.tensor(norm_dict["enc_mean"], device=device)
    enc_std = torch.tensor(norm_dict["enc_std"], device=device)
    obs_mean = torch.tensor(norm_dict["obs_mean"], device=device)
    obs_std = torch.tensor(norm_dict["obs_std"], device=device)
    eps = 1e-6

    float_model.to(device)
    float_model.eval()

    # ── 仿真参数 ──
    CTRL_DT = 0.02
    SIM_DT = 0.005
    CTRL_STEPS = int(CTRL_DT / SIM_DT)
    mj_model.opt.timestep = SIM_DT

    # ── 状态变量 ──
    obs_history = deque(maxlen=OBS_HISTORY_LENGTH)
    last_actions = np.zeros(6, dtype=np.float32)
    pred_vel = 0.0
    cmd_vel = 0.0
    cmd_yaw_rate = 0.0
    cmd_height = 0.29
    qw, qx, qy, qz = mj_data.qpos[3:7]
    target_yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz)))

    enc_inputs = []
    act_inputs = []
    phys_step = 0

    print(f"  收集校准数据 (目标 {num_steps} 步)...")
    while len(enc_inputs) < num_steps:
        q = np.array([mj_data.qpos[idx] for idx in joint_qpos])
        qvel = np.array([mj_data.qvel[joint_dofadr[i]] for i in range(6)])

        is_ctrl_step = (phys_step % CTRL_STEPS == 0)

        # 更新 yaw target
        qw, qx, qy, qz = mj_data.qpos[3:7]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))
        if is_ctrl_step:
            target_yaw += cmd_yaw_rate * CTRL_DT
            target_yaw = np.arctan2(np.sin(target_yaw), np.cos(target_yaw))
        yaw_error = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))

        if is_ctrl_step:
            # 构建观测 (与 test_quantized.py 一致)
            gyro = mj_data.qvel[3:6].astype(np.float32)
            gx_ = 2 * (qx * qz - qw * qy)
            pitch = float(np.arcsin(np.clip(gx_, -1.0, 1.0)))
            gx, gy_val = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx)
            roll = float(np.arctan2(gy_val, np.sqrt(1.0 - gx * gx)))
            dof_pos_err = (default_dof_pos - q[:4]).astype(np.float32)
            dof_vel = qvel[:6].astype(np.float32)

            # 五连杆运动学 (与 test_quantized.py build_observation 一致)
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

            # 归一化
            obs_hist_t = torch.tensor(obs_hist, device=device).unsqueeze(0)
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)
            oh_n = (obs_hist_t - enc_mean) / (enc_std + eps)
            o_n = (obs_t - obs_mean) / (obs_std + eps)

            # 推理获取 latent
            with torch.no_grad():
                latent = float_model.encoder(oh_n) * float_model.vel_scale

            # 存储校准数据
            enc_inputs.append(oh_n.cpu().numpy()[0])
            act_inputs.append(torch.cat([o_n, latent], dim=-1).cpu().numpy()[0])

            # 动作 (用于下一帧的 last_actions)
            with torch.no_grad():
                action = float_model.actor(torch.cat([o_n, latent], dim=-1))
            last_actions = action.cpu().numpy()[0].astype(np.float32)

            # 随机切换命令, 覆盖更多状态分布
            if len(enc_inputs) % 50 == 0:
                cmd_vel = np.random.uniform(-1.0, 1.0)
                cmd_yaw_rate = np.random.uniform(-1.0, 1.0)
                cmd_height = np.random.uniform(0.25, 0.35)

        # 施加控制力矩
        if use_nn_ctrl(phys_step, is_ctrl_step):
            target_joint = default_dof_pos + last_actions[:4] * 0.5
            target_wheel = last_actions[4:] * 50.0
            # PD 控制
            LEG_KP = (50.0, 50.0, 50.0, 50.0)
            LEG_KD = (3.2, 3.2, 3.2, 3.2)
            WHEEL_KD = (0.1, 0.1)
            for i in range(4):
                mj_data.ctrl[i] = LEG_KP[i] * (target_joint[i] - q[i]) - LEG_KD[i] * qvel[i]
            for i in range(2):
                mj_data.ctrl[4 + i] = WHEEL_KD[i] * (target_wheel[i] - qvel[4 + i])

        mujoco.mj_step(mj_model, mj_data)
        phys_step += 1

    enc_inputs = np.array(enc_inputs, dtype=np.float32)
    act_inputs = np.array(act_inputs, dtype=np.float32)
    print(f"  收集完成: enc={enc_inputs.shape}, act={act_inputs.shape}")
    print(f"  enc range: [{enc_inputs.min():.3f}, {enc_inputs.max():.3f}]")
    print(f"  act range: [{act_inputs.min():.3f}, {act_inputs.max():.3f}]")
    return enc_inputs, act_inputs


def use_nn_ctrl(phys_step, is_ctrl_step):
    return True  # 始终使用 NN 控制


def prefill_normalization(model, enc_mean, enc_std, actor_mean, actor_std, count=100000):
    """预填 EmpiricalNormalization buffer 为手动系数。"""
    eps = 1e-6
    for name, mean, std in [
        ("encoder_obs_normalizer", enc_mean, enc_std),
        ("actor_obs_normalizer", actor_mean, actor_std),
    ]:
        norm = getattr(model, name, None)
        if norm is not None and hasattr(norm, "_mean"):
            mean_t = torch.tensor(mean, dtype=torch.float32).unsqueeze(0)
            std_t = torch.tensor(np.maximum(std, eps), dtype=torch.float32).unsqueeze(0)
            norm._mean.copy_(mean_t)
            norm._var.copy_(std_t.pow(2))
            norm._std.copy_(std_t)
            norm.count = torch.tensor([count], dtype=torch.int64)


def load_float_model(checkpoint_path, device="cpu"):
    """加载 float32 PPO ActorCriticSequence 模型，并提取训练归一化系数。

    Returns:
        (model, norm_dict) 其中 norm_dict 包含 enc_mean, enc_std, obs_mean, obs_std
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        obs_normalization=True,
        use_real_vel=False,
    ).to(device)
    model.load_state_dict(model_dict, strict=False)
    model.eval()

    # 从检查点提取训练好的归一化系数
    sd = model_dict
    enc_mean_raw = sd["encoder_obs_normalizer._mean"]
    enc_std_raw = sd["encoder_obs_normalizer._std"]
    if enc_mean_raw.shape[-1] == NUM_ENCODER_OBS:
        enc_mean_per_frame = enc_mean_raw.reshape(1, OBS_HISTORY_LENGTH, NUM_OBS).mean(dim=1).flatten().cpu().numpy()
        enc_std_per_frame = enc_std_raw.reshape(1, OBS_HISTORY_LENGTH, NUM_OBS).mean(dim=1).flatten().cpu().numpy()
    else:
        enc_mean_per_frame = enc_mean_raw.flatten().cpu().numpy()
        enc_std_per_frame = enc_std_raw.flatten().cpu().numpy()
    obs_mean_full = sd["actor_obs_normalizer._mean"].flatten().cpu().numpy()
    obs_std_full = sd["actor_obs_normalizer._std"].flatten().cpu().numpy()

    enc_mean = np.tile(enc_mean_per_frame, OBS_HISTORY_LENGTH).astype(np.float32)
    enc_std = np.tile(enc_std_per_frame, OBS_HISTORY_LENGTH).astype(np.float32)
    actor_mean = obs_mean_full.astype(np.float32)
    actor_std = obs_std_full.astype(np.float32)

    prefill_normalization(model, enc_mean_per_frame.astype(np.float32), enc_std_per_frame.astype(np.float32), actor_mean, actor_std)
    return model, {
        "enc_mean": enc_mean,
        "enc_std": enc_std,
        "obs_mean": obs_mean_full[:NUM_OBS].astype(np.float32),  # 只取 obs 部分
        "obs_std": obs_std_full[:NUM_OBS].astype(np.float32),
        "latent_mean": obs_mean_full[NUM_OBS:].astype(np.float32),  # 后 latent_dim 维: latent 归一化
        "latent_std": obs_std_full[NUM_OBS:].astype(np.float32),
    }


def quantize_and_save(float_model, norm_dict, output_path, device="cpu"):
    """对 encoder 和 actor 做动态 int8 量化，保存 checkpoint。

    Args:
        float_model: float32 ActorCriticSequence
        norm_dict: 训练好的归一化系数
        output_path: 输出 .pt 路径
        device: 目标设备

    Returns:
        量化后的 ActorCriticSequence
    """
    qmodel = copy.deepcopy(float_model).to("cpu")

    qmodel.encoder = torch.ao.quantization.quantize_dynamic(
        qmodel.encoder,
        {nn.Linear},
        dtype=torch.qint8,
    )
    qmodel.actor = torch.ao.quantization.quantize_dynamic(
        qmodel.actor,
        {nn.Linear},
        dtype=torch.qint8,
    )

    qmodel.to(device)
    qmodel.eval()

    save_dict = {
        "encoder_state_dict": qmodel.encoder.state_dict(),
        "actor_state_dict": qmodel.actor.state_dict(),
        "vel_scale": float_model.vel_scale,
        "enc_mean": norm_dict["enc_mean"],
        "enc_std": norm_dict["enc_std"],
        "obs_mean": norm_dict["obs_mean"],
        "obs_std": norm_dict["obs_std"],
        "latent_mean": norm_dict["latent_mean"],
        "latent_std": norm_dict["latent_std"],
        "quantized": True,
        "quantize_dtype": "qint8",
    }
    torch.save(save_dict, output_path)
    return qmodel


class Int8Linear(nn.Module):
    """手动 int8 量化 Linear 层: 权重存 int8 + per-channel float32 scale。
    
    前向: 权重反量化到 float32 → matmul → 输出 float32。
    激活值保持 float32，避免 QuantizedCPU 后端的 ELU 兼容性问题。
    """
    def __init__(self, in_features, out_features, weight_q, scale, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_q", weight_q)       # torch.int8 [out, in]
        self.register_buffer("scale", scale)             # torch.float32 [out, 1]
        if bias is not None:
            self.register_buffer("bias", bias)           # torch.float32 [out]
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight_q.float() * self.scale
        return F.linear(x, w, self.bias)


def _quantize_linear_per_channel(linear, calib_inputs=None, use_calib=False):
    """对单个 nn.Linear 做 per-channel int8 量化。

    Args:
        linear: nn.Linear 层
        calib_inputs: 校准输入 (N, in_features) 或 None
        use_calib: 是否用校准数据微调 scale (保守: 0.9x~1.1x)
    Returns:
        Int8Linear 量化层
    """
    w = linear.weight.data.float()
    out_features, in_features = w.shape

    w_max, _ = w.abs().max(dim=1, keepdim=True)
    scale = (w_max / 127.0).clamp(min=1e-8)

    if calib_inputs is not None and calib_inputs.shape[0] > 0 and use_calib:
        X = calib_inputs.float()
        y_ref = X @ w.T  # [N, out]
        w_q_base = torch.round(w / scale).clamp(-128, 127).float()
        y_q_base = X @ (w_q_base * scale).T
        base_err = (y_ref - y_q_base).pow(2).mean().item()

        # 保守校准: 每个通道在 0.9x~1.1x 范围内搜索最优 scale
        improved = 0
        for i in range(out_features):
            best_err = (y_ref[:, i] - y_q_base[:, i]).pow(2).mean().item()
            best_s = scale[i].item()
            for factor in np.linspace(0.90, 1.10, 11):
                s = scale[i].item() * factor
                s = max(s, 1e-8)
                w_q = torch.round(w[i:i+1] / s).clamp(-128, 127).float()
                y_q = X @ (w_q * s).T
                err = (y_ref[:, i] - y_q[:, 0]).pow(2).mean().item()
                if err < best_err:
                    best_err = err
                    best_s = s
                    improved += 1
            scale[i] = best_s
        scale = scale.clamp(min=1e-8)

        # 验证校准后总误差
        w_q_new = torch.round(w / scale).clamp(-128, 127).float()
        y_q_new = X @ (w_q_new * scale).T
        new_err = (y_ref - y_q_new).pow(2).mean().item()
        rel_base = (y_ref - y_q_base).norm() / (y_ref.norm() + 1e-8)
        rel_new = (y_ref - y_q_new).norm() / (y_ref.norm() + 1e-8)
        print(f"    误差: {rel_base.item():.6f} → {rel_new.item():.6f} (校准改善 {improved}/{out_features} 通道)")

    elif calib_inputs is not None and calib_inputs.shape[0] > 0:
        X = calib_inputs.float()
        y_ref = X @ w.T
        w_q = torch.round(w / scale).clamp(-128, 127).float()
        y_q = X @ (w_q * scale).T
        rel_err = (y_ref - y_q).norm() / (y_ref.norm() + 1e-8)
        print(f"    误差: {rel_err.item():.6f}")

    w_q = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
    bias = linear.bias.data.clone() if linear.bias is not None else None
    return Int8Linear(in_features, out_features, w_q, scale, bias)


def _collect_layer_inputs(module, calib_inputs):
    """逐层跑校准数据，收集每层 Linear 的输入张量。

    Returns:
        dict: {layer_name: input_tensor (N, in_features)}
    """
    layer_inputs = {}
    with torch.no_grad():
        x = calib_inputs
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                layer_inputs[name] = x.clone()  # 保存当前层的输入
                x = child(x)  # 前向传播
            elif isinstance(child, (nn.ELU, nn.ReLU)):
                x = child(x)
            elif isinstance(child, nn.Sequential):
                # 递归处理
                sub = _collect_layer_inputs(child, x)
                layer_inputs.update({f"{name}.{k}": v for k, v in sub.items()})
                # 跑完整个 Sequential 后的输出
                x = child(x)
    return layer_inputs


def _replace_linear_with_int8_calib(module, calib_inputs=None, use_calib=False):
    """递归替换 Sequential 中的 nn.Linear 为 Int8Linear，使用校准数据。"""
    if calib_inputs is not None:
        layer_inputs = _collect_layer_inputs(module, calib_inputs)
    else:
        layer_inputs = {}
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            calib_x = layer_inputs.get(name, None)
            setattr(module, name, _quantize_linear_per_channel(child, calib_x, use_calib))
        elif isinstance(child, nn.Sequential):
            sub_calib = layer_inputs.get(name, None)
            _replace_linear_with_int8_calib(child, sub_calib, use_calib)


def quantize_static_and_save(float_model, norm_dict, output_path, device="cpu",
                              calib_samples=2000, calib_data=None, use_calib=False):
    """手动 per-channel int8 量化: 只量化 Linear 权重，激活值保持 float32。

    优点:
    - 不需要 QuantizedCPU 后端，ELU 等激活函数无兼容性问题
    - per-channel 量化精度高于 per-tensor
    - 不需要 prepare/convert 流程

    Args:
        float_model: float32 ActorCriticSequence
        norm_dict: 训练好的归一化系数
        output_path: 输出 .pt 路径
        device: 目标设备
        calib_samples: 校准样本数 (仅 calib_data=None 时生效)
        calib_data: (enc_inputs, act_inputs) 校准数据, 若为 None 则用随机数据
        use_calib: 是否用校准数据微调 scale (保守 0.9x~1.1x)
    """
    qmodel = copy.deepcopy(float_model).to("cpu")
    qmodel.eval()

    if calib_data is not None:
        enc_inputs, act_inputs = calib_data
        n = min(len(enc_inputs), calib_samples)
        label = "校准微调" if use_calib else "校准验证"
        print(f"  {label} ({n} 样本)...")
        enc_calib = torch.tensor(enc_inputs[:n], dtype=torch.float32)
        act_calib = torch.tensor(act_inputs[:n], dtype=torch.float32)
    else:
        enc_calib = None
        act_calib = None
        print(f"  无校准数据, 仅基于权重量化...")

    _replace_linear_with_int8_calib(qmodel.encoder, enc_calib, use_calib)
    _replace_linear_with_int8_calib(qmodel.actor, act_calib, use_calib)

    qmodel.to(device)
    qmodel.eval()

    # 计算量化后参数数量 (含 buffer 中的 int8 权重)
    enc_params = sum(p.numel() for p in qmodel.encoder.state_dict().values())
    act_params = sum(p.numel() for p in qmodel.actor.state_dict().values())
    print(f"  int8 encoder 参数: {enc_params:,}")
    print(f"  int8 actor   参数: {act_params:,}")
    print(f"  量化完成 ✓")

    save_dict = {
        "encoder_state_dict": qmodel.encoder.state_dict(),
        "actor_state_dict": qmodel.actor.state_dict(),
        "vel_scale": float_model.vel_scale,
        "enc_mean": norm_dict["enc_mean"],
        "enc_std": norm_dict["enc_std"],
        "obs_mean": norm_dict["obs_mean"],
        "obs_std": norm_dict["obs_std"],
        "latent_mean": norm_dict["latent_mean"],
        "latent_std": norm_dict["latent_std"],
        "quantized": True,
        "quantize_dtype": "qint8_per_channel",
    }
    torch.save(save_dict, output_path)
    print(f"  已保存: {os.path.abspath(output_path)}")
    return qmodel


class ExportModel(nn.Module):
    """纯网络: encoder + actor (不含归一化)。MCU 上做完归一化后喂入。"""

    def __init__(self, encoder, actor, vel_scale):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.vel_scale = vel_scale

    def forward(self, obs_history_norm, obs_norm):
        latent = self.encoder(obs_history_norm) * self.vel_scale
        actor_input = torch.cat([obs_norm, latent], dim=-1)
        return self.actor(actor_input)


def _dequantize_module_inplace(module):
    """递归将 Int8Linear / DynamicQuantizedLinear 替换为普通 nn.Linear。"""
    for name, child in module.named_children():
        if isinstance(child, Int8Linear):
            w = child.weight_q.float() * child.scale
            new_linear = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
            new_linear.weight.data = w
            if child.bias is not None:
                new_linear.bias.data = child.bias.clone()
            setattr(module, name, new_linear)
        elif isinstance(child, nn.Sequential):
            _dequantize_module_inplace(child)
        elif hasattr(child, "weight") and hasattr(child, "scale"):
            w = torch.dequantize(child.weight())
            new_linear = nn.Linear(child.in_features, child.out_features, bias=hasattr(child, "bias"))
            new_linear.weight.data = w
            if hasattr(child, "bias") and child.bias is not None:
                new_linear.bias.data = child.bias.clone()
            setattr(module, name, new_linear)


def _export_norm_header(norm_dict, vel_scale, output_dir):
    """导出归一化参数为 C 头文件。"""
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
        f.write("#define ENC_DIM 165\n")
        f.write("#define OBS_DIM 33\n")
        f.write("#define OBS_HISTORY_LENGTH 5\n")
        f.write("#define NORM_EPS 1e-2f\n\n")
        f.write("// MCU 推理流程:\n")
        f.write("// 1. oh = (obs_history - enc_mean) / (enc_std + NORM_EPS)\n")
        f.write("// 2. o  = (obs - obs_mean) / (obs_std + NORM_EPS)\n")
        f.write("// 3. 喂入 ONNX (encoder+actor) → action[6]\n")
    print(f"  归一化头文件: {path}")


def main():
    parser = argparse.ArgumentParser(description="PPO 策略 int8 量化")
    parser.add_argument("--input", type=str, required=True,
                        help="输入 float32 PPO checkpoint (.pt)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 int8 量化 checkpoint (默认自动从输入名派生: model_XXXX.pt → model_XXXX_int8.pt)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="计算设备 (cpu/cuda)")
    parser.add_argument("--static", action="store_true",
                        help="使用静态量化 (默认动态量化)")
    parser.add_argument("--calib_samples", type=int, default=2000,
                        help="静态量化校准样本数 (默认 2000)")
    parser.add_argument("--calib_from_sim", action="store_true",
                        help="从仿真收集校准数据 (默认随机数据)")
    parser.add_argument("--calib_scene", type=str, default="rmuc",
                        help="校准场景 (rmuc/flat, 仅 --calib_from_sim 时生效)")
    parser.add_argument("--calib_tune", action="store_true",
                        help="用校准数据微调 scale (保守 0.9x~1.1x, 默认仅验证不修改)")
    parser.add_argument("--no_export_onnx", action="store_true",
                        help="跳过 ONNX 导出 (默认自动导出 .onnx + norm_params.h)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用, 回退到 CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    # 输入路径处理
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(_PARENT_DIR, input_path)
    input_path = os.path.normpath(input_path)
    if not os.path.isfile(input_path):
        print(f"[ERROR] 模型文件不存在: {input_path}")
        sys.exit(1)
    print(f"加载 float32 模型: {input_path}")

    float_model, norm_dict = load_float_model(input_path, device)

    f32_enc_params = sum(p.numel() for p in float_model.encoder.parameters())
    f32_act_params = sum(p.numel() for p in float_model.actor.parameters())
    print(f"  float32 encoder 参数: {f32_enc_params:,}")
    print(f"  float32 actor   参数: {f32_act_params:,}")
    print(f"  float32 encoder+actor 合计: {f32_enc_params + f32_act_params:,}")

    # 输出路径处理: 自动从输入名派生 model_XXXX.pt → quantize/<项目>/model_XXXX_int8.pt
    if args.output is None:
        project_name = os.path.basename(os.path.dirname(input_path))
        output_dir = os.path.join(_PARENT_DIR, "quantize", project_name)
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.normpath(os.path.join(output_dir, f"{input_name}_int8.pt"))
    else:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(_PARENT_DIR, output_path)
        output_dir = os.path.dirname(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 复制原始 .pt 到导出目录
    src_pt_name = os.path.basename(input_path)
    dst_pt = os.path.join(output_dir, src_pt_name)
    if os.path.abspath(input_path) != os.path.abspath(dst_pt):
        shutil.copy2(input_path, dst_pt)
        print(f"复制原始模型: {dst_pt}")

    # 导出 float32 版 ONNX (量化前)
    if not args.no_export_onnx:
        export_model_f32 = ExportModel(float_model.encoder, float_model.actor, float_model.vel_scale)
        export_model_f32.eval()
        oh = torch.randn(1, NUM_ENCODER_OBS)
        o = torch.randn(1, NUM_OBS)
        f32_onnx_path = os.path.join(output_dir, f"{input_name}_deploy.onnx")
        torch.onnx.export(
            export_model_f32, (oh, o), f32_onnx_path,
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
        print(f"  float32 ONNX: {f32_onnx_path}")

    print(f"输出 int8 模型: {output_path}")

    if args.static:
        print(f"正在进行 int8 静态量化...")
        calib_data = None
        if args.calib_from_sim:
            sim_enc, sim_act = collect_calibration_from_sim(
                float_model, norm_dict, scene=args.calib_scene,
                num_steps=args.calib_samples, device=device,
            )
            # 保存校准测试数据
            np.save(os.path.join(output_dir, "test_enc_inputs.npy"), sim_enc)
            np.save(os.path.join(output_dir, "test_act_inputs.npy"), sim_act)
            print(f"  测试数据已保存: test_enc_inputs.npy ({sim_enc.shape})")
            print(f"  测试数据已保存: test_act_inputs.npy ({sim_act.shape})")
            # 混合: 仿真数据 + 随机数据，各占总样本一半
            rnd_enc = np.random.randn(args.calib_samples, NUM_ENCODER_OBS).astype(np.float32)
            rnd_act = np.random.randn(args.calib_samples, NUM_OBS + LATENT_DIM).astype(np.float32)
            calib_data = (
                np.concatenate([sim_enc, rnd_enc], axis=0),
                np.concatenate([sim_act, rnd_act], axis=0),
            )
            print(f"  混合校准数据: 仿真 {args.calib_samples} + 随机 {args.calib_samples} = {len(calib_data[0])} 样本")
        qmodel = quantize_static_and_save(float_model, norm_dict, output_path, device,
                                          calib_samples=args.calib_samples * 2,
                                          calib_data=calib_data,
                                          use_calib=args.calib_tune)
    else:
        print(f"正在进行 int8 动态量化...")
        qmodel = quantize_and_save(float_model, norm_dict, output_path, device)

    # 验证：加载并跑一次前向
    print(f"\n验证量化模型...")
    ckpt = torch.load(output_path, map_location=device, weights_only=False)
    print(f"  quantized={ckpt.get('quantized')}, dtype={ckpt.get('quantize_dtype')}")

    # 用随机输入跑一次推理，确认可以正常 forward
    dummy_obs_hist = torch.randn(1, NUM_ENCODER_OBS, device=device)
    dummy_obs = torch.randn(1, NUM_OBS, device=device)
    with torch.no_grad():
        latent = qmodel.encoder(dummy_obs_hist) * qmodel.vel_scale
        actor_input = torch.cat([dummy_obs, latent], dim=-1)
        action = qmodel.actor(actor_input)
    print(f"  随机输入推理成功: action shape={action.shape}, "
          f"range=[{action.min().item():.3f}, {action.max().item():.3f}]")
    print("验证通过 ✓")

    # 导出 ONNX 部署文件
    if not args.no_export_onnx:
        print(f"\n导出 ONNX 部署文件...")
        _dequantize_module_inplace(qmodel.encoder)
        _dequantize_module_inplace(qmodel.actor)
        export_model = ExportModel(qmodel.encoder, qmodel.actor, float_model.vel_scale)
        export_model.eval()

        _export_norm_header(norm_dict, float_model.vel_scale, output_dir)

        oh = torch.randn(1, NUM_ENCODER_OBS)
        o = torch.randn(1, NUM_OBS)

        onnx_path = os.path.join(output_dir, f"{input_name}_int8_deploy.onnx")
        torch.onnx.export(
            export_model, (oh, o), onnx_path,
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
        print(f"  ONNX 已保存: {onnx_path}")
        print(f"  输出文件: {output_dir}/")
        print(f"    model_XXXX_int8.pt         ← 量化 PyTorch 模型")
        print(f"    model_XXXX_int8_deploy.onnx ← 量化 ONNX 部署文件")
        print(f"    norm_params.h              ← 归一化 C 头文件")

    print(f"\n{'='*60}")
    print(f"完成 ✓")


if __name__ == "__main__":
    main()