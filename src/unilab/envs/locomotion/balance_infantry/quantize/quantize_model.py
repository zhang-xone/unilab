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
    - 静态量化 (推荐): 手动 per-channel 权重量化, 激活保持 float32;
      实测动作误差约为动态量化的 1/3, 且不依赖 QuantizedCPU 后端

部署契约:
    - ONNX 输入是**已归一化**的 obs_history / obs; latent 归一化在 ONNX 内部
      完成 (训练时 actor_obs_normalizer 归一化的是 29+3 维 cat(obs, latent))
    - 导出后会自动校验 ONNX 与参考推理的一致性, 不一致直接失败
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
sys.path.insert(0, _SCRIPT_DIR)

from deploy_obs import build_observation as build_deploy_observation

from unilab.algos.torch.ppo.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence

# ═══════════════════════════════════════════════════════════════
# 训练产物参数 —— 由 quantize/sync_trained_params.py 生成并覆盖。
# 测试 / 校准 / 导出统一从这里取，不再各自硬编码。
# ═══════════════════════════════════════════════════════════════
from trained_params import (  # noqa: E402
    ACTION_SCALE_JOINT,
    ACTION_SCALE_WHEEL,
    ACTIVATION,
    ACTOR_HIDDEN_DIMS,
    ACTOR_MEAN,
    ACTOR_STD,
    CRITIC_HIDDEN_DIMS,
    CTRL_DT,
    CTRL_STEPS,
    ENC_MEAN,
    ENC_STD,
    ENCODER_HIDDEN_DIMS,
    JOINT_LIMITS,
    JUMP_DURATION,
    LATENT_DIM,
    LEG_KD,
    LEG_KP,
    NORM_EPS,
    NUM_ACTIONS,
    NUM_CRITIC_OBS,
    NUM_ENCODER_OBS,
    NUM_OBS,
    OBS_HISTORY_LENGTH,
    OBS_MEAN,
    OBS_STD,
    SIM_DT,
    SOURCE,
    VEL_MEAN,
    VEL_STD,
    VMC_FEEDFORWARD_FORCE,
    VMC_JUMP_FORCE,
    VMC_L1,
    VMC_L2,
    VMC_L3,
    VMC_L4,
    VMC_L5,
    WHEEL_KD,
)
from ckpt_meta import (  # noqa: E402
    extract_norm_stats,
    infer_dims_from_ckpt,
    warn_if_trained_params_mismatch,
)
from test_vmc_lqr import fk_fivebar  # noqa: E402

# 全局变量：由 load_float_model() 在加载后写入，供其他函数（校准/导出）使用。
DETECTED = {
    "num_obs": NUM_OBS,
    "num_actions": NUM_ACTIONS,
    "obs_history_length": OBS_HISTORY_LENGTH,
    "num_encoder_obs": NUM_ENCODER_OBS,
    "latent_dim": LATENT_DIM,
    "num_critic_obs": NUM_CRITIC_OBS,
    "encoder_hidden_dims": list(ENCODER_HIDDEN_DIMS),
    "actor_hidden_dims": list(ACTOR_HIDDEN_DIMS),
    "critic_hidden_dims": list(CRITIC_HIDDEN_DIMS),
    "activation": ACTIVATION,
}


def _infer_model_params(model_dict):
    """（向后兼容）从 float32 state_dict 推断结构并刷新全局 DETECTED。"""
    dims = infer_dims_from_ckpt({"model_state_dict": model_dict})
    # state_dict 推不出 activation；用训练 config 同步出的值，避免回退到 elu。
    dims["activation"] = ACTIVATION

    DETECTED.update({k: dims[k] for k in (
        "num_obs", "num_actions", "obs_history_length", "num_encoder_obs",
        "latent_dim", "num_critic_obs", "encoder_hidden_dims",
        "actor_hidden_dims", "critic_hidden_dims", "activation",
    ) if dims.get(k) is not None})

    print("[INFER] 从 checkpoint 推断模型结构:")
    print(f"  num_obs={dims['num_obs']}, num_actions={dims['num_actions']}, "
          f"obs_history_len={dims['obs_history_length']}")
    print(f"  num_encoder_obs={dims['num_encoder_obs']}, "
          f"num_critic_obs={dims['num_critic_obs']}, latent_dim={dims['latent_dim']}")
    print(f"  encoder_hidden_dims={dims['encoder_hidden_dims']}")
    print(f"  actor_hidden_dims={dims['actor_hidden_dims']}")
    print(f"  critic_hidden_dims={dims['critic_hidden_dims']}")

    return (dims["num_obs"], dims["num_actions"], dims["obs_history_length"],
            dims["num_encoder_obs"], dims["latent_dim"],
            dims["encoder_hidden_dims"], dims["actor_hidden_dims"],
            dims["critic_hidden_dims"], dims["activation"])


# collect_calibration_from_sim() 会设置该标记：校准轨迹已发散时, --calib_tune
# 会在错误分布上调 scale, 因此禁止。
CALIB_DIVERGED = False


def vmc_feedforward_torques(dof_pos, pitch, default_dof_pos):
    """VMC 重力前馈力矩 —— 与 ``jumpvf.py:_compute_vmc_torques`` /
    ``test_quantized.compute_vmc_feedforward`` 完全一致。

    纯前馈: F = feedforward_force / cos(theta) * ramp, T = 0，再经五连杆雅可比
    映射到 4 个腿关节力矩。校准必须叠加这一项，否则机器人在校准中失去
    支撑力（本 run 每条腿 80 N）直接塌陷，策略看到的全是分布外状态。
    """
    phi1_l = (dof_pos[2] - default_dof_pos[2]) + 5 * np.pi / 6
    phi1_r = (dof_pos[3] - default_dof_pos[3]) + 5 * np.pi / 6
    phi4_l = (dof_pos[0] - default_dof_pos[0]) + np.pi / 6
    phi4_r = (dof_pos[1] - default_dof_pos[1]) + np.pi / 6

    _, phi0_l, phi2_l, phi3_l, sigma1_l = fk_fivebar(phi1_l, phi4_l)
    _, phi0_r, phi2_r, phi3_r, sigma1_r = fk_fivebar(phi1_r, phi4_r)

    theta_l = (np.pi / 2 - phi0_l + pitch + np.pi) % (2.0 * np.pi) - np.pi
    theta_r = (np.pi / 2 - phi0_r + pitch + np.pi) % (2.0 * np.pi) - np.pi

    ramp_l = np.clip((np.pi / 3 - abs(theta_l)) / (np.pi / 6), 0.0, 1.0)
    ramp_r = np.clip((np.pi / 3 - abs(theta_r)) / (np.pi / 6), 0.0, 1.0)

    ff_l = VMC_FEEDFORWARD_FORCE / np.cos(np.clip(theta_l, -1.5, 1.5)) * ramp_l
    ff_r = VMC_FEEDFORWARD_FORCE / np.cos(np.clip(theta_r, -1.5, 1.5)) * ramp_r

    sigma2_l = np.sin(phi3_l - phi4_l)
    sigma3_l = np.sin(phi1_l - phi2_l)
    sigma2_r = np.sin(phi3_r - phi4_r)
    sigma3_r = np.sin(phi1_r - phi2_r)
    eps = 1e-8
    inv_sigma1_l = 1.0 / (sigma1_l + eps)
    inv_sigma1_r = 1.0 / (sigma1_r + eps)

    tau1_l = VMC_L1 * np.sin(phi0_l - phi3_l) * sigma3_l * inv_sigma1_l * ff_l
    tau4_l = VMC_L4 * np.sin(phi0_l - phi2_l) * sigma2_l * inv_sigma1_l * ff_l
    tau1_r = VMC_L1 * np.sin(phi0_r - phi3_r) * sigma3_r * inv_sigma1_r * ff_r
    tau4_r = VMC_L4 * np.sin(phi0_r - phi2_r) * sigma2_r * inv_sigma1_r * ff_r

    vmc = np.zeros(6, dtype=np.float64)
    vmc[0] = tau4_l   # LF
    vmc[1] = tau4_r   # RF
    vmc[2] = tau1_l   # LR
    vmc[3] = tau1_r   # RR
    return np.nan_to_num(vmc, nan=0.0, posinf=40.0, neginf=-40.0)


def collect_calibration_from_sim(float_model, norm_dict, scene="rmuc",
                                  num_steps=500, device="cpu"):
    """用 float32 策略在 MuJoCo 里闭环跑 N 个控制步，收集量化校准输入。

    观测/控制约定来自 :mod:`deploy_obs`（与 ``test_quantized.py`` 同源），
    因此收集到的分布和部署时真正喂给网络的输入一致。

    Returns:
        (enc_inputs, act_inputs):
            enc_inputs  shape (N, num_encoder_obs)  —— 归一化后的 obs_history
            act_inputs  shape (N, num_obs+latent_dim) —— 归一化后的 [obs, latent]
    """
    global CALIB_DIVERGED

    import mujoco

    scene_dir = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..",
                             "assets", "robots", "balance_infantry")
    scene_map = {
        "rmuc": os.path.join(scene_dir, "scene_rmuc.xml"),
        "flat": os.path.join(scene_dir, "scene_flat.xml"),
        # 这两个场景的 home keyframe 带五连杆站立姿态；flat 场景的 qpos 是全 0，
        # 直接拿来校准会让机器人从直腿姿态起步并摔倒。
        "step": os.path.join(scene_dir, "scene_step.xml"),
        "rough": os.path.join(scene_dir, "scene_rough.xml"),
    }
    scene_path = scene_map.get(scene, scene_map["rmuc"])
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

    has_keyframe = False
    try:
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
            has_keyframe = True
    except Exception:
        pass
    if not has_keyframe:
        # 没有 keyframe 的场景才退回固定初始高度
        mj_data.qpos[2] = 0.8
    mujoco.mj_forward(mj_model, mj_data)

    default_dof_pos = np.array([mj_data.qpos[joint_qpos[i]] for i in range(4)], dtype=np.float32)

    enc_mean = torch.tensor(norm_dict["enc_mean"], device=device)
    enc_std = torch.tensor(norm_dict["enc_std"], device=device)
    obs_mean = torch.tensor(norm_dict["obs_mean"], device=device)
    obs_std = torch.tensor(norm_dict["obs_std"], device=device)
    latent_mean = torch.tensor(norm_dict["latent_mean"], device=device)
    latent_std = torch.tensor(norm_dict["latent_std"], device=device)
    eps = NORM_EPS

    float_model.to(device)
    float_model.eval()

    mj_model.opt.timestep = SIM_DT

    obs_history = deque(maxlen=DETECTED["obs_history_length"])
    last_actions = np.zeros(6, dtype=np.float32)
    cmd_vel = 0.0
    cmd_yaw_rate = 0.0
    cmd_height = 0.29

    enc_inputs = []
    act_inputs = []
    pitch_log = []
    phys_step = 0

    print(f"  收集校准数据 (目标 {num_steps} 步)...")
    while len(enc_inputs) < num_steps:
        q = np.array([mj_data.qpos[idx] for idx in joint_qpos])
        qvel = np.array([mj_data.qvel[joint_dofadr[i]] for i in range(6)])
        is_ctrl_step = (phys_step % CTRL_STEPS == 0)

        if is_ctrl_step:
            pitch_log.append(float(np.arcsin(np.clip(
                2.0 * (mj_data.qpos[4] * mj_data.qpos[6] - mj_data.qpos[3] * mj_data.qpos[5]),
                -1.0, 1.0))))
            obs = build_deploy_observation(
                gyro=mj_data.qvel[3:6],
                quat=mj_data.qpos[3:7],
                dof_pos=q,
                dof_vel=qvel,
                last_actions=last_actions,
                cmd_vel=cmd_vel,
                cmd_yaw_rate=cmd_yaw_rate,
                cmd_height=cmd_height,
                default_dof_pos=default_dof_pos,
            )

            obs_history.append(obs)
            if len(obs_history) == DETECTED["obs_history_length"]:
                obs_hist = np.concatenate(list(obs_history)).astype(np.float32)
            else:
                obs_hist = np.tile(obs, DETECTED["obs_history_length"]).astype(np.float32)

            obs_hist_t = torch.tensor(obs_hist, device=device).unsqueeze(0).clamp(-100.0, 100.0)
            obs_t = torch.tensor(obs, device=device).unsqueeze(0).clamp(-100.0, 100.0)
            oh_n = (obs_hist_t - enc_mean) / (enc_std + eps)
            o_n = (obs_t - obs_mean) / (obs_std + eps)

            with torch.no_grad():
                latent = float_model.encoder(oh_n) * float_model.vel_scale
                latent_n = (latent - latent_mean) / (latent_std + eps)
                actor_input = torch.cat([o_n, latent_n], dim=-1)
                action = float_model.actor(actor_input)

            enc_inputs.append(oh_n.cpu().numpy()[0])
            act_inputs.append(actor_input.cpu().numpy()[0])
            last_actions = action.cpu().numpy()[0].astype(np.float32)

            # 随机切换命令, 覆盖更多状态分布
            if len(enc_inputs) % 50 == 0:
                cmd_vel = float(np.random.uniform(-1.0, 1.0))
                cmd_yaw_rate = float(np.random.uniform(-1.0, 1.0))
                cmd_height = float(np.random.uniform(0.25, 0.35))

        # 关节 PD + 轮子速度 PD + VMC 重力前馈（与 jumpvf.py:_pre_step_motor_control 一致）
        target_joint = np.clip(
            default_dof_pos + last_actions[:4] * ACTION_SCALE_JOINT,
            JOINT_LIMITS[:4, 0],
            JOINT_LIMITS[:4, 1],
        )
        target_wheel = last_actions[4:] * ACTION_SCALE_WHEEL
        pitch_now = float(np.arcsin(np.clip(
            2.0 * (mj_data.qpos[4] * mj_data.qpos[6] - mj_data.qpos[3] * mj_data.qpos[5]),
            -1.0, 1.0,
        )))
        vmc_tau = vmc_feedforward_torques(
            np.asarray(q[:4], dtype=np.float64), pitch_now, default_dof_pos,
        )
        for i in range(4):
            mj_data.ctrl[i] = (
                LEG_KP[i] * (target_joint[i] - q[i]) - LEG_KD[i] * qvel[i] + vmc_tau[i]
            )
        for i in range(2):
            mj_data.ctrl[4 + i] = WHEEL_KD[i] * (target_wheel[i] - qvel[4 + i])

        mujoco.mj_step(mj_model, mj_data)
        phys_step += 1

    enc_inputs = np.array(enc_inputs, dtype=np.float32)
    act_inputs = np.array(act_inputs, dtype=np.float32)
    print(f"  收集完成: enc={enc_inputs.shape}, act={act_inputs.shape}")
    print(f"  enc range: [{enc_inputs.min():.3f}, {enc_inputs.max():.3f}]")
    print(f"  act range: [{act_inputs.min():.3f}, {act_inputs.max():.3f}]")
    print(f"  原始 pitch 范围: [{min(pitch_log):+.3f}, {max(pitch_log):+.3f}] rad")

    # 归一化后 |x| > 20 说明策略在采集过程中已经发散, 数据不在训练分布内
    CALIB_DIVERGED = max(float(np.abs(enc_inputs).max()), float(np.abs(act_inputs).max())) > 20.0
    if CALIB_DIVERGED:
        print("  [WARN] 校准数据偏离训练分布 (|normalized| > 20): 策略在采集过程中发散了。\n"
              "         这份数据只用于报告, 不能用来微调 scale; 换 --calib_scene flat, "
              "或先用 test_quantized.py 确认策略能站稳。")
    return enc_inputs, act_inputs


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

    所有模型结构参数从 checkpoint 的 state_dict 自动推断，无需手动配置。

    Returns:
        (model, norm_dict) 其中 norm_dict 包含 enc_mean, enc_std, obs_mean, obs_std
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_dict = ckpt.get("model_state_dict", ckpt)

    dims = infer_dims_from_ckpt(ckpt)
    # activation 无法从 state_dict 推断，以训练 config 同步出的值为准。
    dims["activation"] = ACTIVATION
    _infer_model_params(model_dict)  # 刷新全局 DETECTED（与上面的 dims 一致）

    num_obs = dims["num_obs"]
    num_actions = dims["num_actions"]
    obs_history_length = dims["obs_history_length"]
    num_encoder_obs = dims["num_encoder_obs"]
    latent_dim = dims["latent_dim"]
    encoder_hidden_dims = dims["encoder_hidden_dims"]
    actor_hidden_dims = dims["actor_hidden_dims"]
    critic_hidden_dims = dims["critic_hidden_dims"]
    activation = dims["activation"]
    num_critic_obs = dims["num_critic_obs"]

    model = ActorCriticSequence(
        num_obs=num_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        num_encoder_obs=num_encoder_obs,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        obs_normalization=True,
        use_real_vel=False,
    ).to(device)
    model.load_state_dict(model_dict, strict=False)
    model.eval()

    # 从检查点提取训练好的归一化系数（与 sync_trained_params.py 同一实现）
    stats = extract_norm_stats(ckpt, dims)

    prefill_normalization(
        model,
        stats["enc_mean_per_frame"],
        stats["enc_std_per_frame"],
        stats["actor_mean"],
        stats["actor_std"],
    )
    return model, {
        "enc_mean": stats["enc_mean"],
        "enc_std": stats["enc_std"],
        "obs_mean": stats["obs_mean"],
        "obs_std": stats["obs_std"],
        "latent_mean": stats["latent_mean"],
        "latent_std": stats["latent_std"],
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
            base_err = (y_ref[:, i] - y_q_base[:, i]).pow(2).mean().item()
            best_err, best_s = base_err, scale[i].item()
            for factor in np.linspace(0.90, 1.10, 11):
                s = max(scale[i].item() * factor, 1e-8)
                w_q = torch.round(w[i:i+1] / s).clamp(-128, 127).float()
                y_q = X @ (w_q * s).T
                err = (y_ref[:, i] - y_q[:, 0]).pow(2).mean().item()
                if err < best_err:
                    best_err, best_s = err, s
            if best_err < base_err:
                improved += 1
            scale[i] = best_s
        scale = scale.clamp(min=1e-8)

        # 验证校准后总误差
        w_q_new = torch.round(w / scale).clamp(-128, 127).float()
        y_q_new = X @ (w_q_new * scale).T
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


def _collect_layer_inputs(module, calib_inputs, prefix=""):
    """逐层跑校准数据，收集每层 Linear 的输入张量。

    任何非 Linear 子模块都按真实前向执行，避免激活函数类型变化时静默
    产生错位的校准数据。

    Returns:
        dict: {层名: 输入张量}
    """
    layer_inputs = {}
    with torch.no_grad():
        x = calib_inputs
        for name, child in module.named_children():
            key = f"{prefix}{name}"
            if isinstance(child, nn.Linear):
                layer_inputs[key] = x.clone()
                x = child(x)
            elif isinstance(child, nn.Sequential):
                layer_inputs.update(_collect_layer_inputs(child, x, prefix=f"{key}."))
                x = child(x)
            else:
                x = child(x)
    return layer_inputs


def _replace_linear_with_int8_calib(module, calib_inputs=None, use_calib=False,
                                    layer_inputs=None, prefix=""):
    """递归替换 Sequential 中的 nn.Linear 为 Int8Linear，使用校准数据。"""
    if layer_inputs is None:
        layer_inputs = (_collect_layer_inputs(module, calib_inputs)
                        if calib_inputs is not None else {})
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            calib_x = layer_inputs.get(f"{prefix}{name}")
            setattr(module, name, _quantize_linear_per_channel(child, calib_x, use_calib))
        elif isinstance(child, nn.Sequential):
            _replace_linear_with_int8_calib(
                child, None, use_calib, layer_inputs, prefix=f"{prefix}{name}."
            )


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
        calib_samples: 校准样本数上限
        calib_data: (enc_inputs, act_inputs) 校准数据; 为 None 时按纯权重量化
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
        print("  无校准数据, 仅基于权重量化...")

    _replace_linear_with_int8_calib(qmodel.encoder, enc_calib, use_calib)
    _replace_linear_with_int8_calib(qmodel.actor, act_calib, use_calib)

    qmodel.to(device)
    qmodel.eval()

    # 计算量化后参数数量 (含 buffer 中的 int8 权重)
    enc_params = sum(p.numel() for p in qmodel.encoder.state_dict().values())
    act_params = sum(p.numel() for p in qmodel.actor.state_dict().values())
    print(f"  int8 encoder 参数: {enc_params:,}")
    print(f"  int8 actor   参数: {act_params:,}")
    print("  量化完成 ✓")

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
    """部署网络: encoder + latent 归一化 + actor。

    输入是**已归一化**的 obs_history 与 obs（MCU 用 norm_params.h 完成）。
    latent 归一化必须在这里做：训练时 ``actor_obs_normalizer`` 是把
    ``cat(obs, latent)`` 当作一个 29+3 维向量一起归一化的，latent 分量
    并非 (mean=0, std=1)。漏掉它会直接改变策略输出（实测动作偏差 ~0.8）。
    """

    def __init__(self, encoder, actor, vel_scale, latent_mean=None, latent_std=None,
                 eps=NORM_EPS):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.vel_scale = vel_scale
        self.eps = eps
        if latent_mean is not None and latent_std is not None:
            self.register_buffer("latent_mean", torch.as_tensor(
                np.asarray(latent_mean, dtype=np.float32)).reshape(1, -1))
            self.register_buffer("latent_std", torch.as_tensor(
                np.asarray(latent_std, dtype=np.float32)).reshape(1, -1))
        else:
            self.register_buffer("latent_mean", None)
            self.register_buffer("latent_std", None)

    def forward(self, obs_history_norm, obs_norm):
        latent = self.encoder(obs_history_norm) * self.vel_scale
        if self.latent_mean is not None:
            latent = (latent - self.latent_mean) / (self.latent_std + self.eps)
        return self.actor(torch.cat([obs_norm, latent], dim=-1))


_DYNAMIC_LINEAR_CLS = getattr(torch.ao.nn.quantized.dynamic, "Linear", ())
_STATIC_LINEAR_CLS = getattr(torch.ao.nn.quantized, "Linear", ())


def _bias_of_quantized(module):
    """quantized Linear 的 ``bias`` 是方法而不是 tensor，取真实值。"""
    bias = getattr(module, "bias", None)
    if callable(bias):
        bias = bias()
    return bias


def _dequantize_module_inplace(module):
    """递归将 Int8Linear / 动态或静态 quantized Linear 替换为普通 nn.Linear。

    注意：``torch.ao.nn.quantized.dynamic.Linear.bias`` 是**方法**（返回 tensor），
    不能当作 tensor 属性使用。
    """
    for name, child in module.named_children():
        if isinstance(child, Int8Linear):
            w = child.weight_q.float() * child.scale
            new_linear = nn.Linear(child.in_features, child.out_features,
                                   bias=child.bias is not None)
            with torch.no_grad():
                new_linear.weight.copy_(w)
                if child.bias is not None:
                    new_linear.bias.copy_(child.bias)
            setattr(module, name, new_linear)
        elif isinstance(child, nn.Sequential):
            _dequantize_module_inplace(child)
        elif _DYNAMIC_LINEAR_CLS and isinstance(child, _DYNAMIC_LINEAR_CLS):
            w = torch.dequantize(child.weight())
            bias = _bias_of_quantized(child)
            new_linear = nn.Linear(child.in_features, child.out_features, bias=bias is not None)
            with torch.no_grad():
                new_linear.weight.copy_(w)
                if bias is not None:
                    new_linear.bias.copy_(bias)
            setattr(module, name, new_linear)
        elif _STATIC_LINEAR_CLS and isinstance(child, _STATIC_LINEAR_CLS):
            w = torch.dequantize(child.weight())
            bias = _bias_of_quantized(child)
            new_linear = nn.Linear(child.in_features, child.out_features, bias=bias is not None)
            with torch.no_grad():
                new_linear.weight.copy_(w)
                if bias is not None:
                    new_linear.bias.copy_(bias)
            setattr(module, name, new_linear)
        elif hasattr(child, "weight_q"):
            raise TypeError(f"未支持的量化层 {name}: {type(child)}")


def _export_norm_header(norm_dict, vel_scale, output_dir):
    """导出归一化参数为 C 头文件 (MCU 上 float32 计算)。

    ``latent_mean/latent_std`` 由 ONNX 内部使用；导出只为核对，
    MCU 端不要再对 latent 归一化一次。
    """
    path = os.path.join(output_dir, "norm_params.h")
    names = ["enc_mean", "enc_std", "obs_mean", "obs_std", "latent_mean", "latent_std"]
    with open(path, "w") as f:
        f.write("// 自动生成: 归一化参数 (MCU 上 float32 计算)\n")
        f.write(f"// 来源: trained_params.py → run={SOURCE['run']} / "
                f"ckpt={SOURCE['checkpoint']}\n")
        f.write(f"// 公式: y = (x - mean) / (std + {NORM_EPS:g})\n\n")
        f.write("#pragma once\n\n")
        for name in names:
            if name not in norm_dict:
                continue
            arr = np.asarray(norm_dict[name]).reshape(-1)
            f.write(f"static const float {name}[{len(arr)}] = {{\n    ")
            f.write(", ".join(f"{v:.8f}f" for v in arr))
            f.write("\n};\n\n")

        obs_dim = len(np.asarray(norm_dict["obs_mean"]).reshape(-1))
        enc_dim = len(np.asarray(norm_dict["enc_mean"]).reshape(-1))
        latent_dim = len(np.asarray(norm_dict.get("latent_mean", [])).reshape(-1))
        f.write(f"#define ENC_DIM {enc_dim}\n")
        f.write(f"#define OBS_DIM {obs_dim}\n")
        f.write(f"#define LATENT_DIM {latent_dim}\n")
        f.write(f"#define OBS_HISTORY_LENGTH {DETECTED['obs_history_length']}\n")
        f.write(f"#define VEL_SCALE {float(vel_scale):.8f}f\n")
        f.write(f"#define NORM_EPS {NORM_EPS:g}f\n\n")
        f.write("// MCU 推理流程:\n")
        f.write("// 1. oh = (obs_history - enc_mean) / (enc_std + NORM_EPS)\n")
        f.write("// 2. o  = (obs - obs_mean) / (obs_std + NORM_EPS)\n")
        f.write("// 3. 喂入 ONNX -> action[NUM_ACTIONS]\n")
        f.write("//    ONNX 内部已完成: latent = (encoder(oh)*VEL_SCALE - latent_mean)\n")
        f.write("//                              / (latent_std + NORM_EPS)\n")
        f.write("// MCU 不要再对 latent 归一化一次。\n")
    print(f"  归一化头文件: {path}")


def export_ctrl_header(output_dir):
    """导出控制参数为 C 头文件，MCU 部署时与 RLobs.c 保持一致。"""
    path = os.path.join(output_dir, "ctrl_params.h")
    with open(path, "w") as f:
        f.write("// 自动生成: 控制参数 (与 trained_params.py 一致)\n")
        f.write(f"// 来源: trained_params.py → run={SOURCE['run']} / "
                f"ckpt={SOURCE['checkpoint']}\n")
        f.write("// 使用: #include \"ctrl_params.h\"\n\n")
        f.write("#pragma once\n\n")

        f.write("/* ── 控制周期 (s) ── */\n")
        f.write(f"#define CTRL_DT {CTRL_DT:.4f}f\n")
        f.write(f"#define SIM_DT  {SIM_DT:.4f}f\n\n")

        f.write("/* ── PD 参数 (per-joint) ── */\n")
        names = ["LF", "RF", "LR", "RR"]
        for i, n in enumerate(names):
            f.write(f"#define LEG_KP_{n}  {LEG_KP[i]:.1f}f\n")
        for i, n in enumerate(names):
            f.write(f"#define LEG_KD_{n}  {LEG_KD[i]:.1f}f\n")
        for i, n in enumerate(["LW", "RW"]):
            f.write(f"#define WHEEL_KD_{n} {WHEEL_KD[i]:.1f}f\n\n")

        f.write("/* ── Action 缩放 ── */\n")
        f.write(f"#define ACTION_SCALE_JOINT {ACTION_SCALE_JOINT:.1f}f\n")
        f.write(f"#define ACTION_SCALE_WHEEL {ACTION_SCALE_WHEEL:.1f}f\n\n")

        f.write("/* ── 关节限位 (rad) ── */\n")
        f.write(f"#define JOINT_LIMIT_LF_LO {JOINT_LIMITS[0][0]:.2f}f\n")
        f.write(f"#define JOINT_LIMIT_LF_HI {JOINT_LIMITS[0][1]:.2f}f\n")
        f.write(f"#define JOINT_LIMIT_RF_LO {JOINT_LIMITS[1][0]:.2f}f\n")
        f.write(f"#define JOINT_LIMIT_RF_HI {JOINT_LIMITS[1][1]:.2f}f\n")
        f.write(f"#define JOINT_LIMIT_LR_LO {JOINT_LIMITS[2][0]:.2f}f\n")
        f.write(f"#define JOINT_LIMIT_LR_HI {JOINT_LIMITS[2][1]:.2f}f\n")
        f.write(f"#define JOINT_LIMIT_RR_LO {JOINT_LIMITS[3][0]:.2f}f\n")
        f.write(f"#define JOINT_LIMIT_RR_HI {JOINT_LIMITS[3][1]:.2f}f\n\n")

        f.write("/* ── VMC 参数 (与 mujoco.yaml control_config.vmc 一致) ── */\n")
        f.write(f"#define VMC_L1               {VMC_L1:.3f}f\n")
        f.write(f"#define VMC_L2               {VMC_L2:.3f}f\n")
        f.write(f"#define VMC_L3               {VMC_L3:.3f}f\n")
        f.write(f"#define VMC_L4               {VMC_L4:.3f}f\n")
        f.write(f"#define VMC_L5               {VMC_L5:.1f}f\n")
        f.write(f"#define VMC_FEEDFORWARD_FORCE {VMC_FEEDFORWARD_FORCE:.1f}f\n")
        f.write(f"#define VMC_JUMP_FORCE       {VMC_JUMP_FORCE:.1f}f\n")
        f.write(f"#define JUMP_DURATION        {JUMP_DURATION:.2f}f\n\n")

        f.write("#define ACTION_DELAY 2\n")
    print(f"  控制参数头文件: {path}")


def _export_onnx(model, onnx_path, num_encoder_obs, num_obs):
    """导出部署 ONNX，batch 维动态。"""
    oh = torch.randn(1, num_encoder_obs)
    o = torch.randn(1, num_obs)
    torch.onnx.export(
        model, (oh, o), onnx_path,
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
    print(f"  ONNX 已保存: {os.path.abspath(onnx_path)}")


def _verify_onnx_parity(onnx_path, encoder, actor, vel_scale, norm_dict,
                        num_encoder_obs, num_obs, tol=5e-3, n=64):
    """校验 ONNX 部署图 == (encoder → latent 归一化 → actor) 参考实现。

    这条检查专门防止“导出时漏掉 latent 归一化”这类回归：训练时
    ``actor_obs_normalizer`` 归一化的是 32 维 [obs, latent]，latent 分量
    并非零均值单位方差。
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [WARN] 未安装 onnxruntime, 跳过 ONNX 一致性校验")
        return None

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed_names = [i.name for i in sess.get_inputs()]
    rng = np.random.default_rng(0)
    oh = rng.standard_normal((n, num_encoder_obs)).astype(np.float32)
    o = rng.standard_normal((n, num_obs)).astype(np.float32)
    onnx_act = sess.run(None, {feed_names[0]: oh, feed_names[1]: o})[0]

    latent_mean = torch.as_tensor(
        np.asarray(norm_dict["latent_mean"], dtype=np.float32)).reshape(1, -1)
    latent_std = torch.as_tensor(
        np.asarray(norm_dict["latent_std"], dtype=np.float32)).reshape(1, -1)
    with torch.no_grad():
        latent = encoder(torch.from_numpy(oh)) * float(vel_scale)
        latent = (latent - latent_mean) / (latent_std + NORM_EPS)
        ref = actor(torch.cat([torch.from_numpy(o), latent], dim=-1)).numpy()

    max_diff = float(np.abs(onnx_act - ref).max())
    print(f"  ONNX 一致性: max|onnx - reference| = {max_diff:.6f}")
    if max_diff > tol:
        raise SystemExit(
            f"[ERROR] ONNX 部署图与参考推理不一致 (max={max_diff:.6f} > {tol})。\n"
            "检查 ExportModel 是否包含 latent 归一化 (latent_mean/latent_std)。"
        )
    return max_diff


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
    warn_if_trained_params_mismatch(input_path)
    print(f"加载 float32 模型: {input_path}")

    float_model, norm_dict = load_float_model(input_path, device)

    f32_enc_params = sum(p.numel() for p in float_model.encoder.parameters())
    f32_act_params = sum(p.numel() for p in float_model.actor.parameters())
    print(f"  float32 encoder 参数: {f32_enc_params:,}")
    print(f"  float32 actor   参数: {f32_act_params:,}")
    print(f"  float32 encoder+actor 合计: {f32_enc_params + f32_act_params:,}")

    # 输出路径处理: 自动从输入名派生 model_XXXX.pt → quantize/<项目>/model_XXXX_int8.pt
    input_name = os.path.splitext(os.path.basename(input_path))[0]
    if args.output is None:
        project_name = os.path.basename(os.path.dirname(input_path))
        output_dir = os.path.join(_PARENT_DIR, "quantize", project_name)
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

    latent_mean = norm_dict.get("latent_mean")
    latent_std = norm_dict.get("latent_std")
    num_encoder_obs = DETECTED["num_encoder_obs"]
    num_obs = DETECTED["num_obs"]

    # 导出 float32 版 ONNX (量化前)
    if not args.no_export_onnx:
        print("\n导出 float32 ONNX...")
        export_model_f32 = ExportModel(
            float_model.encoder, float_model.actor, float_model.vel_scale,
            latent_mean, latent_std,
        )
        export_model_f32.eval()
        f32_onnx_path = os.path.join(output_dir, f"{input_name}_deploy.onnx")
        _export_onnx(export_model_f32, f32_onnx_path, num_encoder_obs, num_obs)
        _verify_onnx_parity(f32_onnx_path, float_model.encoder, float_model.actor,
                            float_model.vel_scale, norm_dict, num_encoder_obs, num_obs,
                            tol=2e-3)

    print(f"输出 int8 模型: {output_path}")

    if args.static:
        print("正在进行 int8 静态量化...")
        calib_data = None
        if args.calib_from_sim:
            sim_enc, sim_act = collect_calibration_from_sim(
                float_model, norm_dict, scene=args.calib_scene,
                num_steps=args.calib_samples, device=device,
            )
            if args.calib_tune and CALIB_DIVERGED:
                raise SystemExit(
                    "[ERROR] 校准轨迹已发散, --calib_tune 会在错误分布上调 scale。\n"
                    "先让策略在 --calib_scene 下站稳 (或换场景), 或去掉 --calib_tune。"
                )
            np.save(os.path.join(output_dir, "test_enc_inputs.npy"), sim_enc)
            np.save(os.path.join(output_dir, "test_act_inputs.npy"), sim_act)
            print(f"  校准数据已保存: test_enc_inputs.npy {sim_enc.shape}, "
                  f"test_act_inputs.npy {sim_act.shape}")
            calib_data = (sim_enc, sim_act)
        qmodel = quantize_static_and_save(float_model, norm_dict, output_path, device,
                                          calib_samples=args.calib_samples,
                                          calib_data=calib_data,
                                          use_calib=args.calib_tune)
    else:
        print("正在进行 int8 动态量化...")
        qmodel = quantize_and_save(float_model, norm_dict, output_path, device)

    # 验证：加载并跑一次前向
    print("\n验证量化模型...")
    ckpt = torch.load(output_path, map_location=device, weights_only=False)
    print(f"  quantized={ckpt.get('quantized')}, dtype={ckpt.get('quantize_dtype')}")

    # 用随机输入跑一次推理，确认可以正常 forward
    dummy_obs_hist = torch.randn(1, num_encoder_obs, device=device)
    dummy_obs = torch.randn(1, num_obs, device=device)
    with torch.no_grad():
        latent = qmodel.encoder(dummy_obs_hist) * qmodel.vel_scale
        actor_input = torch.cat([dummy_obs, latent], dim=-1)
        action = qmodel.actor(actor_input)
    print(f"  随机输入推理成功: action shape={action.shape}, "
          f"range=[{action.min().item():.3f}, {action.max().item():.3f}]")
    print("验证通过 ✓")

    # 导出 ONNX 部署文件
    if not args.no_export_onnx:
        print("\n导出 ONNX 部署文件...")
        _dequantize_module_inplace(qmodel.encoder)
        _dequantize_module_inplace(qmodel.actor)
        export_model = ExportModel(
            qmodel.encoder, qmodel.actor, float_model.vel_scale,
            latent_mean, latent_std,
        )
        export_model.eval()

        _export_norm_header(norm_dict, float_model.vel_scale, output_dir)
        export_ctrl_header(output_dir)

        onnx_path = os.path.join(output_dir, f"{input_name}_int8_deploy.onnx")
        _export_onnx(export_model, onnx_path, num_encoder_obs, num_obs)
        _verify_onnx_parity(onnx_path, qmodel.encoder, qmodel.actor,
                            float_model.vel_scale, norm_dict, num_encoder_obs, num_obs,
                            tol=5e-3)

        print(f"  输出文件: {output_dir}/")
        print("    model_XXXX.pt               ← 原始 float32 checkpoint")
        print("    model_XXXX_int8.pt          ← 量化 PyTorch 模型 (test_quantized 用)")
        print("    model_XXXX_int8_deploy.onnx ← 部署 ONNX (含 latent 归一化)")
        print("    norm_params.h               ← 归一化 C 头文件")
        print("    ctrl_params.h               ← 控制参数 C 头文件")

    print(f"\n{'='*60}")
    print("完成 ✓")


if __name__ == "__main__":
    main()