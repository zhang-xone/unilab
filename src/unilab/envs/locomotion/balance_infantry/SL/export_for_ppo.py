#!/usr/bin/env python3
"""将 SL 预训练权重导出为 PPO 兼容的 checkpoint。

功能:
  1. 加载 SL 预训练的 encoder/actor 权重
  2. 创建 ActorCriticSequence (结构与 PPO 配置完全一致)
  3. 将手动归一化系数预填入 EmpiricalNormalization buffer
  4. 保存为 PPO 可加载的 checkpoint

用法:
    cd /home/xxx/ubuntu_software/unilab/UniLab/src/unilab/envs/locomotion/balance_infantry
    python SL/export_for_ppo.py --input SL/log/combined_pretrain_xxx.pth --output SL/log/ppo_init.pt
"""

import argparse
import os
import sys
import torch
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UNILAB_ROOT = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "..")
sys.path.insert(0, os.path.join(_UNILAB_ROOT, "src"))

from unilab.algos.torch.ppo.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence

# ═══════════════════════════════════════════════════════════════
# 手动归一化系数 (与 SL 训练 train_sl.py / combined_train.py 完全一致)
# ═══════════════════════════════════════════════════════════════

OBS_MEAN = np.array([
    0.0, 0.0, 0.0,               # 0-2:  gyro
    0.0,                          # 3:    pitch
    0.0,                          # 4:    roll
    0.0, 0.0, 0.29, 0.05,        # 5-8:  cmd(vx,vyaw,height,jump)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 9-14: last_actions
    0.0, 0.0,                     # 15-16: wheel_pos_err
    0.0, 0.0,                     # 17-18: sin/cos yaw_error
    0.0, 0.0, 0.0, 0.0,          # 19-22: dof_pos_err
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 23-28: dof_vel
    0.3, 0.3,                     # 29-30: vmc_theta
    0.28, 0.28,                   # 31-32: vmc_L0
], dtype=np.float32)

OBS_STD = np.array([
    1.5, 1.5, 1.5,               # 0-2:  gyro
    0.15,                         # 3:    pitch
    0.10,                         # 4:    roll
    1.5, 2.0, 0.08, 0.22,        # 5-8:  cmd
    0.6, 0.6, 0.6, 0.6, 0.6, 0.6,  # 9-14: last_actions
    0.3, 0.3,                     # 15-16: wheel_pos_err
    0.6, 0.6,                     # 17-18: sin/cos yaw_error
    0.3, 0.3, 0.3, 0.3,          # 19-22: dof_pos_err
    3.0, 3.0, 3.0, 3.0, 3.0, 3.0,  # 23-28: dof_vel
    0.4, 0.4,                     # 29-30: vmc_theta
    0.05, 0.05,                   # 31-32: vmc_L0
], dtype=np.float32)

# 编码器归一化: 5 帧 obs 拼接
ENC_MEAN = np.tile(OBS_MEAN, 5).astype(np.float32)
ENC_STD = np.tile(OBS_STD, 5).astype(np.float32)

# 速度归一化 (encoder 预测目标, 也是 actor 输入的 latent 部分)
VEL_MEAN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
VEL_STD = np.array([1.5, 0.5, 0.5], dtype=np.float32)

# actor 输入 = obs(33) + latent/privileged_vel(3) = 36
ACTOR_MEAN = np.concatenate([OBS_MEAN, VEL_MEAN]).astype(np.float32)
ACTOR_STD = np.concatenate([OBS_STD, VEL_STD]).astype(np.float32)

# 避免除零
EPS = 1e-6


def prefill_normalizer(normalizer, mean, std, count=100000):
    """预填 EmpiricalNormalization 的 buffer 为手动值。

    Args:
        normalizer: EmpiricalNormalization 实例
        mean: (D,) 均值
        std:  (D,) 标准差
        count: 预设的样本计数 (足够大则后续更新影响小)
    """
    mean_t = torch.tensor(mean, dtype=torch.float32).unsqueeze(0)
    std_t = torch.tensor(np.maximum(std, EPS), dtype=torch.float32).unsqueeze(0)
    var_t = std_t.pow(2)

    normalizer._mean.copy_(mean_t)
    normalizer._var.copy_(var_t)
    normalizer._std.copy_(std_t)
    normalizer.count = count


def main():
    parser = argparse.ArgumentParser(description="导出 SL 预训练权重为 PPO checkpoint")
    parser.add_argument("--input", type=str, required=True, help="SL 预训练 .pth 文件")
    parser.add_argument("--output", type=str, default="ppo_init.pt", help="输出 PPO checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--norm_count", type=int, default=100000,
                        help="EmpiricalNormalization 预设 count (越大归一化越稳定)")
    args = parser.parse_args()

    # 1. 加载 SL checkpoint
    print(f"加载 SL checkpoint: {args.input}")
    sl_ckpt = torch.load(args.input, map_location=args.device, weights_only=False)

    # 2. 创建 ActorCriticSequence
    #    与 conf/ppo_sequence/task/balance_infantry_joystick_flat/mujoco.yaml 完全一致
    #    注意: use_real_vel=True 时 actor 输入用 privileged_vel 替代 encoder latent
    actor_critic = ActorCriticSequence(
        num_obs=33,
        num_critic_obs=147 + 3,       # privileged_obs(147) + latent_dim(3) = 150
        num_actions=6,
        num_encoder_obs=165,
        latent_dim=3,
        encoder_hidden_dims=[128, 64],
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
        orthogonal_init=True,
        init_noise_std=0.5,
        obs_normalization=True,       # 与 PPO 配置 empirical_normalization=True 对应
        use_real_vel=True,
    )

    # 3. 加载 SL 权重
    actor_critic.encoder.load_state_dict(sl_ckpt["encoder_state_dict"])
    print("✅ encoder 权重已加载")

    actor_critic.actor.load_state_dict(sl_ckpt["actor_state_dict"])
    print("✅ actor 权重已加载")

    # 4. 预填手动归一化系数到 EmpiricalNormalization buffer
    # 环境已做 obs_manual_norm → 观测(33维) 已归一化，用 identity
    # encoder 输出已经是归一化速度 (real_vel - VEL_MEAN)/VEL_STD，actor 训练时直接用
    # 所以 actor_obs_normalizer 全部用 identity，不再对速度部分做额外归一化
    ACTOR_MEAN = np.zeros(36, dtype=np.float32)
    ACTOR_STD = np.ones(36, dtype=np.float32)
    prefill_normalizer(actor_critic.actor_obs_normalizer, ACTOR_MEAN, ACTOR_STD, args.norm_count)
    print(f"✅ actor_obs_normalizer:  identity (obs+latent 均已归一化)")

    # 编码器输入也是环境归一化后的观测历史，用 identity
    ENC_MEAN = np.zeros(165, dtype=np.float32)
    ENC_STD = np.ones(165, dtype=np.float32)
    prefill_normalizer(actor_critic.encoder_obs_normalizer, ENC_MEAN, ENC_STD, args.norm_count)
    print(f"✅ encoder_obs_normalizer: identity (环境已归一化)")

    # critic_obs_normalizer 保持默认 (PPO 自己学, 因为 critic 输入的 privileged obs 分布不同)
    print("ℹ️  critic_obs_normalizer: 保持默认初始化 (PPO 会自动学习)")

    # 5. 保存为 PPO checkpoint 格式
    ppo_ckpt = {
        "model_state_dict": actor_critic.state_dict(),
        "optimizer_state_dict": None,
        "iter": 0,
        "infos": {
            "pretrained_from": os.path.basename(args.input),
            "sl_val_loss": sl_ckpt.get("val_loss", None),
            "sl_n_samples": sl_ckpt.get("n_samples", None),
            "manual_norm": True,
            "norm_count": args.norm_count,
        },
    }
    torch.save(ppo_ckpt, args.output)
    print(f"\n✅ PPO checkpoint 已保存到: {args.output}")
    print(f"\n使用方法:")
    print(f"  cd /home/xxx/ubuntu_software/unilab/UniLab")
    print(f"  uv run train --algo ppo_sequence --task balance_infantry_joystick_flat --sim mujoco \\")
    print(f"    algo.resume_path={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()