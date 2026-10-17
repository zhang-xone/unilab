#!/usr/bin/env python3
"""监督学习预训练：用 LQR 收集的数据训练 encoder + actor 网络。

网络结构与 mujoco.yaml 配置完全一致:
  Encoder: 165 → 128 → 64 → 3 (latent)
  Actor:   33+3=36 → 128 → 128 → 6 (raw action)

用法:
    python SL/train_sl.py --data ../lqr_collected.npz --epochs 200 --lr 1e-3
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
import os
import sys

# ═══════════════════════════════════════════════════════════════
# 超参数 (与 mujoco.yaml 完全一致)
# ═══════════════════════════════════════════════════════════════
OBS_HISTORY_LENGTH = 5
NUM_OBS = 32
NUM_ENCODER_OBS = OBS_HISTORY_LENGTH * NUM_OBS   # 160
LATENT_DIM = 3
NUM_ACTIONS = 6
ENCODER_HIDDEN_DIMS = [128, 64]
ACTOR_HIDDEN_DIMS = [128, 64, 32]
ACTIVATION = "elu"


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    else:
        return nn.ELU()


class LQRActor(nn.Module):
    """复制 ActorCriticSequence 的 encoder + actor，用于监督学习预训练。

    与 rsl_rl/modules/actor_critic_sequence.py 中:
      - self.encoder (Linear → ELU → Linear → ELU → Linear)
      - self.actor   (Linear → ELU → Linear → ELU → Linear)
    结构和初始化方式完全一致。
    
    Input dimensions:
      - encoder: 160 (32 obs × 5 history)
      - actor:   32 + 3 latent = 35
    """

    def __init__(self):
        super().__init__()
        activation = get_activation(ACTIVATION)

        # ── Encoder: 165 → 128 → 64 → 3 ──
        encoder_layers = []
        encoder_layers.append(nn.Linear(NUM_ENCODER_OBS, ENCODER_HIDDEN_DIMS[0]))
        torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
        encoder_layers.append(activation)
        for l in range(len(ENCODER_HIDDEN_DIMS)):
            if l == len(ENCODER_HIDDEN_DIMS) - 1:
                encoder_layers.append(nn.Linear(ENCODER_HIDDEN_DIMS[l], LATENT_DIM))
                torch.nn.init.orthogonal_(encoder_layers[-1].weight, 0.01)
                torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
            else:
                encoder_layers.append(
                    nn.Linear(ENCODER_HIDDEN_DIMS[l], ENCODER_HIDDEN_DIMS[l + 1])
                )
                torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
                torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        # ── Actor: 33+3=36 → 128 → 128 → 6 ──
        actor_layers = []
        actor_layers.append(nn.Linear(NUM_OBS + LATENT_DIM, ACTOR_HIDDEN_DIMS[0]))
        torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
        actor_layers.append(activation)
        for l in range(len(ACTOR_HIDDEN_DIMS)):
            if l == len(ACTOR_HIDDEN_DIMS) - 1:
                actor_layers.append(nn.Linear(ACTOR_HIDDEN_DIMS[l], NUM_ACTIONS))
                torch.nn.init.orthogonal_(actor_layers[-1].weight, 0.01)
                torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
            else:
                actor_layers.append(
                    nn.Linear(ACTOR_HIDDEN_DIMS[l], ACTOR_HIDDEN_DIMS[l + 1])
                )
                torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
                torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

    def forward(self, obs_history, obs):
        """前向传播。

        Args:
            obs_history: (batch, 160) 过去5帧观测拼接
            obs:         (batch, 32)  当前帧观测

        Returns:
            action: (batch, 6) 预测的 raw action
        """
        latent = self.encoder(obs_history)
        actor_input = torch.cat([obs, latent], dim=-1)
        return self.actor(actor_input)

    def encode(self, obs_history):
        return self.encoder(obs_history)


def build_sequences(observations, pd_targets, history_length=5):
    """从时序数据构建 (obs_history, obs, target) 三元组。

    对每一步 t (t >= history_length - 1):
      - obs_history[t] = concatenate(obs[t-history_length+1], ..., obs[t])
      - obs[t]          = observations[t]
      - target[t]       = pd_targets[t]

    Args:
        observations: (T, 32) 时序观测
        pd_targets:   (T, 6)  目标动作
        history_length: 观测历史长度

    Returns:
        obs_history: (T - history_length + 1, 160)
        obs:         (T - history_length + 1, 32)
        targets:     (T - history_length + 1, 6)
    """
    T = observations.shape[0]
    if T < history_length:
        raise ValueError(
            f"数据长度 {T} < 历史长度 {history_length}，无法构建序列"
        )

    obs_history_list = []
    obs_list = []
    target_list = []

    for t in range(history_length - 1, T):
        hist = observations[t - history_length + 1 : t + 1].reshape(-1)
        obs_history_list.append(hist)
        obs_list.append(observations[t])
        target_list.append(pd_targets[t])

    return (
        np.stack(obs_history_list).astype(np.float32),
        np.stack(obs_list).astype(np.float32),
        np.stack(target_list).astype(np.float32),
    )


def compute_normalization(obs_history):
    """计算观测历史的 mean/std 用于归一化。

    Returns:
        mean: (165,) 或 0
        std:  (165,) 或 1
    """
    mean = obs_history.mean(axis=0)
    std = obs_history.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="LQR 监督学习预训练")
    parser.add_argument(
        "--data", type=str, default="../lqr/lqr_collected.npz", help="LQR 收集的数据文件"
    )
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch_size", type=int, default=512, help="批次大小")
    parser.add_argument("--val_split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--output", type=str, default="lqr_pretrain.pth", help="输出权重文件")
    parser.add_argument("--no_norm", action="store_true", help="不对观测做归一化")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    args = parser.parse_args()

    # ── 设备 ──
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    # ── 加载数据 ──
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)
    print(f"加载数据: {data_path}")
    data = np.load(data_path)
    observations = data["observations"].astype(np.float32)
    pd_targets = data["pd_targets"].astype(np.float32)
    print(f"  观测: {observations.shape}, 动作: {pd_targets.shape}")

    # ── 构建序列 ──
    obs_history, obs, targets = build_sequences(
        observations, pd_targets, OBS_HISTORY_LENGTH
    )
    print(f"  构建序列后: obs_history={obs_history.shape}, obs={obs.shape}, targets={targets.shape}")

    # ── 归一化 ──
    # 使用与 joystick.py 一致的手动归一化系数
    # 观测组成: gyro(3) pitch(1) roll(1) cmd(3) last_act(6)
    #           sin/cos_yaw(2) dof_pos_err(4) dof_vel(6) vmc_θ(2) vmc_L0(2) = 30
    obs_mean = np.array([
        0.0, 0.0, 0.0,               # 0-2:  gyro
        0.0,                          # 3:    pitch
        0.0,                          # 4:    roll
        0.0, 0.0, 0.29,              # 5-7:  cmd(vx,vyaw,height)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 8-13: last_actions
        0.0, 0.0,                     # 14-15: sin/cos yaw_error
        0.0, 0.0, 0.0, 0.0,          # 16-19: dof_pos_err
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 20-25: dof_vel
        0.3, 0.3,                     # 26-27: vmc_theta
        0.28, 0.28,                   # 28-29: vmc_L0
    ], dtype=np.float32)
    obs_std = np.array([
        1.5, 1.5, 1.5,               # 0-2:  gyro
        0.15,                         # 3:    pitch
        0.10,                         # 4:    roll
        1.5, 2.0, 0.08,              # 5-7:  cmd(vx,vyaw,height)
        0.6, 0.6, 0.6, 0.6, 0.6, 0.6,  # 8-13: last_actions
        0.6, 0.6,                     # 14-15: sin/cos yaw_error
        0.3, 0.3, 0.3, 0.3,          # 16-19: dof_pos_err
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0,  # 20-25: dof_vel
        0.4, 0.4,                     # 26-27: vmc_theta
        0.05, 0.05,                   # 28-29: vmc_L0
    ], dtype=np.float32)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)
    # 编码器归一化: 5 帧 obs 拼接
    enc_mean = np.tile(obs_mean, OBS_HISTORY_LENGTH).astype(np.float32)
    enc_std = np.tile(obs_std, OBS_HISTORY_LENGTH).astype(np.float32)

    obs_history = (obs_history - enc_mean) / enc_std
    obs = (obs - obs_mean) / obs_std
    print("  观测归一化: 手动系数 (joystick.py 一致)")

    # ── 划分训练/验证集 ──
    N = obs_history.shape[0]
    val_size = int(N * args.val_split)
    indices = np.random.permutation(N)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_dataset = TensorDataset(
        torch.tensor(obs_history[train_idx]),
        torch.tensor(obs[train_idx]),
        torch.tensor(targets[train_idx]),
    )
    val_dataset = TensorDataset(
        torch.tensor(obs_history[val_idx]),
        torch.tensor(obs[val_idx]),
        torch.tensor(targets[val_idx]),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"  训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # ── 模型 ──
    model = LQRActor().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    # ── 训练 ──
    best_val_loss = float("inf")
    obs_noise_std = 0.05  # 归一化观测上的高斯噪声标准差
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_hist, batch_obs, batch_target in train_loader:
            batch_hist = batch_hist.to(device)
            batch_obs = batch_obs.to(device)
            batch_target = batch_target.to(device)

            # 观测噪声: 数据增强, 强迫 NN 学习平滑映射
            batch_hist = batch_hist + torch.randn_like(batch_hist) * obs_noise_std
            batch_obs = batch_obs + torch.randn_like(batch_obs) * obs_noise_std

            pred = model(batch_hist, batch_obs)
            loss = loss_fn(pred, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_hist.size(0)

        train_loss /= len(train_dataset)

        # ── 验证 ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_hist, batch_obs, batch_target in val_loader:
                batch_hist = batch_hist.to(device)
                batch_obs = batch_obs.to(device)
                batch_target = batch_target.to(device)
                pred = model(batch_hist, batch_obs)
                val_loss += loss_fn(pred, batch_target).item() * batch_hist.size(0)
        val_loss /= len(val_dataset)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:4d}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    # ── 保存 ──
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    save_dict = {
        "encoder_state_dict": {
            k.replace("encoder.", ""): v
            for k, v in best_state.items()
            if k.startswith("encoder.")
        },
        "actor_state_dict": {
            k.replace("actor.", ""): v
            for k, v in best_state.items()
            if k.startswith("actor.")
        },
        "enc_mean": enc_mean,
        "enc_std": enc_std,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "val_loss": best_val_loss,
        "config": {
            "obs_history_length": OBS_HISTORY_LENGTH,
            "num_obs": NUM_OBS,
            "num_encoder_obs": NUM_ENCODER_OBS,
            "latent_dim": LATENT_DIM,
            "num_actions": NUM_ACTIONS,
            "encoder_hidden_dims": ENCODER_HIDDEN_DIMS,
            "actor_hidden_dims": ACTOR_HIDDEN_DIMS,
            "activation": ACTIVATION,
        },
    }
    torch.save(save_dict, output_path)

    print(f"\n✅ 训练完成!")
    print(f"   最佳验证损失: {best_val_loss:.6f}")
    print(f"   权重保存到: {output_path}")
    print(f"\n加载到 PPO policy 示例:")
    print(f"   checkpoint = torch.load('{args.output}')")
    print(f"   policy.encoder.load_state_dict(checkpoint['encoder_state_dict'])")
    print(f"   policy.actor.load_state_dict(checkpoint['actor_state_dict'])")


if __name__ == "__main__":
    main()