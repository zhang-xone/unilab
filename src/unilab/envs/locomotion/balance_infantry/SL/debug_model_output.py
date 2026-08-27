"""调试: 对比 SL 模型和 PPO 导出模型的推理输出"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from combined_train import (
    OBS_MEAN, OBS_STD, ENC_MEAN, ENC_STD, VEL_MEAN, VEL_STD,
    OBS_HISTORY_LENGTH, DEFAULT_DOF_POS, JOINT_LIMITS, ACTION_SCALE_JOINT, ACTION_SCALE_WHEEL,
    build_observation, fk_fivebar,
)
from train_sl import LQRActor

# 加载 SL 模型
SL_CKPT = os.path.join(os.path.dirname(__file__), "log/combined_pretrain_0826_1534_79378samples.pth")
PPO_CKPT = os.path.join(os.path.dirname(__file__), "log/ppo_init_v4.pt")

print("=" * 60)
print("1. 加载 SL 模型")
print("=" * 60)
sl_ckpt = torch.load(SL_CKPT, map_location='cpu', weights_only=False)
sl_net = LQRActor()
sl_net.encoder.load_state_dict(sl_ckpt['encoder_state_dict'])
sl_net.actor.load_state_dict(sl_ckpt['actor_state_dict'])
sl_net.eval()
print(f"SL encoder keys: {list(sl_net.encoder.state_dict().keys())[:3]}...")
print(f"SL actor keys: {list(sl_net.actor.state_dict().keys())[:3]}...")

# 构造模拟初始状态
# 模拟机器人初始状态: 站立姿态, 关节在默认位置附近
q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
qvel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
last_actions = np.zeros(6, dtype=np.float32)
cmd_vel = 0.0
cmd_yaw = 0.0
cmd_height = 0.31

# 模拟 data 结构
class MockData:
    qpos = np.array([0, 0, 0.35, 1, 0, 0, 0], dtype=np.float64)  # x,y,z,qw,qx,qy,qz
    qvel = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)  # vx,vy,vz,wx,wy,wz

mock_data = MockData()

# 计算 theta 和 L0
phi1_l = q[2] + 5 * np.pi / 6
phi1_r = q[3] + 5 * np.pi / 6
phi4_l = q[0] + np.pi / 6
phi4_r = q[1] + np.pi / 6
L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)

print(f"\n初始状态: q={q}, qvel={qvel}")
print(f"L0=({L0_l:.4f}, {L0_r:.4f}), phi0=({phi0_l:.4f}, {phi0_r:.4f})")

# SL 推理
print("\n" + "=" * 60)
print("2. SL 推理")
print("=" * 60)
obs = build_observation(None, mock_data, q[:4], qvel, last_actions, cmd_vel, cmd_yaw, cmd_height)
print(f"obs (raw): {obs}")
print(f"obs shape: {obs.shape}")

# 归一化
o_n = (obs - OBS_MEAN) / (OBS_STD + 1e-6)
print(f"obs (norm): {o_n}")

# 构建 obs_history (5帧)
obs_hist = np.tile(obs, OBS_HISTORY_LENGTH)
oh_n = (obs_hist - ENC_MEAN) / (ENC_STD + 1e-6)
print(f"obs_hist shape: {oh_n.shape}")

# SL 推理
with torch.no_grad():
    oh_t = torch.tensor(oh_n).unsqueeze(0)
    o_t = torch.tensor(o_n).unsqueeze(0)
    est_vel = sl_net.encoder(oh_t)
    sl_actor_input = torch.cat([o_t, est_vel], dim=-1)
    sl_action = sl_net.actor(sl_actor_input).cpu().numpy()[0]
print(f"SL encoder output (est_vel): {est_vel.cpu().numpy()[0]}")
print(f"SL actor input: {sl_actor_input.cpu().numpy()[0]}")
print(f"SL raw action: {sl_action}")

# 计算 PD targets
def joint_scale(idx):
    lo, hi = JOINT_LIMITS[idx]
    return max(hi - DEFAULT_DOF_POS[idx], DEFAULT_DOF_POS[idx] - lo) * ACTION_SCALE_JOINT

sl_targets = np.array([
    sl_action[0] * joint_scale(0) + DEFAULT_DOF_POS[0],
    sl_action[1] * joint_scale(1) + DEFAULT_DOF_POS[1],
    sl_action[2] * joint_scale(2) + DEFAULT_DOF_POS[2],
    sl_action[3] * joint_scale(3) + DEFAULT_DOF_POS[3],
    sl_action[4] * ACTION_SCALE_WHEEL,
    sl_action[5] * ACTION_SCALE_WHEEL,
])
print(f"SL PD targets (deg): lf={np.rad2deg(sl_targets[0]):.1f} rf={np.rad2deg(sl_targets[1]):.1f} lr={np.rad2deg(sl_targets[2]):.1f} rr={np.rad2deg(sl_targets[3]):.1f} lw={sl_targets[4]:.2f} rw={sl_targets[5]:.2f}")

# PPO 推理
print("\n" + "=" * 60)
print("3. PPO 模型推理")
print("=" * 60)
from unilab.algos.torch.ppo.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence

ppo_ckpt = torch.load(PPO_CKPT, map_location='cpu', weights_only=False)
ppo_model = ActorCriticSequence(
    num_obs=33,
    num_critic_obs=150,
    num_actions=6,
    num_encoder_obs=165,
    latent_dim=3,
    encoder_hidden_dims=[128, 64],
    actor_hidden_dims=[128, 128],
    critic_hidden_dims=[128, 128],
    activation='elu',
    orthogonal_init=True,
    init_noise_std=0.5,
    obs_normalization=True,
    use_real_vel=False,
)
ppo_model.load_state_dict(ppo_ckpt['model_state_dict'])
ppo_model.eval()
print(f"PPO encoder keys: {list(ppo_model.encoder.state_dict().keys())[:3]}...")
print(f"PPO actor keys: {list(ppo_model.actor.state_dict().keys())[:3]}...")

# 检查 encoder 权重是否一致
sl_enc_w = sl_net.encoder.state_dict()
ppo_enc_w = ppo_model.encoder.state_dict()
for k in ['0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias']:
    diff = (sl_enc_w[k] - ppo_enc_w[k]).abs().max().item()
    print(f"  encoder.{k} diff: {diff:.2e}")

sl_act_w = sl_net.actor.state_dict()
ppo_act_w = ppo_model.actor.state_dict()
for k in ['0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias']:
    diff = (sl_act_w[k] - ppo_act_w[k]).abs().max().item()
    print(f"  actor.{k} diff: {diff:.2e}")

# PPO 推理: encoder 输入是归一化后的 obs_history
# 环境已做 obs_manual_norm, 所以 obs 已经是归一化值
# obs_history 是5帧归一化 obs 拼接
ppo_obs_history = torch.tensor(o_n).repeat(1, 5)  # (1, 165)
ppo_obs = torch.tensor(o_n).unsqueeze(0)  # (1, 33)

with torch.no_grad():
    ppo_latent = ppo_model.encode(ppo_obs_history)
    ppo_actor_input = ppo_model.actor_obs_normalizer(torch.cat([ppo_obs, ppo_latent], dim=-1))
    ppo_action = ppo_model.actor(ppo_actor_input).cpu().numpy()[0]

print(f"\nPPO encoder output (latent): {ppo_latent.cpu().numpy()[0]}")
print(f"PPO actor input (after norm): {ppo_actor_input.cpu().numpy()[0]}")
print(f"PPO raw action: {ppo_action}")

ppo_targets = np.array([
    ppo_action[0] * joint_scale(0) + DEFAULT_DOF_POS[0],
    ppo_action[1] * joint_scale(1) + DEFAULT_DOF_POS[1],
    ppo_action[2] * joint_scale(2) + DEFAULT_DOF_POS[2],
    ppo_action[3] * joint_scale(3) + DEFAULT_DOF_POS[3],
    ppo_action[4] * ACTION_SCALE_WHEEL,
    ppo_action[5] * ACTION_SCALE_WHEEL,
])
print(f"PPO PD targets (deg): lf={np.rad2deg(ppo_targets[0]):.1f} rf={np.rad2deg(ppo_targets[1]):.1f} lr={np.rad2deg(ppo_targets[2]):.1f} rr={np.rad2deg(ppo_targets[3]):.1f} lw={ppo_targets[4]:.2f} rw={ppo_targets[5]:.2f}")

# 对比
print("\n" + "=" * 60)
print("4. 对比")
print("=" * 60)
action_diff = np.abs(sl_action - ppo_action).max()
print(f"action max diff: {action_diff:.6f}")
latent_diff = np.abs(est_vel.cpu().numpy()[0] - ppo_latent.cpu().numpy()[0]).max()
print(f"latent max diff: {latent_diff:.6f}")

# 检查 normalizer
print(f"\nencoder_obs_normalizer: mean={ppo_model.encoder_obs_normalizer._mean.cpu().numpy()[0, :3]}... std={ppo_model.encoder_obs_normalizer._std.cpu().numpy()[0, :3]}...")
print(f"actor_obs_normalizer: mean={ppo_model.actor_obs_normalizer._mean.cpu().numpy()[0, :3]}... std={ppo_model.actor_obs_normalizer._std.cpu().numpy()[0, :3]}...")
print(f"actor_obs_normalizer std[33:36]: {ppo_model.actor_obs_normalizer._std.cpu().numpy()[0, 33:36]}")