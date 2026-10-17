#!/usr/bin/env python3
"""LQR 仿真 + NN 在线训练 一体化脚本。

- LQR 跑仿真，你键盘操作（W/S 调腿长、A/D 调腿角等）
- 后台自动收集 (观测, LQR动作) 训练数据
- 每 500 步自动训练一轮
- 按 N 切换 NN 控制，验证效果
- 按 M 保存模型
- Backspace 重置

用法:
    python SL/combined_train.py --scene flat
"""

import argparse
import os
import sys
import numpy as np
import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
from collections import deque

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.join(_SCRIPT_DIR, "..")
_LQR_DIR = os.path.join(_PARENT_DIR, "lqr")
sys.path.insert(0, _PARENT_DIR)
sys.path.insert(0, _LQR_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from train_sl import LQRActor, OBS_HISTORY_LENGTH, NUM_OBS, NUM_ACTIONS
from test_vmc_lqr import (
    fk_fivebar, compute_jacobian_partials, inverse_pd_targets,
    compute_lqr_gains, lqr_virtual_torque,
)

# ═══════════════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════════════
LEG_KP = (50.0, 50.0, 50.0, 50.0)
LEG_KD = (3.2, 3.2, 3.2, 3.2)
WHEEL_KD = (0.1, 0.1)
DEFAULT_DOF_POS = np.array([0.0, 0.0, -0.0, -0.0, 0.0, 0.0], dtype=np.float64)
INIT_Z = 0.50

# 控制频率匹配 PPO: sim_dt=0.005, ctrl_dt=0.02 → decimation=4
DECIMATION = 4

# 观测噪声参数 (与 PPO balance_infantry 实际加噪字段一致)
NOISE_LEVEL = 1.0
NOISE_SCALE = {
    "gyro": 0.2,
    "joint_vel": 0.4,
    "vmc_phi0": 0.008,
    "vmc_L0": 0.003,
}


def _add_obs_noise(val, scale, rng=None):
    if rng is None:
        rng = np.random
    noise = rng.uniform(-1.0, 1.0, np.shape(val)).astype(np.float32) * NOISE_LEVEL * scale
    return val + noise


# RL 动作空间参数 (与 test_vmc_lqr.py inverse_pd_targets 一致)
ACTION_SCALE_JOINT = 0.5
ACTION_SCALE_WHEEL = 50.0
JOINT_LIMITS = np.array([
    [-1.57, 2.40],
    [-1.57, 2.40],
    [-1.57, 1.57],
    [-1.57, 1.57],
    [-np.inf, np.inf],
    [-np.inf, np.inf],
], dtype=np.float64)

# 手动归一化系数 (joystick.py 一致)
OBS_MEAN = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.29,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.28, 0.28,
], dtype=np.float32)
OBS_STD = np.array([
    0.34, 0.62, 2.0, 0.15, 0.10,
    1.5, 2.0, 0.08,
    0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
    0.6, 0.6,
    0.3, 0.3, 0.3, 0.3,
    1.5, 1.5, 1.5, 1.5, 22.0, 22.0,
    0.4, 0.4, 0.05, 0.05,
], dtype=np.float32)
ENC_MEAN = np.tile(OBS_MEAN, OBS_HISTORY_LENGTH).astype(np.float32)
ENC_STD = np.tile(OBS_STD, OBS_HISTORY_LENGTH).astype(np.float32)


def build_observation(model, data, dof_pos, dof_vel, last_actions, cmd_vel, cmd_yaw, cmd_height, yaw_error, noisy=False):
    gyro = data.qvel[3:6].copy()
    qw, qx, qy, qz = data.qpos[3:7]
    gx = 2 * (qx * qz - qw * qy)
    gy = 2 * (qy * qz + qw * qx)
    gz = 1 - 2 * (qx * qx + qy * qy)
    pitch = np.arcsin(np.clip(gx, -1.0, 1.0))
    roll = np.arcsin(np.clip(-gy, -1.0, 1.0))
    commands = np.array([cmd_vel, cmd_yaw, cmd_height])

    q_lf, q_rf, q_lr, q_rr = dof_pos[0], dof_pos[1], dof_pos[2], dof_pos[3]
    phi1_l = q_lr + 5 * np.pi / 6
    phi1_r = q_rr + 5 * np.pi / 6
    phi4_l = q_lf + np.pi / 6
    phi4_r = q_rf + np.pi / 6
    L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
    L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)
    theta_l = np.pi / 2 - phi0_l - pitch
    theta_r = np.pi / 2 - phi0_r - pitch
    dof_pos_err = dof_pos[:4] - DEFAULT_DOF_POS[:4]

    if noisy:
        gyro = _add_obs_noise(gyro, NOISE_SCALE["gyro"])
        dof_vel = _add_obs_noise(dof_vel, NOISE_SCALE["joint_vel"])
        theta_l = _add_obs_noise(theta_l, NOISE_SCALE["vmc_phi0"])
        theta_r = _add_obs_noise(theta_r, NOISE_SCALE["vmc_phi0"])
        L0_l = _add_obs_noise(L0_l, NOISE_SCALE["vmc_L0"])
        L0_r = _add_obs_noise(L0_r, NOISE_SCALE["vmc_L0"])

    obs = np.concatenate([
        gyro,
        np.array([pitch]), np.array([roll]),
        commands,
        last_actions,
        np.array([np.sin(yaw_error)]), np.array([np.cos(yaw_error)]),
        dof_pos_err, dof_vel,
        np.array([theta_l]), np.array([theta_r]),
        np.array([L0_l]), np.array([L0_r]),
    ])
    return obs.astype(np.float32)


def lqr_control(model, data, q, qvel, L0_l, L0_r, phi0_l, phi0_r, phi2_l, phi3_l,
                sigma1_l, phi2_r, phi3_r, sigma1_r, pitch, pitch_dot, base_vx, dt,
                lqr_state, prev_lqr, cmd_vel, step_count, L0_target, ff,
                roll, roll_dot, yaw, yaw_dot, cmd_yaw_rate, kp_leg, kd_leg,
                kp_roll, kd_roll, kp_yaw, kd_yaw, target_yaw):
    phi1_l = q[2] + 5 * np.pi / 6
    phi1_r = q[3] + 5 * np.pi / 6
    phi4_l = q[0] + np.pi / 6
    phi4_r = q[1] + np.pi / 6

    J00_l, J01_l, J10_l, J11_l = compute_jacobian_partials(
        phi1_l, phi4_l, phi0_l, phi2_l, phi3_l, sigma1_l, L0_l)
    J00_r, J01_r, J10_r, J11_r = compute_jacobian_partials(
        phi1_r, phi4_r, phi0_r, phi2_r, phi3_r, sigma1_r, L0_r)

    theta_l = np.pi / 2 - phi0_l - pitch
    theta_r = np.pi / 2 - phi0_r - pitch
    theta_dot_l = -(J10_l * qvel[2] + J11_l * qvel[0])
    theta_dot_r = -(J10_r * qvel[3] + J11_r * qvel[1])

    lqr_state['L'][0] = theta_l
    lqr_state['L'][1] = theta_dot_l
    lqr_state['L'][2] -= base_vx * dt
    lqr_state['L'][2] = np.clip(lqr_state['L'][2], -1.0, 1.0)
    lqr_state['L'][3] = -base_vx
    lqr_state['L'][4] = pitch
    lqr_state['L'][5] = pitch_dot
    lqr_state['R'][0] = theta_r
    lqr_state['R'][1] = theta_dot_r
    lqr_state['R'][2] -= base_vx * dt
    lqr_state['R'][2] = np.clip(lqr_state['R'][2], -1.0, 1.0)
    lqr_state['R'][3] = -base_vx
    lqr_state['R'][4] = pitch
    lqr_state['R'][5] = pitch_dot

    K_l = compute_lqr_gains(L0_l)
    K_r = compute_lqr_gains(L0_r)
    lqr_target = np.zeros(6, dtype=np.float64)
    lqr_target[3] = -cmd_vel
    target_x = lqr_state['L'][2] if abs(base_vx) >= 0.5 else 0.0
    lqr_target[2] = target_x

    wheel_t_l, T_lqr_l = lqr_virtual_torque(K_l, lqr_state['L'], lqr_target)
    wheel_t_r, T_lqr_r = lqr_virtual_torque(K_r, lqr_state['R'], lqr_target)
    ramp = min(1.0, step_count / 300.0)
    T_l = np.clip(T_lqr_l * ramp, -20.0, 20.0)
    T_r = np.clip(T_lqr_r * ramp, -20.0, 20.0)
    wheel_t_l = np.clip(wheel_t_l * ramp, -10.0, 10.0)
    wheel_t_r = np.clip(wheel_t_r * ramp, -10.0, 10.0)

    L0_dot_l = (L0_l - prev_lqr['L0_l']) / dt if prev_lqr['L0_l'] > 0 else 0.0
    L0_dot_r = (L0_r - prev_lqr['L0_r']) / dt if prev_lqr['L0_r'] > 0 else 0.0
    F_l = kp_leg * (L0_target - L0_l) - kd_leg * L0_dot_l + ff
    F_r = kp_leg * (L0_target - L0_r) - kd_leg * L0_dot_r + ff
    prev_lqr['L0_l'] = L0_l
    prev_lqr['L0_r'] = L0_r

    # ── Roll 轴 PD: 通过左右腿长差纠正 roll ──
    roll_correction = kp_roll * roll + kd_roll * roll_dot
    F_l -= roll_correction
    F_r += roll_correction

    tau4_l = np.clip(J01_l * F_l + J11_l * T_l, -40.0, 40.0)
    tau1_l = np.clip(J00_l * F_l + J10_l * T_l, -40.0, 40.0)
    tau4_r = np.clip(J01_r * F_r + J11_r * T_r, -40.0, 40.0)
    tau1_r = np.clip(J00_r * F_r + J10_r * T_r, -40.0, 40.0)

    # ── 转向 PD: 目标 yaw 角 PD 控制 ──
    cmd_yaw_rate = np.clip(cmd_yaw_rate, -3.0, 3.0)
    target_yaw[0] += cmd_yaw_rate * dt
    target_yaw[0] = (target_yaw[0] + np.pi) % (2.0 * np.pi) - np.pi
    yaw_err = (target_yaw[0] - yaw + np.pi) % (2.0 * np.pi) - np.pi
    yaw_err = np.clip(yaw_err, -1.0, 1.0)
    if abs(yaw_err) < 0.02 and abs(cmd_yaw_rate) < 0.01:
        yaw_torque = 0.0
    else:
        yaw_torque = kp_yaw * yaw_err - kd_yaw * yaw_dot
    wheel_t_l -= yaw_torque
    wheel_t_r += yaw_torque

    ramp_l = np.clip((np.pi / 3 - abs(theta_l)) / (np.pi / 6), 0.0, 1.0)
    ramp_r = np.clip((np.pi / 3 - abs(theta_r)) / (np.pi / 6), 0.0, 1.0)
    ff_theta_l = ff / np.cos(np.clip(theta_l, -1.5, 1.5)) * ramp_l
    ff_theta_r = ff / np.cos(np.clip(theta_r, -1.5, 1.5)) * ramp_r
    vmc_tau4_l = J01_l * ff_theta_l
    vmc_tau1_l = J00_l * ff_theta_l
    vmc_tau4_r = J01_r * ff_theta_r
    vmc_tau1_r = J00_r * ff_theta_r
    return (tau4_l, tau4_r, tau1_l, tau1_r, wheel_t_l, wheel_t_r,
            vmc_tau4_l, vmc_tau4_r, vmc_tau1_l, vmc_tau1_r)


# 速度归一化系数 (vx, vy, vz) -- 与 joystick.py obs_vel 的 local_linvel 一致
VEL_MEAN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
VEL_STD = np.array([1.5, 0.5, 0.5], dtype=np.float32)

# 关节缩放 (用于 raw_action → PD target)
def _joint_scale_fn(joint_idx):
    lo, hi = JOINT_LIMITS[joint_idx]
    return max(hi - DEFAULT_DOF_POS[joint_idx], DEFAULT_DOF_POS[joint_idx] - lo) * ACTION_SCALE_JOINT

LEG_SCALES = np.array([_joint_scale_fn(i) for i in range(4)], dtype=np.float32)
LEG_DEFAULTS = DEFAULT_DOF_POS[:4].astype(np.float32)


def train_step_actor(net, optimizer, loss_fn, obs, velocities, step_q, step_qvel, step_lqr, step_vmc, batch_size, device):
    """训练 actor: obs(32) + 真实速度(3) → 一个 PD 目标, 对 DECIMATION 步分别计算力矩,
    loss = mean_i MSE(PD力矩_i, LQR纯力矩_i)。
    step_q/qvel/lqr/vmc: (N, DECIMATION, 6)
    """
    net.train()
    N = len(obs)
    perm = np.random.permutation(N)
    total_loss = 0.0
    obs_noise = 0.05
    vel_noise = 0.2
    vel_mean_t = torch.tensor(VEL_MEAN).to(device)
    vel_std_t = torch.tensor(VEL_STD).to(device)
    kp_t = torch.tensor(LEG_KP, dtype=torch.float32).to(device)
    kd_t = torch.tensor(LEG_KD, dtype=torch.float32).to(device)
    wk_t = torch.tensor(WHEEL_KD, dtype=torch.float32).to(device)
    leg_scales_t = torch.tensor(LEG_SCALES).to(device)
    leg_defaults_t = torch.tensor(LEG_DEFAULTS).to(device)
    wheel_scale = ACTION_SCALE_WHEEL
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        bo = torch.tensor(obs[idx]).to(device)
        bv = torch.tensor(velocities[idx]).to(device)
        bq = torch.tensor(step_q[idx]).to(device)       # (B, D, 6)
        bqv = torch.tensor(step_qvel[idx]).to(device)
        blqr = torch.tensor(step_lqr[idx]).to(device)
        bvmc = torch.tensor(step_vmc[idx]).to(device)
        bo = bo + torch.randn_like(bo) * obs_noise
        bv = bv + torch.randn_like(bv) * vel_noise
        bv_n = (bv - vel_mean_t) / vel_std_t
        actor_input = torch.cat([bo, bv_n], dim=-1)
        raw_action = net.actor(actor_input)  # (B, 6), [-1, 1]
        # 腿 PD 目标: t = action * scale + default
        leg_targets = raw_action[:, :4] * leg_scales_t + leg_defaults_t  # (B, 4)
        # 轮 PD 目标: t = action * scale
        wheel_targets = raw_action[:, 4:] * wheel_scale  # (B, 2)
        # 对 DECIMATION 步分别计算力矩
        leg_tau = kp_t * (leg_targets.unsqueeze(1) - bq[:, :, :4]) - kd_t * bqv[:, :, :4]  # (B, D, 4)
        wheel_tau = wk_t * (wheel_targets.unsqueeze(1) - bqv[:, :, 4:])  # (B, D, 2)
        pred_tau = torch.cat([leg_tau, wheel_tau], dim=-1)  # (B, D, 6)
        btarget = blqr - bvmc  # (B, D, 6), LQR 纯 PD 力矩
        loss = loss_fn(pred_tau, btarget)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    return total_loss / N


def compute_optimal_pd_target(step_buffer, default_dof_pos):
    """从 DECIMATION 步的 (q, qvel, lqr_tau, vmc_tau) 计算最优 PD 目标。
    min_t sum_i [Kp*(t - q_i) - Kd*qvel_i - tau_pure_i]²
    腿: t = mean(q) + mean(Kd*qvel + tau_pure) / Kp
    轮: t = mean(qvel) + mean(tau_pure) / Kd
    Returns: raw_action(6) in [-1, 1]
    """
    q_arr = np.array(step_buffer['q'])
    qvel_arr = np.array(step_buffer['qvel'])
    lqr_arr = np.array(step_buffer['lqr_tau'])
    vmc_arr = np.array(step_buffer['vmc_tau'])
    tau_pure = lqr_arr - vmc_arr  # (N, 6), pure PD torque

    raw_action = np.zeros(6, dtype=np.float32)
    # 腿: t = mean(q) + mean(Kd*qvel + tau_pure) / Kp
    for j in range(4):
        lo, hi = JOINT_LIMITS[j]
        scale = max(hi - default_dof_pos[j], default_dof_pos[j] - lo) * ACTION_SCALE_JOINT
        t_opt = np.mean(q_arr[:, j]) + np.mean(LEG_KD[j] * qvel_arr[:, j] + tau_pure[:, j]) / LEG_KP[j]
        raw_action[j] = (t_opt - default_dof_pos[j]) / scale
    # 轮: t = mean(qvel) + mean(tau_pure) / Kd
    for j in range(2):
        t_opt = np.mean(qvel_arr[:, 4 + j]) + np.mean(tau_pure[:, 4 + j]) / WHEEL_KD[j]
        raw_action[4 + j] = t_opt / ACTION_SCALE_WHEEL
    return raw_action


def train_step_encoder(net, enc_optimizer, loss_fn, obs_hist, velocities, batch_size, device):
    """训练 encoder: 输入 obs_hist(165) → 预测速度(3), 用真实速度做监督。
    只更新 encoder 参数。
    """
    net.train()
    N = len(obs_hist)
    perm = np.random.permutation(N)
    total_loss = 0.0
    vel_mean_t = torch.tensor(VEL_MEAN).to(device)
    vel_std_t = torch.tensor(VEL_STD).to(device)
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        bh = torch.tensor(obs_hist[idx]).to(device)
        bv = torch.tensor(velocities[idx]).to(device)
        bv_n = (bv - vel_mean_t) / vel_std_t
        pred_v = net.encoder(bh)
        loss = loss_fn(pred_v, bv_n)
        enc_optimizer.zero_grad()
        loss.backward()
        enc_optimizer.step()
        total_loss += loss.item() * len(idx)
    return total_loss / N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="flat")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--train_every", type=int, default=240, help="每 N 个控制步(50Hz)训练一次, 默认240=4.8s")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="combined_pretrain.pth")
    parser.add_argument("--load", type=str, default=None, help="加载已有模型继续训练")
    parser.add_argument("--num_envs", type=int, default=1, help="并行环境数(>1时无头模式收集数据)")
    parser.add_argument("--headless", action="store_true", help="无头模式(无渲染)")
    parser.add_argument("--test", action="store_true", help="纯测试模式: 不训练, 只跑仿真")
    parser.add_argument("--dagger", action="store_true", help="DAgger模式: NN和LQR交替运行, 用LQR标注NN状态")
    parser.add_argument("--mix_steps", type=int, default=50000, help="混合输出步数: alpha从0→1, 0=纯LQR, 1=纯NN")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    # ── 场景 ──
    scene_dir = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "assets", "robots", "balance_infantry")
    scene_map = {"flat": os.path.join(scene_dir, "scene_flat.xml")}
    scene_path = scene_map.get(args.scene, scene_map["flat"])
    print(f"加载场景: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    joint_names = [
        "left_front_joint", "right_front_joint", "left_rear_joint", "right_rear_joint",
        "left_wheel_joint", "right_wheel_joint",
    ]
    joint_qpos = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                  for n in joint_names]
    joint_dofadr = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                    for n in joint_names]
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_actuator") for n in joint_names]

    # ── 初始姿态 ──
    try:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    except Exception:
        pass
    data.qpos[2] = INIT_Z
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    # 保存初始姿态，reset 时手动恢复
    init_qpos = data.qpos.copy()
    init_qvel = data.qvel.copy()
    default_dof_pos = np.array([data.qpos[joint_qpos[0]], data.qpos[joint_qpos[1]],
                                 data.qpos[joint_qpos[2]], data.qpos[joint_qpos[3]]], dtype=np.float32)

    # ── 域随机化 (与 PPO domain_rand 一致) ──
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    original_base_mass = model.body_mass[base_body_id]
    original_base_ipos = model.body_ipos[base_body_id].copy()
    DR_MASS_RANGE = [-2.0, 3.0]
    DR_COM_X = [-0.05, 0.05]
    DR_COM_Y = [-0.05, 0.05]
    DR_COM_Z = [-0.05, 0.05]

    # ── NN 模型 ──
    net = LQRActor().to(device)
    optimizer = torch.optim.Adam(net.actor.parameters(), lr=args.lr)
    enc_optimizer = torch.optim.Adam(net.encoder.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    if args.load:
        ckpt = torch.load(args.load, map_location=device, weights_only=False)
        net.encoder.load_state_dict(ckpt["encoder_state_dict"])
        net.actor.load_state_dict(ckpt["actor_state_dict"])
        print(f"加载模型: {args.load}")
    else:
        import glob as _glob
        log_dir = os.path.join(_SCRIPT_DIR, "log")
        pattern = os.path.join(log_dir, os.path.splitext(args.output)[0] + "_*" + os.path.splitext(args.output)[1])
        files = sorted(_glob.glob(pattern))
        if files:
            ckpt = torch.load(files[-1], map_location=device, weights_only=False)
            net.encoder.load_state_dict(ckpt["encoder_state_dict"])
            net.actor.load_state_dict(ckpt["actor_state_dict"])
            print(f"自动加载最新模型: {os.path.basename(files[-1])}")

    enc_mean_t = torch.tensor(ENC_MEAN).to(device)
    enc_std_t = torch.tensor(ENC_STD).to(device)
    obs_mean_t = torch.tensor(OBS_MEAN).to(device)
    obs_std_t = torch.tensor(OBS_STD).to(device)

    # ── 状态变量 ──
    lqr_state = {'L': np.zeros(6, dtype=np.float64), 'R': np.zeros(6, dtype=np.float64)}
    prev_lqr = {'L0_l': -1.0, 'L0_r': -1.0}
    step_count = [0]
    ctrl_step = [0]  # 控制步计数 (每 DECIMATION sim步+1)
    stored_ctrl = [None]  # 缓存的力矩, 用于非控制步
    stored_nn_pd_target = [None]  # NN PD目标 (50Hz更新, 500Hz施力用)
    need_forward = [False]
    need_reset = [False]
    paused = [False]
    use_nn = [False]
    last_actions = [np.zeros(6, dtype=np.float32)]
    obs_history = deque(maxlen=OBS_HISTORY_LENGTH)
    last_loss = [0.0]
    nn_debug_count = [0]  # NN 切换后前几步打印对比
    step_buffer = {'q': [], 'qvel': [], 'lqr_tau': [], 'vmc_tau': []}  # DECIMATION 步 buffer, 用于计算最优 PD 目标
    last_ctrl_obs = [None]  # 上一个控制步的 obs, 与当前 buffer 配对
    last_ctrl_vel = [None]  # 上一个控制步的速度, 与当前 buffer 配对
    last_ctrl_obs_hist = [None]  # 上一个控制步的 obs_hist, 与当前 buffer 配对

    # VMC 参数
    params = {
        'L0_target': 0.35, 'phi0_target': np.deg2rad(90), 'feedforward': 80.0,
        'kp_leg': 1000.0, 'kd_leg': 100.0,
        'kp_roll': 500.0, 'kd_roll': 20.0,
        'kp_yaw': 10.0, 'kd_yaw': 1.0,
        'cmd_vel': 0.0, 'cmd_yaw_rate': 0.0,
    }
    target_yaw = [0.0]

    # 训练数据
    train_obs_hist = []
    train_obs = []
    train_velocities = []
    train_step_q = []
    train_step_qvel = []
    train_step_lqr = []
    train_step_vmc = []
    train_sample_count = [0]  # 控制步计数, 用于触发训练和保存

    print(f"  设备={args.device}, 每{args.train_every}步训练, L0={params['L0_target']:.2f}m")
    if args.test:
        print(f"  *** 测试模式: 不收集数据, 不训练 ***")
    print(f"  kp_leg={params['kp_leg']:.0f} kd_leg={params['kd_leg']:.0f} "
          f"kp_roll={params['kp_roll']:.0f} kd_roll={params['kd_roll']:.0f} "
          f"kp_yaw={params['kp_yaw']:.0f} kd_yaw={params['kd_yaw']:.1f}")
    print(f"  按键: N=切换LQR/NN  W/S=L0±  A/D=φ₀±  F/G=kp_leg±  V/B=kp_roll±  C/X=kd_roll±  Y/U=kp_yaw±  I/O=kd_yaw±")
    print(f"        ↑↓←→=速度指令  H=停止  M=保存  Backspace=重置  Space=暂停  Esc=退出\n")

    def reset_sim():
        data.qpos[:] = init_qpos
        data.qvel[:] = init_qvel
        data.ctrl[:] = 0.0
        # 域随机化: 质量 + 质心偏移
        model.body_mass[base_body_id] = original_base_mass + np.random.uniform(*DR_MASS_RANGE)
        com_offset = np.array([np.random.uniform(*DR_COM_X),
                               np.random.uniform(*DR_COM_Y),
                               np.random.uniform(*DR_COM_Z)])
        model.body_ipos[base_body_id] = original_base_ipos + com_offset
        # 随机出生高度，练习下落抗扰
        data.qpos[2] = INIT_Z + np.random.uniform(0.0, 0.25)
        # 随机 yaw 角度扰动
        yaw_rand = np.random.uniform(-0.5, 0.5)
        data.qpos[3:7] = [np.cos(yaw_rand / 2), 0.0, 0.0, np.sin(yaw_rand / 2)]
        # 随机线速度、角速度扰动
        data.qvel[0] = np.random.uniform(-0.5, 0.5)
        data.qvel[1] = np.random.uniform(-0.5, 0.5)
        data.qvel[3] = np.random.uniform(-0.3, 0.3)
        data.qvel[4] = np.random.uniform(-0.3, 0.3)
        data.qvel[5] = np.random.uniform(-0.5, 0.5)
        need_forward[0] = True
        lqr_state['L'].fill(0.0)
        lqr_state['R'].fill(0.0)
        prev_lqr['L0_l'] = -1.0
        prev_lqr['L0_r'] = -1.0
        step_count[0] = 0
        ctrl_step[0] = 0
        stored_ctrl[0] = None
        stored_nn_pd_target[0] = None
        last_actions[0] = np.zeros(6, dtype=np.float32)
        obs_history.clear()
        step_buffer['q'].clear()
        step_buffer['qvel'].clear()
        step_buffer['lqr_tau'].clear()
        step_buffer['vmc_tau'].clear()
        last_ctrl_obs[0] = None
        last_ctrl_vel[0] = None
        last_ctrl_obs_hist[0] = None
        if args.test:
            pass  # test 模式保持当前模式, 不切回 LQR
        else:
            use_nn[0] = False
        target_yaw[0] = 0.0
        if args.headless and not args.test:
            params['cmd_vel'] = np.random.uniform(-1.5, 1.5)
            params['cmd_yaw_rate'] = np.random.uniform(-2.0, 2.0)
            params['L0_target'] = np.random.uniform(0.20, 0.40)
        else:
            params['cmd_vel'] = 0.0
            params['cmd_yaw_rate'] = 0.0
        print(f">>> 重置 {'(NN)' if use_nn[0] else '(LQR)'} <<<")

    def save_model():
        import datetime
        log_dir = os.path.join(_SCRIPT_DIR, "log")
        os.makedirs(log_dir, exist_ok=True)
        base = os.path.splitext(args.output)[0]
        ext = os.path.splitext(args.output)[1]
        ts = datetime.datetime.now().strftime("%m%d_%H%M")
        output_path = os.path.join(log_dir, f"{base}_{ts}_{train_sample_count[0]}samples{ext}")
        torch.save({
            'encoder_state_dict': net.encoder.state_dict(),
            'actor_state_dict': net.actor.state_dict(),
            'enc_mean': ENC_MEAN, 'enc_std': ENC_STD,
            'obs_mean': OBS_MEAN, 'obs_std': OBS_STD,
            'n_samples': train_sample_count[0],
        }, output_path)
        print(f">>> 模型已保存: {output_path} ({train_sample_count[0]} 样本) <<<")

    def controller(m, d):
        if paused[0]:
            return
        step_count[0] += 1

        # ── 每步都算: 关节状态、运动学、姿态 (LQR 500Hz 需要) ──
        q = np.array([d.qpos[joint_qpos[i]] for i in range(6)], dtype=np.float64)
        qvel = np.array([d.qvel[joint_dofadr[i]] for i in range(6)], dtype=np.float64)

        q_lf, q_rf, q_lr, q_rr = q[0], q[1], q[2], q[3]
        phi1_l = q_lr + 5 * np.pi / 6
        phi1_r = q_rr + 5 * np.pi / 6
        phi4_l = q_lf + np.pi / 6
        phi4_r = q_rf + np.pi / 6
        L0_l, phi0_l, phi2_l, phi3_l, sigma1_l = fk_fivebar(phi1_l, phi4_l)
        L0_r, phi0_r, phi2_r, phi3_r, sigma1_r = fk_fivebar(phi1_r, phi4_r)

        qw, qx, qy, qz = d.qpos[3:7]
        Rmat = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
        ])
        gravity_body = Rmat.T @ np.array([0.0, 0.0, -1.0])
        pitch = np.arcsin(np.clip(gravity_body[0], -1.0, 1.0))
        pitch_dot = d.qvel[4]
        roll = np.arcsin(np.clip(-gravity_body[1], -1.0, 1.0))
        roll_dot = d.qvel[3]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        yaw_dot = d.qvel[5]
        base_vx = d.qvel[0]

        # ── LQR 500Hz: 每步更新, 使用 sim_dt (保证 LQR 精度最高) ──
        lqr_tau = lqr_control(model, d, q, qvel, L0_l, L0_r, phi0_l, phi0_r,
                              phi2_l, phi3_l, sigma1_l, phi2_r, phi3_r, sigma1_r,
                              pitch, pitch_dot, base_vx, dt, lqr_state, prev_lqr,
                              params['cmd_vel'], step_count[0], params['L0_target'], params['feedforward'],
                              roll, roll_dot, yaw, yaw_dot, params['cmd_yaw_rate'],
                              params['kp_leg'], params['kd_leg'],
                              params['kp_roll'], params['kd_roll'],
                              params['kp_yaw'], params['kd_yaw'], target_yaw)
        tau4_l, tau4_r, tau1_l, tau1_r, wheel_t_l, wheel_t_r, vmc_tau4_l, vmc_tau4_r, vmc_tau1_l, vmc_tau1_r = lqr_tau

        # 每 sim 步追加到 buffer，用于控制步时计算最优 PD 目标
        step_buffer['q'].append(q[:6].copy())
        step_buffer['qvel'].append(qvel[:6].copy())
        step_buffer['lqr_tau'].append(np.array([tau4_l, tau4_r, tau1_l, tau1_r, wheel_t_l, wheel_t_r], dtype=np.float32))
        step_buffer['vmc_tau'].append(np.array([vmc_tau4_l, vmc_tau4_r, vmc_tau1_l, vmc_tau1_r, 0.0, 0.0], dtype=np.float32))

        # ── PD / NN 50Hz: 每 DECIMATION=4 步更新一次 ──
        is_ctrl_step = (step_count[0] % DECIMATION == 0)
        if is_ctrl_step:
            ctrl_step[0] += 1

            # 从 10 步 buffer 计算最优 PD 目标 (动作空间, [-1, 1])
            lqr_pd_target_opt = compute_optimal_pd_target(step_buffer, default_dof_pos)

            # 观测
            up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
            height_cmd = params['L0_target'] + 0.023  # L0_target + diff, 对齐 PPO 的 base_height_from_scan+0.141*up_z
            if ctrl_step[0] % 50 == 0:
                print(f"[DEBUG height] height_cmd={height_cmd:.3f}m  L0_actual=({L0_l:.3f},{L0_r:.3f})  "
                      f"L0_target={params['L0_target']:.3f}  diff_cmd_L0={height_cmd - L0_l:.3f}m")
            obs = build_observation(model, d, q[:4], qvel[:6], last_actions[0],
                                    params['cmd_vel'], params['cmd_yaw_rate'], height_cmd,
                                    (target_yaw[0] - yaw + np.pi) % (2.0 * np.pi) - np.pi,
                                    noisy=True)
            obs_history.append(obs)
            if len(obs_history) == OBS_HISTORY_LENGTH:
                obs_hist = np.concatenate(list(obs_history))
            else:
                obs_hist = np.tile(obs, OBS_HISTORY_LENGTH)

            # 收集训练数据: 用上一个控制步的 obs (buffer 窗口起始状态) 与当前 buffer 配对
            if not args.test and last_ctrl_obs[0] is not None:
                train_obs_hist.append(last_ctrl_obs_hist[0].astype(np.float32))
                train_obs.append(last_ctrl_obs[0].astype(np.float32))
                train_velocities.append(last_ctrl_vel[0].astype(np.float32))
                train_step_q.append(np.array(step_buffer['q'], dtype=np.float32))
                train_step_qvel.append(np.array(step_buffer['qvel'], dtype=np.float32))
                train_step_lqr.append(np.array(step_buffer['lqr_tau'], dtype=np.float32))
                train_step_vmc.append(np.array(step_buffer['vmc_tau'], dtype=np.float32))
                train_sample_count[0] += 1

            # 保存当前 obs 作为下一个 buffer 的配对观测
            last_ctrl_obs[0] = obs.copy()
            last_ctrl_vel[0] = np.array([d.qvel[0], d.qvel[1], d.qvel[2]], dtype=np.float32)
            last_ctrl_obs_hist[0] = obs_hist.copy()

            # 清空 buffer
            step_buffer['q'].clear()
            step_buffer['qvel'].clear()
            step_buffer['lqr_tau'].clear()
            step_buffer['vmc_tau'].clear()

            # NN 推理 (50Hz)
            nn_tau = None
            if use_nn[0] or (not args.test and args.headless):
                try:
                    oh_t = torch.tensor(obs_hist).unsqueeze(0).to(device)
                    o_t = torch.tensor(obs).unsqueeze(0).to(device)
                    oh_n = (oh_t - enc_mean_t) / enc_std_t
                    o_n = (o_t - obs_mean_t) / obs_std_t
                    with torch.no_grad():
                        est_vel = net.encoder(oh_n)
                        actor_input = torch.cat([o_n, est_vel], dim=-1)
                        nn_action = net.actor(actor_input).cpu().numpy()[0]
                    def _joint_scale(joint_idx):
                        lo, hi = JOINT_LIMITS[joint_idx]
                        return max(hi - DEFAULT_DOF_POS[joint_idx], DEFAULT_DOF_POS[joint_idx] - lo) * ACTION_SCALE_JOINT
                    lf_target = nn_action[0] * _joint_scale(0) + DEFAULT_DOF_POS[0]
                    rf_target = nn_action[1] * _joint_scale(1) + DEFAULT_DOF_POS[1]
                    lr_target = nn_action[2] * _joint_scale(2) + DEFAULT_DOF_POS[2]
                    rr_target = nn_action[3] * _joint_scale(3) + DEFAULT_DOF_POS[3]
                    lw_target = nn_action[4] * ACTION_SCALE_WHEEL
                    rw_target = nn_action[5] * ACTION_SCALE_WHEEL
                    stored_nn_pd_target[0] = (lf_target, rf_target, lr_target, rr_target, lw_target, rw_target)
                    last_actions[0] = nn_action.copy()

                    nn_debug_count[0] += 1
                    if nn_debug_count[0] <= 10:
                        nn_tau_lf = LEG_KP[0] * (lf_target - q_lf) - LEG_KD[0] * qvel[0]
                        nn_tau_rf = LEG_KP[1] * (rf_target - q_rf) - LEG_KD[1] * qvel[1]
                        nn_tau_lr = LEG_KP[2] * (lr_target - q_lr) - LEG_KD[2] * qvel[2]
                        nn_tau_rr = LEG_KP[3] * (rr_target - q_rr) - LEG_KD[3] * qvel[3]
                        nn_tau_lw = WHEEL_KD[0] * (lw_target - qvel[4])
                        nn_tau_rw = WHEEL_KD[1] * (rw_target - qvel[5])
                        nn_tau_debug = np.array([nn_tau_lf, nn_tau_rf, nn_tau_lr, nn_tau_rr, nn_tau_lw, nn_tau_rw])
                        lqr_tau_arr = np.array([tau4_l, tau4_r, tau1_l, tau1_r, wheel_t_l, wheel_t_r])
                        vmc_arr = np.array([vmc_tau4_l, vmc_tau4_r, vmc_tau1_l, vmc_tau1_r, 0.0, 0.0])
                        nn_tau_with_vmc = nn_tau_debug + vmc_arr
                        diff = nn_tau_with_vmc - lqr_tau_arr
                        names = ["lf_τ", "rf_τ", "lr_τ", "rr_τ", "lw_τ", "rw_τ"]
                        items = [f"{n}:{nn:+.3f}|{lqr:+.3f}(Δ{d:+.3f})" for n, nn, lqr, d in zip(names, nn_tau_with_vmc, lqr_tau_arr, diff)]
                        print(f"  [NN debug {nn_debug_count[0]:2d}] {'  '.join(items)}")
                        vel_gt = np.array([d.qvel[0], d.qvel[1], d.qvel[2]])
                        est_v = est_vel.cpu().numpy()[0]
                        print(f"            enc_est(vx,vy,vz)=({est_v[0]:+.3f},{est_v[1]:+.3f},{est_v[2]:+.3f}) "
                              f"gt=({vel_gt[0]:+.3f},{vel_gt[1]:+.3f},{vel_gt[2]:+.3f})")
                except Exception:
                    stored_nn_pd_target[0] = None
                    last_actions[0] = lqr_pd_target_opt.copy()
            else:
                nn_debug_count[0] = 0
                last_actions[0] = lqr_pd_target_opt.copy()

            # 定期训练 (50Hz 触发, 每 train_every 个样本)
            if not args.test and train_sample_count[0] >= args.train_every and train_sample_count[0] % args.train_every == 0:
                o_arr = np.array(train_obs)
                v_arr = np.array(train_velocities)
                sq_arr = np.array(train_step_q)
                sqv_arr = np.array(train_step_qvel)
                sl_arr = np.array(train_step_lqr)
                sv_arr = np.array(train_step_vmc)
                oh_arr = np.array(train_obs_hist)
                o_arr_n = (o_arr - OBS_MEAN) / OBS_STD
                oh_arr_n = (oh_arr - ENC_MEAN) / ENC_STD

                loss_actor = train_step_actor(net, optimizer, loss_fn, o_arr_n, v_arr, sq_arr, sqv_arr, sl_arr, sv_arr, args.batch_size, device)
                loss_enc = train_step_encoder(net, enc_optimizer, loss_fn, oh_arr_n, v_arr, args.batch_size, device)
                last_loss[0] = loss_actor
                alpha = min(train_sample_count[0] / args.mix_steps, 1.0)
                print(f"  [训练] 控步{ctrl_step[0]:5d} | 样本{train_sample_count[0]:6d} | actor_loss={loss_actor:.6f} enc_loss={loss_enc:.6f} | α={alpha:.2f}")

                if train_sample_count[0] % 2000 == 0:
                    net.eval()
                    idx = np.random.choice(len(o_arr), min(5, len(o_arr)), replace=False)
                    with torch.no_grad():
                        bo = torch.tensor(o_arr_n[idx]).to(device)
                        bv = torch.tensor(v_arr[idx]).to(device)
                        bsq = torch.tensor(sq_arr[idx]).to(device)  # (B, D, 6)
                        bsqv = torch.tensor(sqv_arr[idx]).to(device)
                        bsl = torch.tensor(sl_arr[idx]).to(device)
                        bsv = torch.tensor(sv_arr[idx]).to(device)
                        bv_n = (bv - torch.tensor(VEL_MEAN).to(device)) / torch.tensor(VEL_STD).to(device)
                        actor_input = torch.cat([bo, bv_n], dim=-1)
                        raw_action = net.actor(actor_input)
                        kp_t = torch.tensor(LEG_KP, dtype=torch.float32).to(device)
                        kd_t = torch.tensor(LEG_KD, dtype=torch.float32).to(device)
                        wk_t = torch.tensor(WHEEL_KD, dtype=torch.float32).to(device)
                        leg_targets = raw_action[:, :4] * torch.tensor(LEG_SCALES).to(device) + torch.tensor(LEG_DEFAULTS).to(device)
                        leg_tau = kp_t * (leg_targets.unsqueeze(1) - bsq[:, :, :4]) - kd_t * bsqv[:, :, :4]
                        wheel_targets = raw_action[:, 4:] * ACTION_SCALE_WHEEL
                        wheel_tau = wk_t * (wheel_targets.unsqueeze(1) - bsqv[:, :, 4:])
                        pred_tau = torch.cat([leg_tau, wheel_tau], dim=-1)  # (B, D, 6)
                        target = bsl - bsv  # (B, D, 6)
                        step_loss = torch.nn.functional.mse_loss(pred_tau, target, reduction='none').mean(dim=(1, 2))
                    names = ["lf_τ", "rf_τ", "lr_τ", "rr_τ", "lw_τ", "rw_τ"]
                    for i in range(len(idx)):
                        diff = pred_tau[i, 0].cpu().numpy() - target[i, 0].cpu().numpy()
                        items = [f"{n}: {p:+.3f}|{t:+.3f}(Δ{d:+.3f})" for n, p, t, d in zip(names, pred_tau[i, 0].cpu().numpy(), target[i, 0].cpu().numpy(), diff)]
                        print(f"    [{i}] step0: {'  '.join(items)} | 10步loss={step_loss[i]:.4f}")

        # ── 施力 500Hz: 根据当前模式选择力矩来源 ──
        # NN PD 目标 50Hz 更新, 但 PD 力矩每步用当前关节状态重新计算
        if not args.test and args.headless and stored_nn_pd_target[0] is not None:
            # 训练模式: 混合 LQR + NN
            alpha = min(train_sample_count[0] / args.mix_steps, 1.0)
            lf_t, rf_t, lr_t, rr_t, lw_t, rw_t = stored_nn_pd_target[0]
            nn_tau_lf = LEG_KP[0] * (lf_t - q_lf) - LEG_KD[0] * qvel[0]
            nn_tau_rf = LEG_KP[1] * (rf_t - q_rf) - LEG_KD[1] * qvel[1]
            nn_tau_lr = LEG_KP[2] * (lr_t - q_lr) - LEG_KD[2] * qvel[2]
            nn_tau_rr = LEG_KP[3] * (rr_t - q_rr) - LEG_KD[3] * qvel[3]
            nn_tau_lw = WHEEL_KD[0] * (lw_t - qvel[4])
            nn_tau_rw = WHEEL_KD[1] * (rw_t - qvel[5])
            nn_tau_arr = np.array([nn_tau_lf, nn_tau_rf, nn_tau_lr, nn_tau_rr, nn_tau_lw, nn_tau_rw])
            vmc_arr = np.array([vmc_tau4_l, vmc_tau4_r, vmc_tau1_l, vmc_tau1_r, 0.0, 0.0])
            lqr_tau_arr = np.array([tau4_l, tau4_r, tau1_l, tau1_r, wheel_t_l, wheel_t_r])
            mixed_tau = alpha * (nn_tau_arr + vmc_arr) + (1.0 - alpha) * lqr_tau_arr
            tau4_l, tau4_r, tau1_l, tau1_r, wheel_t_l, wheel_t_r = mixed_tau
        elif args.test and use_nn[0] and stored_nn_pd_target[0] is not None:
            # Play 模式: NN PD + VMC
            lf_t, rf_t, lr_t, rr_t, lw_t, rw_t = stored_nn_pd_target[0]
            tau4_l = LEG_KP[0] * (lf_t - q_lf) - LEG_KD[0] * qvel[0] + vmc_tau4_l
            tau4_r = LEG_KP[1] * (rf_t - q_rf) - LEG_KD[1] * qvel[1] + vmc_tau4_r
            tau1_l = LEG_KP[2] * (lr_t - q_lr) - LEG_KD[2] * qvel[2] + vmc_tau1_l
            tau1_r = LEG_KP[3] * (rr_t - q_rr) - LEG_KD[3] * qvel[3] + vmc_tau1_r
            wheel_t_l = WHEEL_KD[0] * (lw_t - qvel[4])
            wheel_t_r = WHEEL_KD[1] * (rw_t - qvel[5])
        # else: 使用 LQR 力矩 (已在上方 lqr_control 计算)

        # 施力
        d.ctrl[act_ids[0]] = tau4_l
        d.ctrl[act_ids[1]] = tau4_r
        d.ctrl[act_ids[2]] = tau1_l
        d.ctrl[act_ids[3]] = tau1_r
        d.ctrl[act_ids[4]] = wheel_t_l
        d.ctrl[act_ids[5]] = wheel_t_r

        # 打印: viewer 每 200 步, headless 每 2000 步
        print_every = 200 if not args.headless else 2000
        if step_count[0] % print_every == 0:
            roll = np.arcsin(np.clip(-gravity_body[1], -1.0, 1.0))
            if args.headless and not args.test:
                alpha = min(train_sample_count[0] / args.mix_steps, 1.0)
                mode_str = f"α={alpha:.2f}"
            else:
                mode_str = 'NN' if use_nn[0] else 'LQR'
            print(f"  [{step_count[0]:5d}] pitch={np.rad2deg(pitch):+.1f}° roll={np.rad2deg(roll):+.1f}° "
                  f"yaw={np.rad2deg(yaw):+.0f}° "
                  f"L0=({L0_l:.3f},{L0_r:.3f})m z={d.qpos[2]:.3f}m "
                  f"mode={mode_str} samples={train_sample_count[0]}")

        # 摔倒检测: pitch > 60° 或 z < 0.05m
        if not args.test and args.headless:
            if abs(pitch) > np.deg2rad(60) or d.qpos[2] < 0.05:
                if nn_test_mode[0]:
                    nn_test_fall[0] = True
                need_reset[0] = True

    def key_callback(keycode):
        step = 0.005
        step_angle = np.deg2rad(2.0)
        step_ff = 5.0

        if keycode == 78:  # N: 切换 LQR/NN
            use_nn[0] = not use_nn[0]
            print(f">>> 模式: {'神经网络' if use_nn[0] else 'LQR'} <<<")
        elif keycode == 77:  # M: 保存模型
            if train_sample_count[0] > 0:
                save_model()
            else:
                print(">>> 无训练数据，跳过保存 <<<")
        elif keycode == 87:  # W
            params['L0_target'] = min(params['L0_target'] + step, 0.45)
            print(f">>> L0={params['L0_target']:.3f}m <<<")
        elif keycode == 83:  # S
            params['L0_target'] = max(params['L0_target'] - step, 0.10)
            print(f">>> L0={params['L0_target']:.3f}m <<<")
        elif keycode == 65:  # A
            params['phi0_target'] = max(params['phi0_target'] - step_angle, np.deg2rad(40))
            print(f">>> φ₀={np.rad2deg(params['phi0_target']):.0f}° <<<")
        elif keycode == 68:  # D
            params['phi0_target'] = min(params['phi0_target'] + step_angle, np.deg2rad(140))
            print(f">>> φ₀={np.rad2deg(params['phi0_target']):.0f}° <<<")
        elif keycode == 81:  # Q
            params['feedforward'] = max(params['feedforward'] - step_ff, 0.0)
            print(f">>> ff={params['feedforward']:.0f}N <<<")
        elif keycode == 69:  # E
            params['feedforward'] = min(params['feedforward'] + step_ff, 200.0)
            print(f">>> ff={params['feedforward']:.0f}N <<<")
        elif keycode == 82:  # R: L0+ (备用)
            params['L0_target'] = min(params['L0_target'] + step, 0.45)
            print(f">>> L0={params['L0_target']:.3f}m <<<")
        elif keycode == 84:  # T: L0- (备用)
            params['L0_target'] = max(params['L0_target'] - step, 0.10)
            print(f">>> L0={params['L0_target']:.3f}m <<<")
        elif keycode == 70:  # F: kp_leg-
            params['kp_leg'] = max(params['kp_leg'] - 100, 100)
            print(f">>> kp_leg={params['kp_leg']:.0f} <<<")
        elif keycode == 71:  # G: kp_leg+
            params['kp_leg'] = min(params['kp_leg'] + 100, 5000)
            print(f">>> kp_leg={params['kp_leg']:.0f} <<<")
        elif keycode == 86:  # V: kp_roll-
            params['kp_roll'] = max(params['kp_roll'] - 50, 0)
            print(f">>> kp_roll={params['kp_roll']:.0f} <<<")
        elif keycode == 66:  # B: kp_roll+
            params['kp_roll'] = min(params['kp_roll'] + 50, 3000)
            print(f">>> kp_roll={params['kp_roll']:.0f} <<<")
        elif keycode == 67:  # C: kd_roll-
            params['kd_roll'] = max(params['kd_roll'] - 5, 0)
            print(f">>> kd_roll={params['kd_roll']:.0f} <<<")
        elif keycode == 88:  # X: kd_roll+
            params['kd_roll'] = min(params['kd_roll'] + 5, 200)
            print(f">>> kd_roll={params['kd_roll']:.0f} <<<")
        elif keycode == 89:  # Y: kp_yaw-
            params['kp_yaw'] = max(params['kp_yaw'] - 2, 0)
            print(f">>> kp_yaw={params['kp_yaw']:.0f} <<<")
        elif keycode == 85:  # U: kp_yaw+
            params['kp_yaw'] = min(params['kp_yaw'] + 2, 100)
            print(f">>> kp_yaw={params['kp_yaw']:.0f} <<<")
        elif keycode == 73:  # I: kd_yaw-
            params['kd_yaw'] = max(params['kd_yaw'] - 0.5, 0)
            print(f">>> kd_yaw={params['kd_yaw']:.1f} <<<")
        elif keycode == 79:  # O: kd_yaw+
            params['kd_yaw'] = min(params['kd_yaw'] + 0.5, 20)
            print(f">>> kd_yaw={params['kd_yaw']:.1f} <<<")
        elif keycode == 32:  # Space
            paused[0] = not paused[0]
            print(f">>> {'暂停' if paused[0] else '继续'} <<<")
        elif keycode == 259:  # Backspace
            need_reset[0] = True
        elif keycode == 265:  # Up: 前进
            params['cmd_vel'] += 0.5
            print(f">>> cmd_vel={params['cmd_vel']:.1f} m/s <<<")
        elif keycode == 264:  # Down: 后退
            params['cmd_vel'] -= 0.5
            print(f">>> cmd_vel={params['cmd_vel']:.1f} m/s <<<")
        elif keycode == 263:  # Left: 左转
            params['cmd_yaw_rate'] += 0.5
            print(f">>> cmd_yaw_rate={params['cmd_yaw_rate']:.1f} rad/s <<<")
        elif keycode == 262:  # Right: 右转
            params['cmd_yaw_rate'] -= 0.5
            print(f">>> cmd_yaw_rate={params['cmd_yaw_rate']:.1f} rad/s <<<")
        elif keycode == 72:  # H: 停止
            params['cmd_vel'] = 0.0
            params['cmd_yaw_rate'] = 0.0
            print(">>> 停止 <<<")
        elif keycode == 256:  # ESC
            save_model()
            return True

        return False

    if args.headless:
        if args.dagger:
            print(f"DAgger模式: NN(50控步) ↔ LQR(50控步), 每{args.train_every}控步训练, 每100控步自动重置")
        else:
            print(f"无头模式: 混合输出 α:0→1 (共{args.mix_steps}控步), 每{args.train_every}控步训练, 每1000控步自动重置")
        from tqdm import tqdm
        total_steps = 0
        pbar = tqdm()
        nn_test_mode = [False]
        nn_test_step = [0]
        nn_test_fall = [False]  # NN 测试中是否摔倒
        dagger_phase = [0]  # 0=LQR, 1=NN
        while True:
            try:
                if need_reset[0]:
                    reset_sim()
                    need_reset[0] = False
                if need_forward[0]:
                    mujoco.mj_forward(model, data)
                    need_forward[0] = False

                if args.dagger:
                    # DAgger: 每500步切换 LQR ↔ NN
                    if step_count[0] >= 500:
                        dagger_phase[0] = 1 - dagger_phase[0]
                        need_reset[0] = True
                        total_steps += 1
                        reset_sim()
                        mujoco.mj_forward(model, data)
                        need_forward[0] = False
                        use_nn[0] = (dagger_phase[0] == 1)
                        mode_str = "NN" if use_nn[0] else "LQR"
                        print(f"\n  [DAgger] 切换→{mode_str}, 步{total_steps}, 样本{train_sample_count[0]}")
                        continue
                    elif total_steps == 0:
                        use_nn[0] = True
                        dagger_phase[0] = 1
                        print(f"  [DAgger] 开始, NN先行, 步{total_steps}, 样本{train_sample_count[0]}")
                else:
                    # 原 NN 测试模式: 每 5000 步用 NN 跑, 上限 1200 步 (≈6s @ 0.005s dt)
                    if nn_test_mode[0]:
                        if nn_test_step[0] >= 1200 or nn_test_fall[0]:
                            nn_test_mode[0] = False
                            use_nn[0] = False
                            fall_info = "摔倒" if nn_test_fall[0] else "完成"
                            print(f"  [NN测试] {fall_info}, 存活{nn_test_step[0]}步, 步{total_steps}")
                            nn_test_step[0] = 0
                            nn_test_fall[0] = False
                            total_steps += 1
                            reset_sim()
                            mujoco.mj_forward(model, data)
                            need_forward[0] = False
                            continue
                        nn_test_step[0] += 1
                    elif total_steps > 0 and total_steps % 5000 == 0 and train_sample_count[0] > 1000:
                        nn_test_mode[0] = True
                        nn_test_step[0] = 0
                        nn_test_fall[0] = False
                        use_nn[0] = True
                        total_steps += 1
                        reset_sim()
                        mujoco.mj_forward(model, data)
                        need_forward[0] = False
                        print(f"\n  [NN测试] 开始, 步{total_steps}, 样本{train_sample_count[0]}")
                        continue

                controller(model, data)
                mujoco.mj_step(model, data)

                if ctrl_step[0] >= 1000:
                    need_reset[0] = True

                total_steps += 1
                if not args.test and total_steps % 500 == 0:
                    params['cmd_vel'] = np.random.uniform(-1.5, 1.5)
                    params['cmd_yaw_rate'] = np.random.uniform(-2.0, 2.0)
                    params['L0_target'] = np.random.uniform(0.20, 0.40)
                pbar.update(1)
                alpha = min(train_sample_count[0] / args.mix_steps, 1.0)
                mode_tag = f"[{'NN' if use_nn[0] else 'LQR'}] " if args.dagger else ""
                pbar.set_description(f"{mode_tag}样本{train_sample_count[0]:6d} loss={last_loss[0]:.6f} α={alpha:.2f}" if last_loss[0] > 0 else f"{mode_tag}样本{train_sample_count[0]:6d} α={alpha:.2f}")
            except KeyboardInterrupt:
                print("\n中断，保存模型...")
                break

        save_model()
        return

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False, key_callback=key_callback,
    ) as viewer:
        while viewer.is_running():
            if need_reset[0]:
                reset_sim()
                need_reset[0] = False
            if need_forward[0]:
                mujoco.mj_forward(model, data)
                need_forward[0] = False
            controller(model, data)
            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.qpos[0:3]
            viewer.sync()

    save_model() if train_sample_count[0] > 0 else None
    print("退出.")


if __name__ == "__main__":
    main()