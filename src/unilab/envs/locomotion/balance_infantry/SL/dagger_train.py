#!/usr/bin/env python3
"""DAgger 在线训练: LQR 跑仿真 → 同时收集数据 → 在线训练 NN → 交替用 NN 跑收集新数据。

用法:
    python SL/dagger_train.py --scene flat --rounds 5 --epochs 50
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

from train_sl import LQRActor, OBS_HISTORY_LENGTH, NUM_OBS, NUM_ACTIONS, NUM_ENCODER_OBS
from test_vmc_lqr import (
    fk_fivebar, compute_jacobian_partials, inverse_pd_targets,
    compute_lqr_gains, lqr_virtual_torque,
)

# ═══════════════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════════════
LEG_KP = (100.0, 100.0, 100.0, 100.0)
LEG_KD = (5.0, 5.0, 5.0, 5.0)
WHEEL_KD = (2.0, 2.0)
DEFAULT_DOF_POS = np.array([0.0, 0.0, -0.6, -0.6, 0.0, 0.0], dtype=np.float64)
L0_TARGET = 0.35
FEEDFORWARD = 80.0


def build_observation(model, data, dof_pos, dof_vel, last_actions, cmd_vel, cmd_yaw_rate, cmd_height):
    gyro = data.qvel[3:6].copy()
    qw, qx, qy, qz = data.qpos[3:7]
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ])
    gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
    pitch = np.arcsin(np.clip(gravity_body[0], -1.0, 1.0))
    roll = np.arcsin(np.clip(-gravity_body[1], -1.0, 1.0))
    commands = np.array([cmd_vel, cmd_yaw_rate, cmd_height, 0.0])
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    yaw_error = 0.0

    q_lr, q_rr, q_lf, q_rf = dof_pos[2], dof_pos[3], dof_pos[0], dof_pos[1]
    phi1_l = q_lr + 5 * np.pi / 6
    phi1_r = q_rr + 5 * np.pi / 6
    phi4_l = q_lf + np.pi / 6
    phi4_r = q_rf + np.pi / 6
    L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
    L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)
    theta_l = np.pi / 2 - phi0_l - pitch
    theta_r = np.pi / 2 - phi0_r - pitch
    dof_pos_err = dof_pos[:4] - DEFAULT_DOF_POS[:4]

    obs = np.concatenate([
        gyro,
        np.array([pitch]), np.array([roll]),
        commands,
        last_actions,
        np.array([0.0]), np.array([0.0]),
        np.array([np.sin(yaw_error)]), np.array([np.cos(yaw_error)]),
        dof_pos_err, dof_vel,
        np.array([theta_l]), np.array([theta_r]),
        np.array([L0_l]), np.array([L0_r]),
    ])
    return obs.astype(np.float32)


def lqr_step(model, data, q, qvel, L0_l, L0_r, phi0_l, phi0_r, phi2_l, phi3_l,
             sigma1_l, phi2_r, phi3_r, sigma1_r, pitch, pitch_dot, base_vx, dt,
             lqr_state, prev_lqr, cmd_vel, step_count):
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
    F_l = 1000.0 * (L0_TARGET - L0_l) - 100.0 * L0_dot_l + FEEDFORWARD
    F_r = 1000.0 * (L0_TARGET - L0_r) - 100.0 * L0_dot_r + FEEDFORWARD
    prev_lqr['L0_l'] = L0_l
    prev_lqr['L0_r'] = L0_r

    tau4_l = np.clip(J01_l * F_l + J11_l * T_l, -40.0, 40.0)
    tau1_l = np.clip(J00_l * F_l + J10_l * T_l, -40.0, 40.0)
    tau4_r = np.clip(J01_r * F_r + J11_r * T_r, -40.0, 40.0)
    tau1_r = np.clip(J00_r * F_r + J10_r * T_r, -40.0, 40.0)

    return (tau4_l, tau4_r, tau1_l, tau1_r, wheel_t_l, wheel_t_r)


def get_obs_norm():
    obs_mean = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.29,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.3, 0.3, 0.28, 0.28,
    ], dtype=np.float32)
    obs_std = np.array([
        1.5, 1.5, 1.5, 0.15, 0.10,
        1.5, 2.0, 0.08,
        0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
        0.3, 0.3, 0.6, 0.6,
        0.3, 0.3, 0.3, 0.3,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        0.4, 0.4, 0.05, 0.05,
    ], dtype=np.float32)
    enc_mean = np.tile(obs_mean, OBS_HISTORY_LENGTH).astype(np.float32)
    enc_std = np.tile(obs_std, OBS_HISTORY_LENGTH).astype(np.float32)
    return enc_mean, enc_std, obs_mean, obs_std


def train_one_epoch(model, optimizer, loss_fn, obs_hist, obs, targets, batch_size, device):
    model.train()
    N = len(obs_hist)
    perm = np.random.permutation(N)
    total_loss = 0.0
    noise_std = 0.05
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        bh = torch.tensor(obs_hist[idx]).to(device)
        bo = torch.tensor(obs[idx]).to(device)
        bt = torch.tensor(targets[idx]).to(device)
        bh = bh + torch.randn_like(bh) * noise_std
        bo = bo + torch.randn_like(bo) * noise_std
        pred = model(bh, bo)
        loss = loss_fn(pred, bt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    return total_loss / N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="flat")
    parser.add_argument("--rounds", type=int, default=5, help="DAgger 轮数")
    parser.add_argument("--epochs", type=int, default=50, help="每轮训练 epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="dagger_pretrain.pth")
    parser.add_argument("--no_viewer", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    # ── 场景 ──
    scene_dir = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "assets", "robots", "balance_infantry")
    scene_map = {"flat": os.path.join(scene_dir, "scene_flat.xml")}
    scene_path = scene_map.get(args.scene, scene_map["flat"])
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    joint_names = [
        "left_front_joint", "right_front_joint", "left_rear_joint", "right_rear_joint",
        "left_wheel_joint", "right_wheel_joint",
    ]
    joint_qpos = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                  for n in joint_names]
    joint_dofadr = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                    for n in joint_names]
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_actuator") for n in joint_names]
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    dt = model.opt.timestep

    try:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    except Exception:
        pass
    init_z = 0.35
    data.qpos[2] = init_z
    data.qvel[0:6] = 0.0

    default_dof_pos = np.array([data.qpos[joint_qpos[0]], data.qpos[joint_qpos[1]],
                                 data.qpos[joint_qpos[2]], data.qpos[joint_qpos[3]]], dtype=np.float32)

    enc_mean, enc_std, obs_mean, obs_std = get_obs_norm()
    enc_mean_t = torch.tensor(enc_mean).to(device)
    enc_std_t = torch.tensor(enc_std).to(device)
    obs_mean_t = torch.tensor(obs_mean).to(device)
    obs_std_t = torch.tensor(obs_std).to(device)

    net = LQRActor().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    all_obs_hist = []
    all_obs = []
    all_targets = []

    print(f"DAgger 训练: {args.rounds} 轮, 每轮 {args.epochs} epoch, 设备={args.device}")
    print(f"  timestep={dt:.4f}s, L0_target={L0_TARGET:.2f}m\n")

    for round_idx in range(args.rounds):
        print(f"{'='*60}")
        print(f"  DAgger 第 {round_idx + 1}/{args.rounds} 轮")

        # 重置仿真
        try:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        except Exception:
            mujoco.mj_resetData(model)
        data.qpos[2] = init_z
        data.qvel[0:6] = 0.0

        lqr_state = {'L': np.zeros(6, dtype=np.float64), 'R': np.zeros(6, dtype=np.float64)}
        prev_lqr = {'L0_l': -1.0, 'L0_r': -1.0}

        last_actions = np.zeros(6, dtype=np.float32)
        obs_history = deque(maxlen=OBS_HISTORY_LENGTH)
        round_obs = []
        round_targets = []
        step_count = 0
        max_steps = 3000

        # 本轮用 NN 还是 LQR 控制
        use_nn = (round_idx > 0)

        while step_count < max_steps:
            q = np.array([data.qpos[joint_qpos[i]] for i in range(6)], dtype=np.float64)
            qvel = np.array([data.qvel[joint_dofadr[i]] for i in range(6)], dtype=np.float64)

            q_lr, q_rr, q_lf, q_rf = q[2], q[3], q[0], q[1]
            phi1_l = q_lr + 5 * np.pi / 6
            phi1_r = q_rr + 5 * np.pi / 6
            phi4_l = q_lf + np.pi / 6
            phi4_r = q_rf + np.pi / 6
            L0_l, phi0_l, phi2_l, phi3_l, sigma1_l = fk_fivebar(phi1_l, phi4_l)
            L0_r, phi0_r, phi2_r, phi3_r, sigma1_r = fk_fivebar(phi1_r, phi4_r)

            qw, qx, qy, qz = data.qpos[3:7]
            Rmat = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
            ])
            gravity_body = Rmat.T @ np.array([0.0, 0.0, -1.0])
            pitch = np.arcsin(np.clip(gravity_body[0], -1.0, 1.0))
            pitch_dot = data.qvel[4]
            base_vx = data.qvel[0]

            terrain_z = 0.0
            base_z = data.qpos[2]
            if base_z - terrain_z < 0.05:
                break

            # LQR 计算（始终计算，用于标注）
            lqr_tau = lqr_step(model, data, q, qvel, L0_l, L0_r, phi0_l, phi0_r,
                               phi2_l, phi3_l, sigma1_l, phi2_r, phi3_r, sigma1_r,
                               pitch, pitch_dot, base_vx, dt, lqr_state, prev_lqr, 0.0, step_count)
            tau4_l, tau4_r, tau1_l, tau1_r, wheel_t_l, wheel_t_r = lqr_tau

            qvel_lw = qvel[4]
            qvel_rw = qvel[5]
            lqr_pd_target = np.array(inverse_pd_targets(
                tau4_l, tau4_r, tau1_l, tau1_r, wheel_t_l, wheel_t_r,
                q_lf, q_rf, q_lr, q_rr,
                qvel[0], qvel[1], qvel[2], qvel[3], qvel_lw, qvel_rw,
                default_dof_pos,
            ))

            # 构建观测
            obs = build_observation(model, data, q[:4], qvel[:6], last_actions, 0.0, 0.0, L0_TARGET)
            obs_history.append(obs)
            if len(obs_history) == OBS_HISTORY_LENGTH:
                obs_hist = np.concatenate(list(obs_history))
            else:
                obs_hist = np.tile(obs, OBS_HISTORY_LENGTH)

            round_obs.append(obs_hist.astype(np.float32))
            round_targets.append(lqr_pd_target.astype(np.float32))

            # NN 推理
            if use_nn:
                oh_t = torch.tensor(obs_hist).unsqueeze(0).to(device)
                o_t = torch.tensor(obs).unsqueeze(0).to(device)
                oh_n = (oh_t - enc_mean_t) / enc_std_t
                o_n = (o_t - obs_mean_t) / obs_std_t
                with torch.no_grad():
                    nn_action = net(oh_n, o_n).cpu().numpy()[0]
                # 用 NN 控制
                tau_lf = LEG_KP[0] * (nn_action[0] - q_lf) - LEG_KD[0] * qvel[0]
                tau_rf = LEG_KP[1] * (nn_action[1] - q_rf) - LEG_KD[1] * qvel[1]
                tau_lr = LEG_KP[2] * (nn_action[2] - q_lr) - LEG_KD[2] * qvel[2]
                tau_rr = LEG_KP[3] * (nn_action[3] - q_rr) - LEG_KD[3] * qvel[3]
                tau_lw = WHEEL_KD[0] * (nn_action[4] - qvel_lw)
                tau_rw = WHEEL_KD[1] * (nn_action[5] - qvel_rw)
                last_actions = nn_action.copy()
            else:
                last_actions = lqr_pd_target.copy()

            # 施力
            data.ctrl[act_ids[0]] = tau4_l if not use_nn else tau_lf
            data.ctrl[act_ids[1]] = tau4_r if not use_nn else tau_rf
            data.ctrl[act_ids[2]] = tau1_l if not use_nn else tau_lr
            data.ctrl[act_ids[3]] = tau1_r if not use_nn else tau_rr
            data.ctrl[act_ids[4]] = wheel_t_l if not use_nn else tau_lw
            data.ctrl[act_ids[5]] = wheel_t_r if not use_nn else tau_rw

            mujoco.mj_step(model, data)
            step_count += 1

        # 本轮收集的数据
        n_round = len(round_obs)
        print(f"  收集 {n_round} 条样本 (控制: {'NN' if use_nn else 'LQR'})")

        if n_round < OBS_HISTORY_LENGTH:
            continue

        # 归一化
        round_obs_arr = np.array(round_obs)
        round_tgt_arr = np.array(round_targets)
        round_obs_arr = (round_obs_arr - enc_mean) / enc_std
        round_obs_norm = (np.array([o[-NUM_OBS:] for o in round_obs]) - obs_mean) / obs_std

        all_obs_hist.append(round_obs_arr)
        all_obs.append(round_obs_norm)
        all_targets.append(round_tgt_arr)

        # 合并所有数据训练
        train_hist = np.concatenate(all_obs_hist, axis=0)
        train_obs = np.concatenate(all_obs, axis=0)
        train_tgt = np.concatenate(all_targets, axis=0)
        print(f"  总训练集: {len(train_hist)} 条, 开始训练 {args.epochs} epoch...")

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(net, optimizer, loss_fn, train_hist, train_obs, train_tgt, args.batch_size, device)
            if epoch % 10 == 0 or epoch == 1:
                print(f"    epoch {epoch:3d}/{args.epochs} | loss={loss:.6f}")

        # 保存
        output_path = os.path.join(_SCRIPT_DIR, args.output)
        torch.save({
            'encoder_state_dict': net.encoder.state_dict(),
            'actor_state_dict': net.actor.state_dict(),
            'enc_mean': enc_mean,
            'enc_std': enc_std,
            'obs_mean': obs_mean,
            'obs_std': obs_std,
            'round': round_idx + 1,
            'val_loss': loss,
        }, output_path)
        print(f"  保存 → {output_path}\n")

    print(f"{'='*60}")
    print("DAgger 训练完成!")


if __name__ == "__main__":
    main()