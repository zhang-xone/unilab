#!/usr/bin/env python3
"""测试预训练模型在仿真中的表现。

用法:
    python SL/test_pretrain.py --scene flat
    python SL/test_pretrain.py --scene flat --model SL/lqr_pretrain.pth
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import torch

# ── 路径设置 ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.join(_SCRIPT_DIR, "..")          # balance_infantry
_LQR_DIR = os.path.join(_PARENT_DIR, "lqr")            # balance_infantry/lqr

sys.path.insert(0, _PARENT_DIR)
sys.path.insert(0, _LQR_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from train_sl import LQRActor, OBS_HISTORY_LENGTH, NUM_OBS, NUM_ACTIONS
from test_vmc_lqr import (
    fk_fivebar,
    compute_jacobian_partials,
    inverse_pd_targets,
    compute_lqr_gains,
    lqr_virtual_torque,
)

# ═══════════════════════════════════════════════════════════════
# PD 参数 (与 base.py 一致)
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

DEFAULT_DOF_POS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def raw_action_to_torque(raw_action, dof_pos, dof_vel, default_dof_pos):
    """RL 原始动作 → 关节力矩 (与 joystick.py 一致)."""
    lf_a, rf_a, lr_a, rr_a, lw_a, rw_a = raw_action

    def joint_scale(idx):
        lo, hi = JOINT_LIMITS[idx]
        return max(hi - default_dof_pos[idx], default_dof_pos[idx] - lo) * ACTION_SCALE_JOINT

    lf_target = lf_a * joint_scale(0) + default_dof_pos[0]
    rf_target = rf_a * joint_scale(1) + default_dof_pos[1]
    lr_target = lr_a * joint_scale(2) + default_dof_pos[2]
    rr_target = rr_a * joint_scale(3) + default_dof_pos[3]
    lw_target = lw_a * ACTION_SCALE_WHEEL
    rw_target = rw_a * ACTION_SCALE_WHEEL

    tau_lf = LEG_KP[0] * (lf_target - dof_pos[0]) - LEG_KD[0] * dof_vel[0]
    tau_rf = LEG_KP[1] * (rf_target - dof_pos[1]) - LEG_KD[1] * dof_vel[1]
    tau_lr = LEG_KP[2] * (lr_target - dof_pos[2]) - LEG_KD[2] * dof_vel[2]
    tau_rr = LEG_KP[3] * (rr_target - dof_pos[3]) - LEG_KD[3] * dof_vel[3]
    tau_lw = WHEEL_KD[0] * (lw_target - dof_vel[4])
    tau_rw = WHEEL_KD[1] * (rw_target - dof_vel[5])

    return np.array([tau_lf, tau_rf, tau_lr, tau_rr, tau_lw, tau_rw])


def pd_target_to_torque(pd_target, dof_pos, dof_vel):
    """NN 输出的物理 PD 目标 → 关节力矩 (PD 目标已是物理量)."""
    tau_lf = LEG_KP[0] * (pd_target[0] - dof_pos[0]) - LEG_KD[0] * dof_vel[0]
    tau_rf = LEG_KP[1] * (pd_target[1] - dof_pos[1]) - LEG_KD[1] * dof_vel[1]
    tau_lr = LEG_KP[2] * (pd_target[2] - dof_pos[2]) - LEG_KD[2] * dof_vel[2]
    tau_rr = LEG_KP[3] * (pd_target[3] - dof_pos[3]) - LEG_KD[3] * dof_vel[3]
    tau_lw = WHEEL_KD[0] * (pd_target[4] - dof_vel[4])
    tau_rw = WHEEL_KD[1] * (pd_target[5] - dof_vel[5])
    return np.array([tau_lf, tau_rf, tau_lr, tau_rr, tau_lw, tau_rw])


def build_observation_raw(model, data, dof_pos, dof_vel, last_actions, cmd_vel, cmd_yaw_rate, cmd_height):
    """构建与 RL 一致的 32 维观测."""
    gyro = data.qvel[3:6].copy()

    qw, qx, qy, qz = data.qpos[3:7]
    gravity_world = np.array([0.0, 0.0, -1.0])
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ])
    gravity_body = R.T @ gravity_world
    pitch = np.arcsin(np.clip(gravity_body[0], -1.0, 1.0))
    roll = np.arcsin(np.clip(-gravity_body[1], -1.0, 1.0))

    commands = np.array([cmd_vel, cmd_yaw_rate, cmd_height])
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


def lqr_control_step(q, qvel, L0_l, L0_r, phi0_l, phi0_r, phi2_l, phi3_l, sigma1_l,
                      phi2_r, phi3_r, sigma1_r, pitch, pitch_dot, base_vx, dt,
                      lqr_state, prev_lqr, cmd_vel, step_count):
    """LQR 平衡控制一步，返回 (tau_lr, tau_lf, tau_rr, tau_rf, tau_lw, tau_rw)."""
    phi1_l = q[2] + 5 * np.pi / 6
    phi1_r = q[3] + 5 * np.pi / 6
    phi4_l = q[0] + np.pi / 6
    phi4_r = q[1] + np.pi / 6

    qvel_lr, qvel_rr = qvel[2], qvel[3]
    qvel_lf, qvel_rf = qvel[0], qvel[1]

    # Jacobian
    J00_l, J01_l, J10_l, J11_l = compute_jacobian_partials(
        phi1_l, phi4_l, phi0_l, phi2_l, phi3_l, sigma1_l, L0_l)
    J00_r, J01_r, J10_r, J11_r = compute_jacobian_partials(
        phi1_r, phi4_r, phi0_r, phi2_r, phi3_r, sigma1_r, L0_r)

    # LQR state
    theta_l = np.pi / 2 - phi0_l - pitch
    theta_r = np.pi / 2 - phi0_r - pitch
    theta_dot_l = -(J10_l * qvel_lr + J11_l * qvel_lf)
    theta_dot_r = -(J10_r * qvel_rr + J11_r * qvel_rf)

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

    lqr_ramp = min(1.0, step_count / 300.0)
    T_l = np.clip(T_lqr_l * lqr_ramp, -20.0, 20.0)
    T_r = np.clip(T_lqr_r * lqr_ramp, -20.0, 20.0)
    wheel_t_l = np.clip(wheel_t_l * lqr_ramp, -10.0, 10.0)
    wheel_t_r = np.clip(wheel_t_r * lqr_ramp, -10.0, 10.0)

    # VMC leg force
    L0_target = 0.35
    ff = 80.0
    L0_dot_l = (L0_l - prev_lqr['L0_l']) / dt if prev_lqr['L0_l'] > 0 else 0.0
    L0_dot_r = (L0_r - prev_lqr['L0_r']) / dt if prev_lqr['L0_r'] > 0 else 0.0
    F_l = 1000.0 * (L0_target - L0_l) - 100.0 * L0_dot_l + ff
    F_r = 1000.0 * (L0_target - L0_r) - 100.0 * L0_dot_r + ff

    prev_lqr['L0_l'] = L0_l
    prev_lqr['L0_r'] = L0_r

    # Jacobian → joint torques
    tau1_l = np.clip(J00_l * F_l + J10_l * T_l, -40.0, 40.0)
    tau4_l = np.clip(J01_l * F_l + J11_l * T_l, -40.0, 40.0)
    tau1_r = np.clip(J00_r * F_r + J10_r * T_r, -40.0, 40.0)
    tau4_r = np.clip(J01_r * F_r + J11_r * T_r, -40.0, 40.0)

    return (tau4_l, tau4_r, tau1_l, tau1_r, wheel_t_l, wheel_t_r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="flat")
    parser.add_argument("--model", type=str, default="SL/lqr_pretrain.pth")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # ── 加载场景 ──
    scene_dir = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "assets", "robots", "balance_infantry")
    scene_map = {"flat": os.path.join(scene_dir, "scene_flat.xml")}
    model_path = scene_map.get(args.scene, scene_map["flat"])
    print(f"加载场景: {model_path}")

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # ── 加载预训练模型 ──
    model_file = args.model
    if not os.path.isabs(model_file):
        model_file = os.path.join(_PARENT_DIR, model_file)
    print(f"加载模型: {model_file}")
    ckpt = torch.load(model_file, map_location=args.device, weights_only=False)
    net = LQRActor().to(args.device)
    net.encoder.load_state_dict(ckpt["encoder_state_dict"])
    net.actor.load_state_dict(ckpt["actor_state_dict"])
    net.eval()
    enc_mean = torch.tensor(ckpt["enc_mean"]).to(args.device)
    enc_std = torch.tensor(ckpt["enc_std"]).to(args.device)
    obs_mean = torch.tensor(ckpt["obs_mean"]).to(args.device)
    obs_std = torch.tensor(ckpt["obs_std"]).to(args.device)

    # ── 关节索引 ──
    joint_names = [
        "left_front_joint", "right_front_joint", "left_rear_joint", "right_rear_joint",
        "left_wheel_joint", "right_wheel_joint",
    ]
    joint_qpos = [mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                  for n in joint_names]
    act_names = [n + "_actuator" for n in joint_names]
    act_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]

    # ── 初始状态 ──
    try:
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    except Exception:
        pass
    mj_data.qpos[2] = 0.35           # 抬高出生高度
    mj_data.qvel[0:6] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # 改 qpos0 让 Backspace 复位也抬高
    mj_model.qpos0 = mj_data.qpos.copy()

    # ── 观测历史缓冲区 ──
    obs_history_buffer = []
    last_actions = np.zeros(6, dtype=np.float32)

    # ── 命令 ──
    cmd_vel = [0.0]
    cmd_yaw_rate = [0.0]
    cmd_height = [0.29]

    # ── 模式 ──
    use_nn = [True]

    # ── LQR 状态 ──
    lqr_state = {'L': np.zeros(6), 'R': np.zeros(6)}
    prev_lqr = {'L0_l': 0.0, 'L0_r': 0.0}
    step_count = [0]

    print(f"  timestep={0.002:.4f}s, L0_target=0.35m, phi0_target=90°, ff=80.0N")
    print(f"  模式={'神经网络' if use_nn[0] else 'LQR'}")
    print(f"\n  {'='*50}")
    print(f"  预训练网络测试 (场景: {args.scene})")
    print(f"  N=神经网络  L=原始LQR  Space=暂停  Esc=退出")
    print(f"  {'='*50}\n")

    paused = [False]

    def controller(model, data):
        nonlocal last_actions
        dt = model.opt.timestep
        step_count[0] += 1

        # ── 读取关节状态 ──
        q = np.array([data.qpos[idx] for idx in joint_qpos])
        qvel = np.array([data.qvel[model.jnt_dofadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        ]] for n in joint_names])

        # ── 构建观测 ──
        obs = build_observation_raw(
            model, data, q, qvel, last_actions,
            cmd_vel[0], cmd_yaw_rate[0], cmd_height[0],
        )

        # ── 维护观测历史 ──
        obs_history_buffer.append(obs)
        if len(obs_history_buffer) > OBS_HISTORY_LENGTH:
            obs_history_buffer.pop(0)

        # ── 推理 ──
        if use_nn[0]:
            if len(obs_history_buffer) == OBS_HISTORY_LENGTH:
                obs_hist = np.concatenate(obs_history_buffer).astype(np.float32)
            else:
                obs_hist = np.tile(obs, OBS_HISTORY_LENGTH).astype(np.float32)
            obs_hist_t = torch.tensor(obs_hist).unsqueeze(0).to(args.device)
            obs_t = torch.tensor(obs).unsqueeze(0).to(args.device)
            obs_hist_norm = (obs_hist_t - enc_mean) / enc_std
            obs_norm = (obs_t - obs_mean) / obs_std
            with torch.no_grad():
                raw_action = net(obs_hist_norm, obs_norm).cpu().numpy()[0]
            tau = raw_action_to_torque(raw_action, q, qvel, DEFAULT_DOF_POS)
        else:
            # LQR 直接输出力矩
            q_lf, q_rf, q_lr, q_rr = q[0], q[1], q[2], q[3]
            phi1_l = q_lr + 5 * np.pi / 6
            phi1_r = q_rr + 5 * np.pi / 6
            phi4_l = q_lf + np.pi / 6
            phi4_r = q_rf + np.pi / 6
            L0_l, phi0_l, phi2_l, phi3_l, sigma1_l = fk_fivebar(phi1_l, phi4_l)
            L0_r, phi0_r, phi2_r, phi3_r, sigma1_r = fk_fivebar(phi1_r, phi4_r)

            qw, qx, qy, qz = data.qpos[3:7]
            pitch = np.arcsin(np.clip(2.0*(qw*qy - qx*qz), -1.0, 1.0))
            pitch_dot = data.qvel[4]
            base_vx = data.qvel[0]

            tau = lqr_control_step(
                q, qvel, L0_l, L0_r, phi0_l, phi0_r,
                phi2_l, phi3_l, sigma1_l, phi2_r, phi3_r, sigma1_r,
                pitch, pitch_dot, base_vx, dt, lqr_state, prev_lqr,
                cmd_vel[0], step_count[0],
            )
            raw_action = np.array(tau)

        # ── 力矩 → 执行 ──
        for i in range(6):
            data.ctrl[act_ids[i]] = tau[i]

        last_actions = raw_action.copy()

        # ── 定期打印状态 ──
        if step_count[0] % 100 == 1:
            qw, qx, qy, qz = data.qpos[3:7]
            pitch = np.arcsin(np.clip(2.0*(qw*qy - qx*qz), -1.0, 1.0))
            roll = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
            q_lf, q_rf, q_lr, q_rr = q[0], q[1], q[2], q[3]
            phi1_l = q_lr + 5*np.pi/6; phi4_l = q_lf + np.pi/6
            phi1_r = q_rr + 5*np.pi/6; phi4_r = q_rf + np.pi/6
            L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
            L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)
            print(f"  [{step_count[0]:5d}] "
                  f"pitch={np.rad2deg(pitch):+6.2f}° roll={np.rad2deg(roll):+6.2f}° "
                  f"L0=({L0_l:.3f},{L0_r:.3f})m "
                  f"φ₀=({np.rad2deg(phi0_l):.1f},{np.rad2deg(phi0_r):.1f})° "
                  f"z={data.qpos[2]:.3f}m "
                  f"q=[{q[0]:+.2f},{q[1]:+.2f},{q[2]:+.2f},{q[3]:+.2f}] "
                  f"nn=[{raw_action[0]:+.3f},{raw_action[1]:+.3f},{raw_action[2]:+.3f},{raw_action[3]:+.3f},{raw_action[4]:+.3f},{raw_action[5]:+.3f}] "
                  f"τ=[{tau[0]:+.2f},{tau[1]:+.2f},{tau[2]:+.2f},{tau[3]:+.2f},{tau[4]:+.2f},{tau[5]:+.2f}]")

    def key_callback(keycode):
        step = 0.05
        if keycode == 32:       # Space
            paused[0] = not paused[0]
            print(f"  {'⏸ 暂停' if paused[0] else '▶ 继续'}")
        elif keycode == 265:    # ↑
            cmd_vel[0] += step
            print(f"  cmd_vel={cmd_vel[0]:.2f}, cmd_yaw={cmd_yaw_rate[0]:.2f}")
        elif keycode == 264:    # ↓
            cmd_vel[0] -= step
            print(f"  cmd_vel={cmd_vel[0]:.2f}, cmd_yaw={cmd_yaw_rate[0]:.2f}")
        elif keycode == 263:    # ←
            cmd_yaw_rate[0] += 0.2
            print(f"  cmd_vel={cmd_vel[0]:.2f}, cmd_yaw={cmd_yaw_rate[0]:.2f}")
        elif keycode == 262:    # →
            cmd_yaw_rate[0] -= 0.2
            print(f"  cmd_vel={cmd_vel[0]:.2f}, cmd_yaw={cmd_yaw_rate[0]:.2f}")
        elif keycode == 72:     # H
            cmd_vel[0] = 0.0
            cmd_yaw_rate[0] = 0.0
        elif keycode == 78:     # N
            use_nn[0] = True
            print("  → 神经网络模式")
        elif keycode == 76:     # L
            use_nn[0] = False
            print("  → LQR 模式")

    # ── 运行 ──
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if not paused[0]:
                controller(mj_model, mj_data)
                mujoco.mj_step(mj_model, mj_data)
            viewer.sync()


if __name__ == "__main__":
    main()