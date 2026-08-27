"""LQR 数据收集脚本：运行 LQR 控制器，收集 (obs, pd_target) 监督学习数据。

调用 test_vmc_lqr 中的 LQR 函数，在无渲染模式下运行仿真，
每步记录完整状态和 LQR 等效 PD 目标，保存为 h5 文件。

用法:
    python collect_lqr_data.py                        # 默认 flat 地形, 5000 步
    python collect_lqr_data.py --scene flat --steps 10000 --seed 42
    python collect_lqr_data.py --scene rough --episodes 10 --episode_steps 500
"""

import argparse
import os
import sys
import time
import numpy as np

import mujoco

# ── 从 test_vmc_lqr 导入核心函数 ──
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CUR_DIR)

from test_vmc_lqr import (
    fk_fivebar,
    compute_jacobian_partials,
    compute_lqr_gains,
    lqr_virtual_torque,
    get_terrain_height,
    SCENE_MAP,
    ASSET_DIR,
    L1, L4,
)

# ═══════════════════════════════════════════════════════════════
# PD 参数 (与 base.py ControlConfig 一致)
# ═══════════════════════════════════════════════════════════════
LEG_KP = (100., 100., 100., 100.)  # LF, RF, LR, RR
LEG_KD = (5., 5., 5., 5.)
WHEEL_KD = (2.0, 2.0)              # LW, RW

# ── 动作空间参数 (与 mujoco.yaml 一致) ──
ACTION_SCALE_JOINT = 0.5
ACTION_SCALE_WHEEL = 10.0

# ── 关节限位 (与 base.py JOINT_LIMITS 一致) ──
JOINT_LIMITS = np.array([
    [-1.57, 2.40],     # left_front_joint
    [-1.57, 2.40],     # right_front_joint
    [-1.57, 1.57],     # left_rear_joint
    [-1.57, 1.57],     # right_rear_joint
    [-np.inf, np.inf], # left_wheel_joint
    [-np.inf, np.inf], # right_wheel_joint
], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════
# 数据记录结构
# ═══════════════════════════════════════════════════════════════
# 每条记录包含:
#   obs: 完整观测 (33维, 与 RL policy actor 观测格式一致)
#   pd_target: LQR 等效 RL 原始动作 (6维, 范围 ≈[-1, 1], 与 policy 输出一致)
#   lqr_torque: LQR 原始力矩 [tau_lf, tau_rf, tau_lr, tau_rr, tau_lw, tau_rw]


def inverse_pd_targets(tau_lf, tau_rf, tau_lr, tau_rr,
                       tau_lw, tau_rw,
                       q_lf, q_rf, q_lr, q_rr,
                       qvel_lf, qvel_rf, qvel_lr, qvel_rr,
                       qvel_lw, qvel_rw,
                       default_dof_pos):
    """LQR 力矩 → 等效 RL 原始动作 (反解 PD 控制器).

    腿关节 PD: torque = kp * (target - q) - kd * qvel
    → absolute_target = q + (torque + kd * qvel) / kp

    RL 动作解码 (与 joystick.py _pre_step_motor_control 一致):
      joint_target = raw_action * scale + default_dof_pos
      scale = max(upper - default, default - lower) * action_scale_joint

    → raw_action = (absolute_target - default_dof_pos) / scale

    轮子 PD: torque = wheel_kd * (target_vel - qvel)
    → target_vel = qvel + torque / wheel_kd

    RL 动作解码:
      wheel_vel_target = raw_action * action_scale_wheel
    → raw_action = wheel_vel_target / action_scale_wheel

    Returns:
        (lf_action, rf_action, lr_action, rr_action, lw_action, rw_action) 范围 ≈[-1, 1]
    """
    lf_target = q_lf + (tau_lf + LEG_KD[0] * qvel_lf) / LEG_KP[0]
    rf_target = q_rf + (tau_rf + LEG_KD[1] * qvel_rf) / LEG_KP[1]
    lr_target = q_lr + (tau_lr + LEG_KD[2] * qvel_lr) / LEG_KP[2]
    rr_target = q_rr + (tau_rr + LEG_KD[3] * qvel_rr) / LEG_KP[3]
    lw_target = qvel_lw + tau_lw / WHEEL_KD[0]
    rw_target = qvel_rw + tau_rw / WHEEL_KD[1]

    # ── 转换为 RL 原始动作空间 ──
    def joint_scale(joint_idx, default_pos):
        lo, hi = JOINT_LIMITS[joint_idx]
        return max(hi - default_pos, default_pos - lo) * ACTION_SCALE_JOINT

    lf_action = (lf_target - default_dof_pos[0]) / joint_scale(0, default_dof_pos[0])
    rf_action = (rf_target - default_dof_pos[1]) / joint_scale(1, default_dof_pos[1])
    lr_action = (lr_target - default_dof_pos[2]) / joint_scale(2, default_dof_pos[2])
    rr_action = (rr_target - default_dof_pos[3]) / joint_scale(3, default_dof_pos[3])
    lw_action = lw_target / ACTION_SCALE_WHEEL
    rw_action = rw_target / ACTION_SCALE_WHEEL

    return (lf_action, rf_action, lr_action, rr_action, lw_action, rw_action)


def build_observation(model, data, joint_qpos, dt, last_actions, default_dof_pos, cmd_vel, cmd_yaw_rate, cmd_height):
    """从 MuJoCo 状态构建 RL 策略 actor 观测 (与 joystick.py _compute_obs 完全一致).

    观测维度:
        gyro(3) + pitch(1) + roll(1) + commands[:, [0,2,3,4]](4)
        + last_actions(6) + wheel_pos_err_l(1) + wheel_pos_err_r(1)
        + sin(yaw_error)(1) + cos(yaw_error)(1) + dof_pos_err(4) + dof_vel(6)
        + VMC θ_l(1) + θ_r(1) + L₀_l(1) + L₀_r(1)
        = 33 维
    """
    # ── 关节位置 ──
    q_lf = data.qpos[joint_qpos[0]]
    q_rf = data.qpos[joint_qpos[1]]
    q_lr = data.qpos[joint_qpos[2]]
    q_rr = data.qpos[joint_qpos[3]]
    q_lw = data.qpos[joint_qpos[4]]
    q_rw = data.qpos[joint_qpos[5]]
    dof_pos = np.array([q_lf, q_rf, q_lr, q_rr, q_lw, q_rw])

    # ── 关节速度 ──
    qvel_lf = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_front_joint")]]
    qvel_rf = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_front_joint")]]
    qvel_lr = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_rear_joint")]]
    qvel_rr = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_rear_joint")]]
    qvel_lw = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")]]
    qvel_rw = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")]]
    dof_vel = np.array([qvel_lf, qvel_rf, qvel_lr, qvel_rr, qvel_lw, qvel_rw])

    # ── 机体角速度 (gyro) ──
    gyro = data.qvel[3:6].copy()

    # ── 重力向量 → pitch, roll ──
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

    # ── commands: [vx, yaw_rate, height, jump] ──
    commands = np.array([cmd_vel, cmd_yaw_rate, cmd_height, 0.0])

    # ── yaw_error ──
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    yaw_error = 0.0

    # ── VMC L0, theta ──
    phi1_l = q_lr + 5 * np.pi / 6
    phi1_r = q_rr + 5 * np.pi / 6
    phi4_l = q_lf + np.pi / 6
    phi4_r = q_rf + np.pi / 6
    L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
    L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)
    theta_l = np.pi / 2 - phi0_l - pitch
    theta_r = np.pi / 2 - phi0_r - pitch

    # ── dof_pos_err: dof_pos[:4] - default_dof_pos[:4] ──
    dof_pos_err = dof_pos[:4] - default_dof_pos[:4]

    # ── 拼接 (33维) ──
    obs = np.concatenate([
        gyro,                                          # 0-2:   gyro (3)
        np.array([pitch]),                             # 3:     pitch (1)
        np.array([roll]),                              # 4:     roll (1)
        commands,                                      # 5-8:   cmd (4)
        last_actions,                                  # 9-14:  last_actions (6)
        np.array([0.0]),                               # 15:    wheel_pos_err_l (1)
        np.array([0.0]),                               # 16:    wheel_pos_err_r (1)
        np.array([np.sin(yaw_error)]),                 # 17:    sin(yaw_error) (1)
        np.array([np.cos(yaw_error)]),                 # 18:    cos(yaw_error) (1)
        dof_pos_err,                                   # 19-22: dof_pos_err (4)
        dof_vel,                                       # 23-28: dof_vel (6)
        np.array([theta_l]),                           # 29:    vmc_theta_l (1)
        np.array([theta_r]),                           # 30:    vmc_theta_r (1)
        np.array([L0_l]),                              # 31:    vmc_L0_l (1)
        np.array([L0_r]),                              # 32:    vmc_L0_r (1)
    ])
    return obs.astype(np.float32)


def collect_episode(model, data, joint_qpos, act_ids,
                    max_steps=500, dt=0.002, ctrl_dt=0.02,
                    cmd_vel_range=(-0.5, 0.5), cmd_yaw_range=(-1.0, 1.0),
                    add_perturbation=False, default_dof_pos=None):
    """运行一个 episode 的 LQR 控制，收集数据.

    Returns:
        obs_list: list of (33,) arrays
        target_list: list of (6,) arrays [lf, rf, lr, rr, lw_vel, rw_vel]
        torque_list: list of (6,) arrays [tau_lf, tau_rf, tau_lr, tau_rr, tau_lw, tau_rw]
    """
    ctrl_steps_per_action = int(ctrl_dt / dt)

    # 控制器状态
    prev_L0 = {"L": 0.0, "R": 0.0}
    lqr_state = {"L": np.zeros(6), "R": np.zeros(6)}
    lqr_target = np.zeros(6)
    lqr_target[0] = np.deg2rad(1.0)  # target theta = 1°

    # VMC 参数
    L0_target = 0.35
    kp_leg = 1000.0
    kd_leg = 100.0
    feedforward = 80.0
    kp_roll = 500.0
    kd_roll = 20.0
    kp_sync = 30.0
    kd_sync = 5.0
    kp_yaw = 10.0
    kd_yaw = 1.0

    # 随机命令
    cmd_vel = np.random.uniform(*cmd_vel_range)
    cmd_yaw_rate = np.random.uniform(*cmd_yaw_range)
    cmd_height = 0.29  # 默认站立高度
    target_yaw = 0.0

    # 上一帧 PD 目标 (初始化为 0)
    last_actions = np.zeros(6, dtype=np.float32)

    prev_roll = 0.0
    prev_phi0_diff = 0.0
    step_count = 0

    obs_list = []
    target_list = []
    torque_list = []

    for step in range(max_steps):
        # ── 获取关节角度 ──
        q_lf = data.qpos[joint_qpos[0]]
        q_rf = data.qpos[joint_qpos[1]]
        q_lr = data.qpos[joint_qpos[2]]
        q_rr = data.qpos[joint_qpos[3]]

        # ── 五连杆正运动学 ──
        phi1_l = q_lr + 5 * np.pi / 6
        phi1_r = q_rr + 5 * np.pi / 6
        phi4_l = q_lf + np.pi / 6
        phi4_r = q_rf + np.pi / 6

        L0_l, phi0_l, phi2_l, phi3_l, sigma1_l = fk_fivebar(phi1_l, phi4_l)
        L0_r, phi0_r, phi2_r, phi3_r, sigma1_r = fk_fivebar(phi1_r, phi4_r)

        # ── 速度估计 ──
        L0_dot_l = (L0_l - prev_L0["L"]) / dt
        L0_dot_r = (L0_r - prev_L0["R"]) / dt
        prev_L0["L"] = L0_l
        prev_L0["R"] = L0_r

        # ── 机体姿态 ──
        qw, qx, qy, qz = data.qpos[3:7]
        pitch = np.arcsin(np.clip(2.0 * (qw * qy - qx * qz), -1.0, 1.0))
        pitch_dot = data.qvel[3 + 1]
        roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        roll_dot = data.qvel[3 + 0]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        yaw_dot = data.qvel[3 + 2]
        base_vx = data.qvel[0]

        # ── 电机速度 ──
        qvel_lr = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_rear_joint")]]
        qvel_lf = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_front_joint")]]
        qvel_rr = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_rear_joint")]]
        qvel_rf = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_front_joint")]]
        qvel_lw = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")]]
        qvel_rw = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")]]

        # ── 雅可比 ──
        J00_l, J01_l, J10_l, J11_l = compute_jacobian_partials(
            phi1_l, phi4_l, phi0_l, phi2_l, phi3_l, sigma1_l, L0_l)
        J00_r, J01_r, J10_r, J11_r = compute_jacobian_partials(
            phi1_r, phi4_r, phi0_r, phi2_r, phi3_r, sigma1_r, L0_r)

        # ── LQR 状态 ──
        theta_l = np.pi / 2 - phi0_l - pitch
        theta_r = np.pi / 2 - phi0_r - pitch
        theta_dot_l = -(J10_l * qvel_lr + J11_l * qvel_lf)
        theta_dot_r = -(J10_r * qvel_rr + J11_r * qvel_rf)

        lqr_state["L"][0] = theta_l
        lqr_state["L"][1] = theta_dot_l
        lqr_state["L"][2] -= base_vx * dt
        lqr_state["L"][2] = np.clip(lqr_state["L"][2], -1.0, 1.0)
        lqr_state["L"][3] = -base_vx
        lqr_state["L"][4] = pitch
        lqr_state["L"][5] = pitch_dot

        lqr_state["R"][0] = theta_r
        lqr_state["R"][1] = theta_dot_r
        lqr_state["R"][2] -= base_vx * dt
        lqr_state["R"][2] = np.clip(lqr_state["R"][2], -1.0, 1.0)
        lqr_state["R"][3] = -base_vx
        lqr_state["R"][4] = pitch
        lqr_state["R"][5] = pitch_dot

        # ── VMC 虚拟力 ──
        F_l = kp_leg * (L0_target - L0_l) - kd_leg * L0_dot_l + feedforward
        F_r = kp_leg * (L0_target - L0_r) - kd_leg * L0_dot_r + feedforward

        # ── LQR 虚拟力矩 ──
        K_l = compute_lqr_gains(L0_l)
        K_r = compute_lqr_gains(L0_r)

        target_x = lqr_state["L"][2] if abs(base_vx) >= 0.5 else 0.0
        lqr_target[2] = target_x
        lqr_target[3] = -cmd_vel

        wheel_torque_l, T_lqr_l = lqr_virtual_torque(K_l, lqr_state["L"], lqr_target)
        wheel_torque_r, T_lqr_r = lqr_virtual_torque(K_r, lqr_state["R"], lqr_target)

        lqr_ramp = min(1.0, step_count / 300.0)
        T_l = T_lqr_l * lqr_ramp
        T_r = T_lqr_r * lqr_ramp
        wheel_torque_l = wheel_torque_l * lqr_ramp
        wheel_torque_r = wheel_torque_r * lqr_ramp

        T_l = np.clip(T_l, -20.0, 20.0)
        T_r = np.clip(T_r, -20.0, 20.0)
        wheel_torque_l = np.clip(wheel_torque_l, -10.0, 10.0)
        wheel_torque_r = np.clip(wheel_torque_r, -10.0, 10.0)

        # ── Roll 纠正 ──
        roll_correction = kp_roll * roll + kd_roll * roll_dot
        F_l -= roll_correction
        F_r += roll_correction

        # ── 两腿同步 ──
        phi0_diff = phi0_l - phi0_r
        phi0_diff_dot = (phi0_diff - prev_phi0_diff) / dt
        prev_phi0_diff = phi0_diff
        sync_torque = kp_sync * phi0_diff + kd_sync * phi0_diff_dot
        T_l -= sync_torque
        T_r += sync_torque

        # ── 转向 ──
        target_yaw += cmd_yaw_rate * dt
        tw = target_yaw
        while tw > np.pi:
            tw -= 2 * np.pi
        while tw < -np.pi:
            tw += 2 * np.pi
        yaw_err = tw - yaw
        while yaw_err > np.pi:
            yaw_err -= 2 * np.pi
        while yaw_err < -np.pi:
            yaw_err += 2 * np.pi
        yaw_err = np.clip(yaw_err, -1.0, 1.0)
        if abs(yaw_err) < 0.02 and abs(cmd_yaw_rate) < 0.01:
            yaw_torque = 0.0
        else:
            yaw_torque = kp_yaw * yaw_err - kd_yaw * yaw_dot
        wheel_torque_l -= yaw_torque
        wheel_torque_r += yaw_torque

        # ── Jacobian 映射 → 关节力矩 ──
        tau_lf = np.clip(J01_l * F_l + J11_l * T_l, -40.0, 40.0)
        tau_lr = np.clip(J00_l * F_l + J10_l * T_l, -40.0, 40.0)
        tau_rf = np.clip(J01_r * F_r + J11_r * T_r, -40.0, 40.0)
        tau_rr = np.clip(J00_r * F_r + J10_r * T_r, -40.0, 40.0)

        # ── 记录 (每 ctrl_dt 记录一次) ──
        if step_count % ctrl_steps_per_action == 0:
            obs = build_observation(model, data, joint_qpos, dt,
                                    last_actions, default_dof_pos,
                                    cmd_vel, cmd_yaw_rate, cmd_height)
            pd_target = np.array(inverse_pd_targets(
                tau_lf, tau_rf, tau_lr, tau_rr,
                wheel_torque_l, wheel_torque_r,
                q_lf, q_rf, q_lr, q_rr,
                qvel_lf, qvel_rf, qvel_lr, qvel_rr,
                qvel_lw, qvel_rw,
                default_dof_pos,
            ))
            lqr_torque = np.array([tau_lf, tau_rf, tau_lr, tau_rr, wheel_torque_l, wheel_torque_r])

            obs_list.append(obs.astype(np.float32))
            target_list.append(pd_target.astype(np.float32))
            torque_list.append(lqr_torque.astype(np.float32))

            # 更新 last_actions
            last_actions = pd_target.astype(np.float32)

        # ── 施加力矩 ──
        data.ctrl[act_ids[0]] = tau_lf
        data.ctrl[act_ids[1]] = tau_rf
        data.ctrl[act_ids[2]] = tau_lr
        data.ctrl[act_ids[3]] = tau_rr
        data.ctrl[act_ids[4]] = wheel_torque_l
        data.ctrl[act_ids[5]] = wheel_torque_r

        # ── 随机扰动 (10% 概率, 模拟外力) ──
        if add_perturbation and np.random.random() < 0.002:
            data.qvel[0:3] += np.random.uniform(-0.5, 0.5, 3)
            data.qvel[3:6] += np.random.uniform(-1.0, 1.0, 3)

        mujoco.mj_step(model, data)
        step_count += 1

        # ── 终止条件 ──
        qw, qx, qy, qz = data.qpos[3:7]
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        if up_z < 0.3:  # tilt > 70°, 摔倒
            break
        base_z = data.qpos[2]
        terrain_z = get_terrain_height(model, data, data.qpos[0], data.qpos[1])
        if base_z - terrain_z < 0.05:  # 太低
            break

    return obs_list, target_list, torque_list


def main():
    parser = argparse.ArgumentParser(description="LQR 数据收集")
    parser.add_argument("--scene", type=str, default="flat",
                        choices=list(SCENE_MAP.keys()), help="地形场景")
    parser.add_argument("--episodes", type=int, default=20,
                        help="episode 数量")
    parser.add_argument("--episode_steps", type=int, default=500,
                        help="每 episode 最大步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default="lqr_data.npz",
                        help="输出文件路径")
    parser.add_argument("--perturb", action="store_true",
                        help="添加随机扰动")
    parser.add_argument("--no_perturb", action="store_true",
                        help="不添加随机扰动")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── 加载场景 ──
    scene_file = SCENE_MAP[args.scene]
    scene_path = os.path.join(ASSET_DIR, scene_file)
    print(f"加载场景: {scene_path} ({args.scene})")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # ── actuator IDs ──
    act_names = [
        "left_front_joint_actuator",
        "right_front_joint_actuator",
        "left_rear_joint_actuator",
        "right_rear_joint_actuator",
        "left_wheel_joint_actuator",
        "right_wheel_joint_actuator",
    ]
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]

    # ── 关节 qpos 地址 ──
    joint_names = [
        "left_front_joint",
        "right_front_joint",
        "left_rear_joint",
        "right_rear_joint",
        "left_wheel_joint",
        "right_wheel_joint",
    ]
    joint_qpos = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                  for n in joint_names]

    dt = model.opt.timestep

    # ── 初始姿态 ──
    try:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    except Exception:
        pass
    data.qpos[2] = 0.35
    data.qvel[0:6] = 0.0

    # ── 获取 default_dof_pos (home keyframe 的腿关节角度) ──
    q_lf = data.qpos[joint_qpos[0]]
    q_rf = data.qpos[joint_qpos[1]]
    q_lr = data.qpos[joint_qpos[2]]
    q_rr = data.qpos[joint_qpos[3]]
    default_dof_pos = np.array([q_lf, q_rf, q_lr, q_rr], dtype=np.float32)

    add_perturb = args.perturb if args.perturb else not args.no_perturb

    all_obs = []
    all_targets = []
    all_torques = []

    total_collected = 0
    start_time = time.time()

    for ep in range(args.episodes):
        # 重置
        try:
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        except Exception:
            pass
        data.qpos[2] = 0.35
        data.qvel[0:6] = 0.0

        obs_list, target_list, torque_list = collect_episode(
            model, data, joint_qpos, act_ids,
            max_steps=args.episode_steps, dt=dt,
            add_perturbation=add_perturb,
            default_dof_pos=default_dof_pos,
        )

        all_obs.extend(obs_list)
        all_targets.extend(target_list)
        all_torques.extend(torque_list)
        total_collected += len(obs_list)

        print(f"  Episode {ep+1}/{args.episodes}: {len(obs_list)} steps, "
              f"累计 {total_collected}")

    elapsed = time.time() - start_time

    # ── 保存 ──
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    np.savez_compressed(
        output_path,
        observations=np.stack(all_obs),
        pd_targets=np.stack(all_targets),
        lqr_torques=np.stack(all_torques),
    )

    print(f"\n{'='*60}")
    print(f"数据收集完成:")
    print(f"  场景: {args.scene}")
    print(f"  总样本数: {total_collected}")
    print(f"  观测维度: {all_obs[0].shape[0]}")
    print(f"  动作维度: {all_targets[0].shape[0]}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  输出: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()