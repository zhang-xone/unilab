"""VMC LQR 测试：验证 BalanceInfantry 物理模型在地形上能否维持高度。

直接用 VMC 空间 PD 控制 (L0, φ₀)，不经过 RL policy。
如果 VMC 能维持高度，问题在 RL 训练；如果 VMC 也不行，问题在物理模型。

用法:
    python test_vmc_lqr.py                        # 默认 rough 地形
    python test_vmc_lqr.py --scene flat           # 平地
    python test_vmc_lqr.py --scene rough          # 粗糙地形 (hfield)
    python test_vmc_lqr.py --scene step           # 台阶地形
    python test_vmc_lqr.py --scene flat_slope     # 平地+坡
    python test_vmc_lqr.py --scene rough_slope    # 粗糙+坡
    python test_vmc_lqr.py --scene crawl_slope    # 爬坡场景
    python test_vmc_lqr.py --scene rmuc           # RMUC 综合场景
    python test_vmc_lqr.py --scene flat --collect # 运行并收集监督学习数据
    python test_vmc_lqr.py --scene flat --collect --output my_data.npz  # 指定输出

键盘控制:
    W/S         - 增加/减少目标腿长 L0
    A/D         - 增加/减少目标腿角 φ₀
    Q/E         - 减少/增加前馈力
    R/F         - 减少/增加 kp_leg
    T/G         - 减少/增加 kp_angle
    Space       - 切换暂停
    Enter       - 保存收集的数据 (--collect 时)
    Backspace   - 重置 (物理+控制器状态同步清零)
    ESC         - 退出
"""

import argparse
import math
import os
import sys
import time
import numpy as np

import mujoco
import mujoco.viewer

# ═══════════════════════════════════════════════════════════════
# PD 参数 (与 base.py ControlConfig 一致)
# ═══════════════════════════════════════════════════════════════
LEG_KP = (100., 100., 100., 100.)
LEG_KD = (5., 5., 5., 5.)
WHEEL_KD = (2.0, 2.0)

# ── 动作空间参数 (与 mujoco.yaml 一致) ──
ACTION_SCALE_JOINT = 0.5
ACTION_SCALE_WHEEL = 10.0

# ── 关节限位 (与 base.py JOINT_LIMITS 一致) ──
JOINT_LIMITS = np.array([
    [-1.57, 2.40],
    [-1.57, 2.40],
    [-1.57, 1.57],
    [-1.57, 1.57],
    [-np.inf, np.inf],
    [-np.inf, np.inf],
], dtype=np.float64)

# ── 路径设置 ──
ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets", "robots", "balance_infantry")

SCENE_MAP = {
    "flat":        "scene_flat.xml",
    "rough":       "scene_rough.xml",
    "step":        "scene_step.xml",
    "flat_slope":  "scene_flat_slope.xml",
    "rough_slope": "scene_rough_slope.xml",
    "crawl_slope": "scene_crawl_slope.xml",
    "rmuc":        "scene_rmuc.xml",
}

# ── 五连杆参数 ──
L1 = 0.215
L2 = 0.254
L3 = 0.254
L4 = 0.215
L5 = 0.0


def fk_fivebar(phi1, phi4):
    """五连杆正运动学 → (L0, φ₀, φ₂, φ₃, sin(φ₃-φ₂))."""
    x_B = L1 * np.cos(phi1)
    y_B = L1 * np.sin(phi1)
    x_D = L5 + L4 * np.cos(phi4)
    y_D = L4 * np.sin(phi4)

    dx = x_D - x_B
    dy = y_D - y_B
    BD_sq = dx * dx + dy * dy

    A0 = 2.0 * L2 * dx
    b0 = 2.0 * L2 * dy
    C0 = L2 * L2 + BD_sq - L3 * L3

    disc = max(A0 * A0 + b0 * b0 - C0 * C0, 0.0)
    temp = b0 + np.sqrt(disc)
    phi2 = 2.0 * np.arctan2(temp, A0 + C0)

    x_C = x_B + L2 * np.cos(phi2)
    y_C = y_B + L2 * np.sin(phi2)
    x_C_shifted = x_C - L5 / 2.0

    L0 = np.sqrt(x_C_shifted ** 2 + y_C ** 2)
    phi0 = np.arctan2(y_C, x_C_shifted)

    phi3 = np.arctan2(y_C - y_D, x_C - x_D)
    sigma1 = np.sin(phi3 - phi2)

    return L0, phi0, phi2, phi3, sigma1


def compute_jacobian_partials(phi1, phi4, phi0, phi2, phi3, sigma1, L0):
    """五连杆雅可比偏导数 (参考 debug_vmc_fk.py 虚功验证的正确映射).

    J 矩阵: [L₀_dot, φ₀_dot]ᵀ = J · [φ₁_dot, φ₄_dot]ᵀ
         J = [[dL₀/dφ₁, dL₀/dφ₄],
              [dφ₀/dφ₁, dφ₀/dφ₄]]

    Returns:
        J00: dL₀/dφ₁
        J01: dL₀/dφ₄
        J10: dφ₀/dφ₁
        J11: dφ₀/dφ₄
    """
    sigma2 = np.sin(phi3 - phi4)
    sigma3 = np.sin(phi1 - phi2)

    eps = 1e-8
    inv_sigma1 = 1.0 / (sigma1 + eps)
    inv_L0 = 1.0 / (L0 + eps)

    J00 = L1 * np.sin(phi0 - phi3) * sigma3 * inv_sigma1   # dL₀/dφ₁
    J01 = L4 * np.sin(phi0 - phi2) * sigma2 * inv_sigma1   # dL₀/dφ₄
    J10 = L1 * np.cos(phi0 - phi3) * sigma3 * inv_L0 * inv_sigma1  # dφ₀/dφ₁
    J11 = L4 * np.cos(phi0 - phi2) * sigma2 * inv_L0 * inv_sigma1  # dφ₀/dφ₄

    return J00, J01, J10, J11


# ═══════════════════════════════════════════════════════════════
# 数据收集: LQR 力矩 → RL 原始动作 (与 collect_lqr_data.py 一致)
# ═══════════════════════════════════════════════════════════════

def inverse_pd_targets(tau_lf, tau_rf, tau_lr, tau_rr,
                       tau_lw, tau_rw,
                       q_lf, q_rf, q_lr, q_rr,
                       qvel_lf, qvel_rf, qvel_lr, qvel_rr,
                       qvel_lw, qvel_rw,
                       default_dof_pos):
    """LQR 力矩 → 等效 RL 原始动作 (反解 PD 控制器).

    腿关节: absolute_target = q + (torque + kd * qvel) / kp
    RL 解码: raw_action = (absolute_target - default_dof_pos) / scale
    轮子:   target_vel = qvel + torque / kd
    RL 解码: raw_action = target_vel / action_scale_wheel

    Returns: (lf, rf, lr, rr, lw, rw) 范围 ≈[-1, 1]
    """
    lf_target = q_lf + (tau_lf + LEG_KD[0] * qvel_lf) / LEG_KP[0]
    rf_target = q_rf + (tau_rf + LEG_KD[1] * qvel_rf) / LEG_KP[1]
    lr_target = q_lr + (tau_lr + LEG_KD[2] * qvel_lr) / LEG_KP[2]
    rr_target = q_rr + (tau_rr + LEG_KD[3] * qvel_rr) / LEG_KP[3]
    lw_target = qvel_lw + tau_lw / WHEEL_KD[0]
    rw_target = qvel_rw + tau_rw / WHEEL_KD[1]

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


def build_observation(model, data, joint_qpos, dt, last_actions, default_dof_pos,
                      cmd_vel, cmd_yaw_rate, cmd_height):
    """从 MuJoCo 状态构建 33 维观测 (与 joystick.py _compute_obs 一致)."""
    q_lf = data.qpos[joint_qpos[0]]
    q_rf = data.qpos[joint_qpos[1]]
    q_lr = data.qpos[joint_qpos[2]]
    q_rr = data.qpos[joint_qpos[3]]
    q_lw = data.qpos[joint_qpos[4]]
    q_rw = data.qpos[joint_qpos[5]]
    dof_pos = np.array([q_lf, q_rf, q_lr, q_rr, q_lw, q_rw])

    qvel_lf = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_front_joint")]]
    qvel_rf = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_front_joint")]]
    qvel_lr = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_rear_joint")]]
    qvel_rr = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_rear_joint")]]
    qvel_lw = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")]]
    qvel_rw = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")]]
    dof_vel = np.array([qvel_lf, qvel_rf, qvel_lr, qvel_rr, qvel_lw, qvel_rw])

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

    commands = np.array([cmd_vel, cmd_yaw_rate, cmd_height, 0.0])

    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    yaw_error = 0.0

    phi1_l = q_lr + 5 * np.pi / 6
    phi1_r = q_rr + 5 * np.pi / 6
    phi4_l = q_lf + np.pi / 6
    phi4_r = q_rf + np.pi / 6
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
        np.array([0.0]),
        np.array([0.0]),
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


# ═══════════════════════════════════════════════════════════════
# LQR 平衡控制 (参考 ChassisControl.c)
# ═══════════════════════════════════════════════════════════════

# LQR 增益三次多项式系数: K_coef[i*6 + j][3..0] = [K3, K2, K1, K0]
# K[i][j] = K3*L³ + K2*L² + K1*L + K0
#    i=0 → virtual_torque[0] = 轮力矩 (wheel torque)
#    i=1 → virtual_torque[1] = 摆角力矩 (swing torque)
#    j=0..5 → 状态 [theta, theta_dot, x, vx, pitch, pitch_dot]
K_COEF_ON_GROUND = np.array([
    [-219.3719, 276.6667, -201.4933, -12.4979],
    [33.5412, -32.3856, -18.1929, -0.8847],
    [-10.1891, 20.4230, -12.0295, -3.7717],
    [22.1729, 8.3959, -18.9790, -10.0311],
    [-91.4900, 150.5469, -97.7896, 32.8441],
    [-36.8356, 36.6425, -15.0680, 4.1687],
    [460.9924, -336.3914, 36.9187, 50.8427],
    [42.4265, -48.0207, 21.5147, 3.8196],
    [55.8054, -13.0013, -20.6057, 11.3973],
    [49.8092, 33.4996, -64.1138, 28.2976],
    [697.7343, -728.0338, 284.9198, 19.5743],
    [68.6418, -73.1694, 29.9532, -0.4523],
], dtype=np.float64)


def compute_lqr_gains(L0):
    """根据当前腿长 L0 插值 LQR 增益矩阵 K[2][6].

    Returns:
        K: shape (2, 6), K[0] = 轮力矩增益, K[1] = 摆角力矩增益
    """
    L = max(L0, 0.01)
    L2 = L * L
    L3 = L2 * L
    K = np.zeros((2, 6), dtype=np.float64)
    for i in range(2):
        for j in range(6):
            c = K_COEF_ON_GROUND[i * 6 + j]
            K[i, j] = c[0] * L3 + c[1] * L2 + c[2] * L + c[3]
    return K


def lqr_virtual_torque(K, state, target=None):
    """LQR 虚拟力矩: virtual_torque = K · (target - state).

    Args:
        K: shape (2, 6), LQR 增益矩阵
        state: shape (6,), 当前状态 [theta, theta_dot, x, vx, pitch, pitch_dot]
        target: shape (6,), 目标状态, 默认全零

    Returns:
        wheel_torque, swing_torque (标量)
    """
    if target is None:
        target = np.zeros(6, dtype=np.float64)
    err = state - target
    wheel_torque = np.dot(K[0], err)
    swing_torque = np.dot(K[1], err)
    return wheel_torque, swing_torque


def get_terrain_height(model, data, x, y):
    """用射线检测获取 (x, y) 处的地形高度。"""
    # 从高处 (x, y, 2.0) 向下发射射线
    ray_start = np.array([x, y, 2.0], dtype=np.float64)
    ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # 使用 mujoco.mj_ray
    geomid = np.array([-1], dtype=np.int32)
    distance = np.array([0.0], dtype=np.float64)

    try:
        mujoco.mj_ray(model, data, ray_start, ray_dir, None, 1, -1, geomid, distance)
        if distance[0] > 0:
            return 2.0 - distance[0]
    except Exception:
        pass

    return 0.0


def state_scan(lqr_state_idx=-1):
    """状态扫描：遍历 φ₁, φ₄, pitch, 验证 LQR state 和 virtual_torque。

    Args:
        lqr_state_idx: -1=逐状态全部扫描, 0..5=只测试单个状态分量
    """
    LQR_STATE_NAMES = ["theta", "theta_dot", "x", "vx", "pitch", "pitch_dot"]
    phi1_vals = np.deg2rad([120, 150, 180])
    phi4_vals = np.deg2rad([0, 30, 60])
    pitch_vals = np.deg2rad([-10, 0, 10])

    state_indices = range(6) if lqr_state_idx < 0 else [lqr_state_idx]

    for idx in state_indices:
        mask = np.zeros(6, dtype=np.float64)
        mask[idx] = 1.0

        print("=" * 100)
        print(f"LQR 单状态扫描: [{idx}] {LQR_STATE_NAMES[idx]}")
        print("φ₁(°)  φ₄(°)  pitch(°)  state_val  K_wheel[{0}]  wheel_t  K_swing[{0}]  swing_t".format(idx))
        print("=" * 100)

        for phi1 in phi1_vals:
            for phi4 in phi4_vals:
                L0, phi0, phi2, phi3, sigma1 = fk_fivebar(phi1, phi4)
                if L0 < 0.15 or L0 > 0.5:
                    continue
                J00, J01, J10, J11 = compute_jacobian_partials(
                    phi1, phi4, phi0, phi2, phi3, sigma1, L0)

                for pitch in pitch_vals:
                    theta = np.pi / 2 - phi0 - pitch
                    theta_dot = 1.0 if idx == 1 else 0.0
                    x_pos = 1.0 if idx == 2 else 0.0
                    x_vel = 1.0 if idx == 3 else 0.0
                    pitch_dot = 1.0 if idx == 5 else 0.0
                    state = np.array([theta, theta_dot, x_pos, x_vel, pitch, pitch_dot])
                    K = compute_lqr_gains(L0)

                    K_masked = K * mask[np.newaxis, :]
                    wheel_t, swing_t = lqr_virtual_torque(K_masked, state)
                    state_val = state[idx]
                    print(f"{np.rad2deg(phi1):6.1f}  {np.rad2deg(phi4):6.1f}  "
                          f"{np.rad2deg(pitch):7.1f}  {state_val:9.4f}  "
                          f"{K[0, idx]:8.3f}  {wheel_t:8.3f}  "
                          f"{K[1, idx]:8.3f}  {swing_t:8.3f}")

        print()
        print(f"【单状态 [{idx}] {LQR_STATE_NAMES[idx]}】")
        print(f"  state_val = state[{idx}]")
        print(f"  wheel_t = K[0][{idx}] * (state_val - 0) = K[0][{idx}] * state_val")
        print(f"  swing_t = K[1][{idx}] * (state_val - 0) = K[1][{idx}] * state_val")
        print()


def main():
    parser = argparse.ArgumentParser(description="VMC LQR 地形测试")
    parser.add_argument("--scene", type=str, default="rough", choices=list(SCENE_MAP.keys()),
                        help="地形场景 (default: rough)")
    parser.add_argument("--lqr-state", type=int, default=-1, choices=range(-1, 6),
                        help="单状态测试: -1=全部, 0=theta, 1=theta_dot, 2=x, 3=vx, 4=pitch, 5=pitch_dot")
    parser.add_argument("--scan", action="store_true", help="状态扫描，验证 LQR state 和 virtual_torque")
    parser.add_argument("--no_viewer", action="store_true", help="无渲染模式，只打印数据")
    parser.add_argument("--collect", action="store_true", help="收集监督学习数据 (obs, pd_target)")
    parser.add_argument("--output", type=str, default="lqr_collected.npz", help="数据输出路径 (--collect 时有效)")
    args = parser.parse_args()

    if args.scan:
        state_scan(args.lqr_state)
        return

    scene_file = SCENE_MAP[args.scene]
    scene_path = os.path.join(ASSET_DIR, scene_file)

    print(f"加载场景: {scene_path} ({args.scene})")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # ── 找 actuator ──
    act_names = [
        "left_front_joint_actuator",
        "right_front_joint_actuator",
        "left_rear_joint_actuator",
        "right_rear_joint_actuator",
        "left_wheel_joint_actuator",
        "right_wheel_joint_actuator",
    ]
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]

    # ── 找关节 qpos 地址 ──
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

    # ── 找躯干 body id ──
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    # ── 初始姿态：加载 keyframe 获得合理关节角度，再抬高到空中 ──
    try:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    except Exception:
        pass
    init_z = 0.35  # 初始离地高度 (m)，粗糙地形需要更高
    data.qpos[2] = init_z
    data.qvel[0:6] = 0.0  # 清零初始速度

    # ── 获取 default_dof_pos (home keyframe 的腿关节角度) ──
    q_lf0 = data.qpos[joint_qpos[0]]
    q_rf0 = data.qpos[joint_qpos[1]]
    q_lr0 = data.qpos[joint_qpos[2]]
    q_rr0 = data.qpos[joint_qpos[3]]
    default_dof_pos = np.array([q_lf0, q_rf0, q_lr0, q_rr0], dtype=np.float32)

    # ── 数据收集 ──
    if args.collect:
        collected_obs = []
        collected_targets = []
        collected_torques = []
        last_actions = [np.zeros(6, dtype=np.float32)]  # 用列表包装, 闭包内可修改
        print(f"  [数据收集] 启用, 输出: {args.output}")

        def save_data():
            """保存已收集的数据到 .npz"""
            if len(collected_obs) == 0:
                print("  [数据收集] 无数据, 跳过保存")
                return
            obs_arr = np.stack(collected_obs, axis=0)
            tgt_arr = np.stack(collected_targets, axis=0)
            tor_arr = np.stack(collected_torques, axis=0)
            output_path = os.path.join(os.path.dirname(__file__), args.output) if not os.path.isabs(args.output) else args.output
            np.savez_compressed(output_path,
                               observations=obs_arr,
                               pd_targets=tgt_arr,
                               lqr_torques=tor_arr)
            print(f"  [数据收集] ✔ 保存 {len(obs_arr)} 条样本 → {output_path}")
            print(f"    观测: {obs_arr.shape}, 动作: {tgt_arr.shape}, 力矩: {tor_arr.shape}")
    else:
        save_data = lambda: None

    # 改写 model.qpos0，让 MuJoCo 默认 Backspace reset 也抬高到空中
    model.qpos0 = data.qpos.copy()

    # ── VMC 可变参数（用列表包装以便在回调中修改）─
    LQR_STATE_NAMES = ["theta", "theta_dot", "x", "vx", "pitch", "pitch_dot"]
    params = {
        "L0_target": 0.35,       # 目标腿长 (m)
        "phi0_target": np.pi/2,  # 目标腿角 (rad, 90°=竖直向下)
        "kp_leg": 1000.0,         # 腿长 P 增益
        "kd_leg": 100.0,          # 腿长 D 增益
        "kp_angle": 50.0,        # 腿角 P 增益 (LQR 模式下备用)
        "feedforward": 80.0,     # 重力补偿前馈力 (N)
        "kp_roll": 500.0,        # Roll 轴 P 增益 (通过腿长差纠正)
        "kd_roll": 20.0,         # Roll 轴 D 增益
        "kp_sync": 30.0,         # 两腿同步 P 增益 (phi0 差纠正)
        "kd_sync": 5.0,          # 两腿同步 D 增益
        "paused": False,
        "use_lqr": True,         # True=LQR平衡, False=PD角度
        "lqr_state_idx": args.lqr_state,  # -1=全部, 0..5=单状态测试
        "cmd_vel": 0.0,          # 目标速度 (m/s, 正=前)
        "cmd_yaw_rate": 0.0,     # 目标偏航角速度 (rad/s, 按键累积)
        "target_yaw": 0.0,       # 目标偏航角 (累积)
        "kp_yaw": 10.0,           # 偏航 P 增益
        "kd_yaw": 1.0,           # 偏航 D 增益
    }

    # 前一步 L0 / phi0 用于速度估计
    prev_L0 = {"L": 0.0, "R": 0.0}
    prev_phi0 = {"L": 0.0, "R": 0.0}
    prev_roll = [0.0]
    prev_phi0_diff = [0.0]
    current_yaw = [0.0]
    step_count = [0]
    dt = model.opt.timestep

    # ── LQR 状态 (左右独立) ──
    # state: [theta, theta_dot, x, vx, pitch, pitch_dot]
    lqr_state = {"L": np.zeros(6, dtype=np.float64),
                 "R": np.zeros(6, dtype=np.float64)}
    lqr_target = np.zeros(6, dtype=np.float64)
    lqr_target[0] = np.deg2rad(1.0)  # target theta = 1° (腿微微前倾)

    print(f"  timestep={dt:.4f}s, L0_target={params['L0_target']}m, "
          f"phi0_target={np.rad2deg(params['phi0_target']):.0f}°, ff={params['feedforward']}N")
    if params['lqr_state_idx'] >= 0:
        print(f"  LQR={'ON' if params['use_lqr'] else 'OFF'} "
              f"| 单状态: [{params['lqr_state_idx']}] {LQR_STATE_NAMES[params['lqr_state_idx']]} "
              f"(0-5=单选, -=全开)")
    else:
        print(f"  LQR={'ON' if params['use_lqr'] else 'OFF'} "
              f"(Z=切换LQR/PD)")
    print(f"\n  VMC + LQR 控制中，等待 MuJoCo viewer 窗口...")
    print(f"  W/S ±L0  A/D ±φ₀  Q/E -/+ff  Z=切换LQR/PD  Space=暂停  Esc=退出")
    print(f"  ↑↓=前进/后退  ←→=左/右转  H=停止  1-6=选状态  0=全开\n")

    def controller(model, data):
        """VMC + LQR 控制器 (参考 ChassisControl.c 链路).

        状态 → LQR 虚拟力/力矩 → Jacobian 映射 → 关节力矩
        """
        if params["paused"]:
            return

        step_count[0] += 1

        # ── 获取关节角度 ──
        q_lf = data.qpos[joint_qpos[0]]
        q_rf = data.qpos[joint_qpos[1]]
        q_lr = data.qpos[joint_qpos[2]]
        q_rr = data.qpos[joint_qpos[3]]

        # ── 转为五连杆电机角度 ──
        phi1_l = q_lr + 5 * np.pi / 6
        phi1_r = q_rr + 5 * np.pi / 6
        phi4_l = q_lf + np.pi / 6
        phi4_r = q_rf + np.pi / 6

        # ── 正运动学 ──
        L0_l, phi0_l, phi2_l, phi3_l, sigma1_l = fk_fivebar(phi1_l, phi4_l)
        L0_r, phi0_r, phi2_r, phi3_r, sigma1_r = fk_fivebar(phi1_r, phi4_r)

        # ── 速度估计 ──
        L0_dot_l = (L0_l - prev_L0["L"]) / dt
        L0_dot_r = (L0_r - prev_L0["R"]) / dt
        prev_L0["L"] = L0_l
        prev_L0["R"] = L0_r

        # ── 机体姿态 (pitch: 前面向上为正, 绕 Y 轴) ──
        qw, qx, qy, qz = data.qpos[3:7]
        pitch = np.arcsin(np.clip(2.0 * (qw * qy - qx * qz), -1.0, 1.0))
        pitch_dot = data.qvel[3 + 1]
        roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        roll_dot = data.qvel[3 + 0]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        yaw_dot = data.qvel[3 + 2]
        current_yaw[0] = yaw
        base_vx = data.qvel[0]

        # ── 电机速度 (MuJoCo joint velocity) ──
        qvel_lr = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_rear_joint")]]
        qvel_lf = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_front_joint")]]
        qvel_rr = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_rear_joint")]]
        qvel_rf = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_front_joint")]]

        # ── 雅可比偏导数 (计算一次，复用于 theta_dot + 力矩映射) ──
        J00_l, J01_l, J10_l, J11_l = compute_jacobian_partials(
            phi1_l, phi4_l, phi0_l, phi2_l, phi3_l, sigma1_l, L0_l)
        J00_r, J01_r, J10_r, J11_r = compute_jacobian_partials(
            phi1_r, phi4_r, phi0_r, phi2_r, phi3_r, sigma1_r, L0_r)

        # ── LQR 状态 (参考 C 代码 getTheta): [theta, theta_dot, x, vx, pitch, pitch_dot] ──
        # theta = PI/2 - phi0 - pitch
        theta_l = np.pi / 2 - phi0_l - pitch
        theta_r = np.pi / 2 - phi0_r - pitch
        # theta_dot = -d(phi0)/dt = -(J10 * ω_rear + J11 * ω_front)
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

        # ── VMC 虚拟力/力矩 ──
        # F: 腿长力 (PD)
        F_l = params["kp_leg"] * (params["L0_target"] - L0_l) - params["kd_leg"] * L0_dot_l + params["feedforward"]
        F_r = params["kp_leg"] * (params["L0_target"] - L0_r) - params["kd_leg"] * L0_dot_r + params["feedforward"]

        if params["use_lqr"]:
            # T: 摆角力矩 (LQR virtual_torque[1])
            K_l = compute_lqr_gains(L0_l)
            K_r = compute_lqr_gains(L0_r)

            # 单状态测试: 只保留第 idx 列，其余清零
            if params["lqr_state_idx"] >= 0:
                mask = np.zeros(6, dtype=np.float64)
                mask[params["lqr_state_idx"]] = 1.0
                K_l = K_l * mask[np.newaxis, :]
                K_r = K_r * mask[np.newaxis, :]

            # 速度过大时目标 x 等于当前 x，防止积分器发散
            target_x = lqr_state["L"][2] if abs(base_vx) >= 0.5 else 0.0
            lqr_target[2] = target_x
            lqr_target[3] = -params["cmd_vel"]  # 状态 vx 负=前，目标也负=前

            wheel_torque_l, T_lqr_l = lqr_virtual_torque(K_l, lqr_state["L"], lqr_target)
            wheel_torque_r, T_lqr_r = lqr_virtual_torque(K_r, lqr_state["R"], lqr_target)

            # 启动斜坡: 前 300 步 LQR 力矩从 0 线性 ramp 到 1 (参考 C 代码 stand_done_flag)
            lqr_ramp = min(1.0, step_count[0] / 300.0)

            T_l = T_lqr_l * lqr_ramp
            T_r = T_lqr_r * lqr_ramp
            wheel_torque_l = wheel_torque_l * lqr_ramp
            wheel_torque_r = wheel_torque_r * lqr_ramp

            T_l = np.clip(T_l, -20.0, 20.0)
            T_r = np.clip(T_r, -20.0, 20.0)
            wheel_torque_l = np.clip(wheel_torque_l, -10.0, 10.0)
            wheel_torque_r = np.clip(wheel_torque_r, -10.0, 10.0)
        else:
            T_l = params["kp_angle"] * (params["phi0_target"] - phi0_l)
            T_r = params["kp_angle"] * (params["phi0_target"] - phi0_r)
            wheel_torque_l = 0.0
            wheel_torque_r = 0.0

        # ── Roll 轴 PD: 通过左右腿长差纠正 roll ──
        # roll > 0 右倾 → 右腿多发力、左腿少发力
        roll_correction = params["kp_roll"] * roll + params["kd_roll"] * roll_dot
        F_l -= roll_correction
        F_r += roll_correction

        # ── 两腿同步 PD: 保持 phi0_l ≈ phi0_r ──
        phi0_diff = phi0_l - phi0_r
        phi0_diff_dot = (phi0_diff - prev_phi0_diff[0]) / dt
        prev_phi0_diff[0] = phi0_diff
        sync_torque = params["kp_sync"] * phi0_diff + params["kd_sync"] * phi0_diff_dot
        T_l -= sync_torque
        T_r += sync_torque

        # ── 转向 PD: 目标 yaw 角 PD 控制 ──
        params["cmd_yaw_rate"] = np.clip(params["cmd_yaw_rate"], -3.0, 3.0)
        params["target_yaw"] += params["cmd_yaw_rate"] * dt
        # 跳变包络: target 包到 [-π, π], 再算差
        tw = params["target_yaw"]
        while tw > np.pi:
            tw -= 2 * np.pi
        while tw < -np.pi:
            tw += 2 * np.pi
        yaw_err = tw - yaw
        while yaw_err > np.pi:
            yaw_err -= 2 * np.pi
        while yaw_err < -np.pi:
            yaw_err += 2 * np.pi
        yaw_err = np.clip(yaw_err, -1.0, 1.0)  # 限幅防止力矩过大
        if abs(yaw_err) < 0.02 and abs(params["cmd_yaw_rate"]) < 0.01:
            yaw_torque = 0.0
        else:
            yaw_torque = params["kp_yaw"] * yaw_err - params["kd_yaw"] * yaw_dot
        wheel_torque_l -= yaw_torque
        wheel_torque_r += yaw_torque

        # ── Jacobian 映射: τ = J^T · [F, T] (debug_vmc_fk.py 虚功验证通过) ──
        # tau1 = J00*F + J10*T = dL₀/dφ₁ * F + dφ₀/dφ₁ * T  → rear motor
        # tau4 = J01*F + J11*T = dL₀/dφ₄ * F + dφ₀/dφ₄ * T  → front motor
        tau1_l = np.clip(J00_l * F_l + J10_l * T_l, -40.0, 40.0)
        tau4_l = np.clip(J01_l * F_l + J11_l * T_l, -40.0, 40.0)
        tau1_r = np.clip(J00_r * F_r + J10_r * T_r, -40.0, 40.0)
        tau4_r = np.clip(J01_r * F_r + J11_r * T_r, -40.0, 40.0)

        # ── 填入 ctrl ──
        data.ctrl[act_ids[0]] = tau4_l  # LF (front motor)
        data.ctrl[act_ids[1]] = tau4_r  # RF (front motor)
        data.ctrl[act_ids[2]] = tau1_l  # LR (rear motor)
        data.ctrl[act_ids[3]] = tau1_r  # RR (rear motor)
        data.ctrl[act_ids[4]] = wheel_torque_l  # LW
        data.ctrl[act_ids[5]] = wheel_torque_r  # RW

        # ── 数据收集 ──
        if args.collect:
            qvel_lw = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")]]
            qvel_rw = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")]]

            obs = build_observation(model, data, joint_qpos, dt,
                                    last_actions[0].copy(), default_dof_pos,
                                    params["cmd_vel"], params["cmd_yaw_rate"], params["L0_target"])
            pd_target = np.array(inverse_pd_targets(
                tau4_l, tau4_r, tau1_l, tau1_r,
                wheel_torque_l, wheel_torque_r,
                q_lf, q_rf, q_lr, q_rr,
                qvel_lf, qvel_rf, qvel_lr, qvel_rr,
                qvel_lw, qvel_rw,
                default_dof_pos,
            ))
            lqr_torque = np.array([tau4_l, tau4_r, tau1_l, tau1_r, wheel_torque_l, wheel_torque_r])

            collected_obs.append(obs.astype(np.float32))
            collected_targets.append(pd_target.astype(np.float32))
            collected_torques.append(lqr_torque.astype(np.float32))

            last_actions[0] = pd_target.copy()

        # ── 随机扰动 (采集模式下每 150 步推一下，增加数据多样性) ──
        if args.collect and step_count[0] > 300 and step_count[0] % 150 == 0:
            fx = np.random.uniform(-30, 30)
            fy = np.random.uniform(-30, 30)
            fz = np.random.uniform(0, 50)
            tx = np.random.uniform(-3, 3)
            data.xfrc_applied[base_id] = [fx, fy, fz, tx, 0, 0]

        # ── 定期打印 ──
        if step_count[0] % 100 == 0:
            base_x = data.qpos[0]
            base_y = data.qpos[1]
            base_z = data.qpos[2]
            terrain_z = get_terrain_height(model, data, base_x, base_y)
            height_above_ground = base_z - terrain_z

            print(f"[{step_count[0]:5d}] t={data.time:.3f}s "
                  f"base=({base_x:.2f},{base_y:.2f},{base_z:.3f}) vx={base_vx:.3f} "
                  f"terrain={terrain_z:.3f} H_above={height_above_ground:.3f} "
                  f"pitch={np.rad2deg(pitch):.1f}° roll={np.rad2deg(roll):.1f}° "
                  f"yaw=({np.rad2deg(params['target_yaw']):.1f}→{np.rad2deg(yaw):.1f})° err={np.rad2deg(yaw_err):.1f}° "
                  f"L0=({L0_l:.3f},{L0_r:.3f}) "
                  f"φ₀=({np.rad2deg(phi0_l):.1f},{np.rad2deg(phi0_r):.1f})° "
                  f"θ=({np.rad2deg(theta_l):.1f},{np.rad2deg(theta_r):.1f})° "
                  f"φ₁=({np.rad2deg(phi1_l):.1f},{np.rad2deg(phi1_r):.1f})° "
                  f"φ₄=({np.rad2deg(phi4_l):.1f},{np.rad2deg(phi4_r):.1f})° "
                  f"LQR: x=({lqr_state['L'][2]:.3f},{lqr_state['R'][2]:.3f}) vx=({lqr_state['L'][3]:.3f},{lqr_state['R'][3]:.3f}) "
                  f"wT=({wheel_torque_l:5.1f},{wheel_torque_r:5.1f}) sT=({T_lqr_l:5.1f},{T_lqr_r:5.1f}) "
                  f"T=({T_l:5.1f},{T_r:5.1f}) R=({roll_correction:5.1f}) S=({sync_torque:5.1f}) "
                  f"τ=({tau4_l:5.1f},{tau4_r:5.1f},{tau1_l:5.1f},{tau1_r:5.1f})")

    def key_callback(keycode):
        """键盘回调：实时调整 VMC 参数。"""
        step = 0.005  # L0 步长
        step_angle = np.deg2rad(2.0)  # φ₀ 步长
        step_ff = 5.0  # 前馈力步长
        step_kp = 10.0  # 增益步长

        if keycode == 87:   # W: 增加腿长
            params["L0_target"] = min(params["L0_target"] + step, 0.45)
        elif keycode == 83:  # S: 减少腿长
            params["L0_target"] = max(params["L0_target"] - step, 0.10)
        elif keycode == 65:  # A: 减少腿角 (更倾斜)
            params["phi0_target"] = max(params["phi0_target"] - step_angle, np.deg2rad(40))
        elif keycode == 68:  # D: 增加腿角 (更竖直)
            params["phi0_target"] = min(params["phi0_target"] + step_angle, np.deg2rad(140))
        elif keycode == 81:  # Q: 减少前馈力
            params["feedforward"] = max(params["feedforward"] - step_ff, 0.0)
        elif keycode == 69:  # E: 增加前馈力
            params["feedforward"] = min(params["feedforward"] + step_ff, 200.0)
        elif keycode == 82:  # R: 增加腿长
            params["L0_target"] = min(params["L0_target"] + step, 0.45)
        elif keycode == 84:  # T: 减少腿长
            params["L0_target"] = max(params["L0_target"] - step, 0.10)
        elif keycode == 90:  # Z: 切换 LQR/PD
            params["use_lqr"] = not params["use_lqr"]
            print(f">>> LQR={'ON' if params['use_lqr'] else 'OFF'} <<<")
        elif keycode in (49, 50, 51, 52, 53, 54):  # 1-6: 选择单状态
            params["lqr_state_idx"] = keycode - 49
            print(f">>> LQR 单状态: [{params['lqr_state_idx']}] {LQR_STATE_NAMES[params['lqr_state_idx']]} <<<")
        elif keycode in (48, 45):  # 0 或 -: 全开
            params["lqr_state_idx"] = -1
            print(f">>> LQR 全状态 <<<")
        elif keycode == 32:  # Space: 暂停
            params["paused"] = not params["paused"]
            print(f">>> {'暂停' if params['paused'] else '继续'} <<<")
        elif keycode == 259:  # Backspace: 重置
            prev_L0["L"] = 0.0
            prev_L0["R"] = 0.0
            prev_phi0["L"] = 0.0
            prev_phi0["R"] = 0.0
            lqr_state["L"].fill(0.0)
            lqr_state["R"].fill(0.0)
            params["cmd_vel"] = 0.0
            params["cmd_yaw_rate"] = 0.0
            params["target_yaw"] = current_yaw[0]
            step_count[0] = 0
            print(">>> 重置: 物理状态 + 控制器状态已同步清零 <<<")
        elif keycode == 257:  # Enter: 保存数据
            save_data()
        elif keycode == 256:  # ESC: 退出
            return True
        elif keycode == 265:  # Up: 前进 +0.5
            params["cmd_vel"] += 0.5
        elif keycode == 264:  # Down: 后退 -0.5
            params["cmd_vel"] -= 0.5
        elif keycode == 263:  # Left: 左转 +0.5 rad/s
            params["cmd_yaw_rate"] += 0.5
        elif keycode == 262:  # Right: 右转 -0.5 rad/s
            params["cmd_yaw_rate"] -= 0.5
        elif keycode == 72:  # H: 停止
            params["cmd_vel"] = 0.0
            params["cmd_yaw_rate"] = 0.0
            params["target_yaw"] = current_yaw[0]

        # 打印当前参数
        if keycode in (87, 83, 65, 68, 81, 69, 82, 70, 84, 71):
            print(f"  L0={params['L0_target']:.3f}m  φ₀={np.rad2deg(params['phi0_target']):.0f}°  "
                  f"ff={params['feedforward']:.0f}N  kp_leg={params['kp_leg']:.0f}  kp_angle={params['kp_angle']:.0f}")

        return False

    if args.no_viewer:
        # 无渲染模式
        print("无渲染模式运行中... (Ctrl+C 退出)")
        try:
            while True:
                controller(model, data)
                mujoco.mj_step(model, data)
                if step_count[0] >= 5000:
                    break
        except KeyboardInterrupt:
            pass
    else:
        # 带 MuJoCo viewer 渲染
        with mujoco.viewer.launch_passive(
            model, data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=key_callback,
        ) as viewer:
            while viewer.is_running():
                controller(model, data)
                mujoco.mj_step(model, data)
                viewer.cam.lookat[:] = data.qpos[0:3]
                viewer.sync()

    # ── 最终状态 ──
    q_lf = data.qpos[joint_qpos[0]]
    q_rf = data.qpos[joint_qpos[1]]
    q_lr = data.qpos[joint_qpos[2]]
    q_rr = data.qpos[joint_qpos[3]]
    phi1_l = q_lr + 5 * np.pi / 6
    phi1_r = q_rr + 5 * np.pi / 6
    phi4_l = q_lf + np.pi / 6
    phi4_r = q_rf + np.pi / 6
    L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
    L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)

    base_x = data.qpos[0]
    base_y = data.qpos[1]
    base_z = data.qpos[2]
    terrain_z = get_terrain_height(model, data, base_x, base_y)
    qw, qx, qy, qz = data.qpos[3:7]
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    tilt = np.rad2deg(np.arccos(np.clip(up_z, 0.0, 1.0)))

    # ── 保存收集的数据 ──
    save_data()

    print(f"\n{'='*60}")
    print(f"最终状态 ({args.scene} 地形):")
    print(f"  base: ({base_x:.3f}, {base_y:.3f}, {base_z:.3f})m")
    print(f"  terrain_z: {terrain_z:.3f}m, height_above: {base_z - terrain_z:.3f}m")
    print(f"  tilt: {tilt:.1f}°")
    print(f"  L0: ({L0_l:.3f}, {L0_r:.3f})m")
    print(f"  φ₀: ({np.rad2deg(phi0_l):.1f}, {np.rad2deg(phi0_r):.1f})°")
    print(f"  关节: LF={np.rad2deg(q_lf):.1f}° RF={np.rad2deg(q_rf):.1f}° "
          f"LR={np.rad2deg(q_lr):.1f}° RR={np.rad2deg(q_rr):.1f}°")
    print(f"  总步数: {step_count[0]}, 仿真时间: {data.time:.3f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()