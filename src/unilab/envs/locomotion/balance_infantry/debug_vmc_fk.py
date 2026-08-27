#!/usr/bin/env python3
"""调试脚本：悬空调腿，实时打印 joint_pos / VMC φ角 / L₀ / φ₀  +  J 验证"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path

# ============================================================
# 五连杆正运动学 (与 base.py 保持一致)
# ============================================================
L1, L2, L3, L4, L5 = 0.215, 0.254, 0.254, 0.215, 0.0
OFFSET_PHI1 = 5 * np.pi / 6   # 150°
OFFSET_PHI4 = np.pi / 6        # 30°

# 关节顺序: [LF, RF, LR, RR, LW, RW]
# 映射: φ₁=后大腿(LR/RR), φ₄=前大腿(LF/RF)


def forward_kinematics_fivebar(phi1, phi4, debug=False):
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

    if debug:
        print(f"    FK: B=({x_B:.4f},{y_B:.4f}) D=({x_D:.4f},{y_D:.4f}) BD={np.sqrt(BD_sq):.4f}")
        print(f"    A0={A0:.4f} b0={b0:.4f} C0={C0:.4f} disc={disc:.4f}")
        print(f"    phi2={np.rad2deg(phi2):.1f}° C=({x_C:.4f},{y_C:.4f}) shifted=({x_C_shifted:.4f},{y_C:.4f})")

    L0 = np.sqrt(x_C_shifted * x_C_shifted + y_C * y_C)
    phi0 = np.arctan2(y_C, x_C_shifted)

    phi3 = np.arctan2(y_C - y_D, x_C - x_D)
    sigma1 = np.sin(phi3 - phi2)

    return L0, phi0, phi2, phi3, sigma1


def compute_vmc_state(m, d):
    """从 MuJoCo data 计算 VMC 状态"""
    def _qpos(name):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        return d.qpos[m.jnt_qposadr[jid]]

    lf = _qpos("left_front_joint")
    rf = _qpos("right_front_joint")
    lr = _qpos("left_rear_joint")
    rr = _qpos("right_rear_joint")

    phi1_l = lr + OFFSET_PHI1
    phi4_l = lf + OFFSET_PHI4
    L0_l, phi0_l, phi2_l, phi3_l, sigma1_l = forward_kinematics_fivebar(phi1_l, phi4_l)

    phi1_r = rr + OFFSET_PHI1
    phi4_r = rf + OFFSET_PHI4
    L0_r, phi0_r, phi2_r, phi3_r, sigma1_r = forward_kinematics_fivebar(phi1_r, phi4_r)

    return {
        "joint": [lf, rf, lr, rr],
        "phi1": [phi1_l, phi1_r],
        "phi4": [phi4_l, phi4_r],
        "L0": [L0_l, L0_r],
        "phi0": [phi0_l, phi0_r],
        "phi2": [phi2_l, phi2_r],
        "phi3": [phi3_l, phi3_r],
        "sigma1": [sigma1_l, sigma1_r],
    }


def compute_sensor_swing_theta(m, d):
    hip_z = 0.141

    bw_lw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wheel_link")
    bw_rw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wheel_link")
    base_pos = d.qpos[0:3].copy()
    base_quat = d.qpos[3:7].copy()
    R_flat = np.zeros(9)
    mujoco.mju_quat2Mat(R_flat, base_quat)
    R_base = R_flat.reshape(3, 3)

    wheel_world = [d.xpos[bw_lw].copy(), d.xpos[bw_rw].copy()]
    wheel_positions = [R_base.T @ (wheel_world[i] - base_pos) for i in range(2)]

    thetas = []
    for leg in range(2):
        wp = wheel_positions[leg]
        theta = np.arctan2(wp[0], -(wp[2] - hip_z))
        thetas.append(theta)

    return thetas, wheel_positions


def print_status(m, d):
    s = compute_vmc_state(m, d)
    sensor_theta, wheel_pos = compute_sensor_swing_theta(m, d)

    vmc_theta_body = [np.pi / 2 - s["phi0"][0], np.pi / 2 - s["phi0"][1]]

    print(
        f"  joint=[{s['joint'][0]:+.3f}, {s['joint'][1]:+.3f}, "
        f"{s['joint'][2]:+.3f}, {s['joint'][3]:+.3f}]  "
        f"φ₁=[{np.rad2deg(s['phi1'][0]):+.1f}°, {np.rad2deg(s['phi1'][1]):+.1f}°]  "
        f"φ₄=[{np.rad2deg(s['phi4'][0]):+.1f}°, {np.rad2deg(s['phi4'][1]):+.1f}°]  "
        f"L₀=[{s['L0'][0]:.4f}, {s['L0'][1]:.4f}]  "
        f"φ₀=[{np.rad2deg(s['phi0'][0]):+.1f}°, {np.rad2deg(s['phi0'][1]):+.1f}°]"
    )
    print(
        f"  Swing θ: sensor=[{np.rad2deg(sensor_theta[0]):+.1f}°, {np.rad2deg(sensor_theta[1]):+.1f}°]  "
        f"VMC=[{np.rad2deg(vmc_theta_body[0]):+.1f}°, {np.rad2deg(vmc_theta_body[1]):+.1f}°]  "
        f"Δ=[{np.rad2deg(sensor_theta[0] - vmc_theta_body[0]):+.1f}°, "
        f"{np.rad2deg(sensor_theta[1] - vmc_theta_body[1]):+.1f}°]"
    )
    hip_z = 0.141
    hip_y = np.array([+0.187, -0.187])
    sensor_L = []
    for leg in range(2):
        dx = wheel_pos[leg][0] - 0.0
        dz = wheel_pos[leg][2] - hip_z
        sensor_L.append(np.sqrt(dx * dx + dz * dz))
    print(
        f"  leg_len: sensor=[{sensor_L[0]:.4f}, {sensor_L[1]:.4f}]  "
        f"VMC=[{s['L0'][0]:.4f}, {s['L0'][1]:.4f}]  "
        f"Δ=[{sensor_L[0] - s['L0'][0]:+.4f}, {sensor_L[1] - s['L0'][1]:+.4f}]"
    )
    print(
        f"  wheel_base: L=({wheel_pos[0][0]:+.4f},{wheel_pos[0][1]:+.4f},{wheel_pos[0][2]:+.4f})  "
        f"R=({wheel_pos[1][0]:+.4f},{wheel_pos[1][1]:+.4f},{wheel_pos[1][2]:+.4f})"
    )
    for leg in range(2):
        L0_i = s['L0'][leg]
        phi0_i = s['phi0'][leg]
        x_c = L0_i * np.cos(phi0_i)
        z_c = hip_z - L0_i * np.sin(phi0_i)
        sign = "L" if leg == 0 else "R"
        print(
            f"  VMC C_{sign}: (x={x_c:+.4f}, z={z_c:+.4f})  "
            f"wheel: (x={wheel_pos[leg][0]:+.4f}, z={wheel_pos[leg][2]:+.4f})  "
            f"Δx={wheel_pos[leg][0]-x_c:+.4f}  Δz={wheel_pos[leg][2]-z_c:+.4f}"
        )


# ============================================================
# VMC 雅可比验证 (数值验证 vs 解析)
# ============================================================
def verify_vmc_jacobian(m, d):
    """用三种方法验证 VMC 雅可比映射的正确性

    实际 J 矩阵布局 (经过数值验证确认):
        J = [[dφ₀/dφ₁, dL₀/dφ₁],
             [dφ₀/dφ₄, dL₀/dφ₄]]
    映射: [φ₀_dot, L₀_dot]ᵀ = J · [φ₁_dot, φ₄_dot]ᵀ

    虚功原理: τ₁·δφ₁ + τ₄·δφ₄ = F·δL₀ + T·δφ₀
        → τ₁ = (dL₀/dφ₁)·F + (dφ₀/dφ₁)·T = J01·F + J00·T
        → τ₄ = (dL₀/dφ₄)·F + (dφ₀/dφ₄)·T = J11·F + J10·T
    """
    s = compute_vmc_state(m, d)
    leg_names = ["左腿", "右腿"]

    print("\n" + "=" * 70)
    print("VMC 雅可比验证")
    print("=" * 70)

    for leg_idx in range(2):
        phi1 = s["phi1"][leg_idx]
        phi4 = s["phi4"][leg_idx]
        phi0 = s["phi0"][leg_idx]
        phi2 = s["phi2"][leg_idx]
        phi3 = s["phi3"][leg_idx]
        sigma1 = s["sigma1"][leg_idx]
        L0 = s["L0"][leg_idx]

        # --- 1. 解析雅可比 (与 base.py vmc_mapping_fivebar 完全一致) ---
        sigma2 = np.sin(phi3 - phi4)
        sigma3 = np.sin(phi1 - phi2)
        eps = 1e-8
        inv_sigma1 = 1.0 / (sigma1 + eps)
        inv_L0 = 1.0 / (L0 + eps)

        J00 = L1 * np.sin(phi0 - phi3) * sigma3 * inv_sigma1            # dL₀/dφ₁
        J01 = L4 * np.sin(phi0 - phi2) * sigma2 * inv_sigma1            # dL₀/dφ₄
        J10 = L1 * np.cos(phi0 - phi3) * sigma3 * inv_L0 * inv_sigma1  # dφ₀/dφ₁
        J11 = L4 * np.cos(phi0 - phi2) * sigma2 * inv_L0 * inv_sigma1  # dφ₀/dφ₄

        J_analytic = np.array([[J00, J01], [J10, J11]])

        # --- 2. 数值雅可比 (有限差分) ---
        dphi = 1e-6
        L0_p, phi0_p, _, _, _ = forward_kinematics_fivebar(phi1 + dphi, phi4)
        L0_n, phi0_n, _, _, _ = forward_kinematics_fivebar(phi1 - dphi, phi4)

        dphi0_dphi1 = (phi0_p - phi0_n) / (2 * dphi)
        dL0_dphi1 = (L0_p - L0_n) / (2 * dphi)

        L0_p, phi0_p, _, _, _ = forward_kinematics_fivebar(phi1, phi4 + dphi)
        L0_n, phi0_n, _, _, _ = forward_kinematics_fivebar(phi1, phi4 - dphi)

        dphi0_dphi4 = (phi0_p - phi0_n) / (2 * dphi)
        dL0_dphi4 = (L0_p - L0_n) / (2 * dphi)

        J_num = np.array([[dL0_dphi1, dL0_dphi4],
                          [dphi0_dphi1, dphi0_dphi4]])

        # --- 3. 验证虚功原理 ---
        F_test, T_test = 50.0, 10.0

        print(f"\n--- {leg_names[leg_idx]} ---")
        print(f"  φ₁={np.rad2deg(phi1):.1f}°, φ₄={np.rad2deg(phi4):.1f}°, "
              f"φ₀={np.rad2deg(phi0):.1f}°, L₀={L0:.4f}")
        print(f"  σ₁={sigma1:.4f}, σ₂={sigma2:.4f}, σ₃={sigma3:.4f}")

        print(f"\n  解析 J = [[dL₀/dφ₁={J00:.4f}, dL₀/dφ₄={J01:.4f}], "
              f"[dφ₀/dφ₁={J10:.4f}, dφ₀/dφ₄={J11:.4f}]]")
        print(f"  数值 J = [[dL₀/dφ₁={J_num[0,0]:.4f}, dL₀/dφ₄={J_num[0,1]:.4f}], "
              f"[dφ₀/dφ₁={J_num[1,0]:.4f}, dφ₀/dφ₄={J_num[1,1]:.4f}]]")

        J_err = np.max(np.abs(J_num - J_analytic))
        print(f"  J 误差 max|Δ| = {J_err:.2e} {'✓' if J_err < 1e-5 else '✗ 不对!'}")

        # --- 3. 正映射: F, T → τ₁, τ₄ ---
        tau1 = J00 * F_test + J10 * T_test
        tau4 = J01 * F_test + J11 * T_test
        print(f"\n  正映射 (F={F_test:.0f}N, T={T_test:.0f}Nm → τ):")
        print(f"    τ₁ = J00·F + J10·T = {tau1:+.2f}")
        print(f"    τ₄ = J01·F + J11·T = {tau4:+.2f}")

        # --- 4. 虚功原理验证 ---
        dphi_test = 0.001
        L0_b, phi0_b, _, _, _ = forward_kinematics_fivebar(phi1, phi4)
        L0_a, phi0_a, _, _, _ = forward_kinematics_fivebar(
            phi1 + tau1 * dphi_test * 0.001,
            phi4 + tau4 * dphi_test * 0.001)
        dL0 = L0_a - L0_b
        dphi0 = phi0_a - phi0_b
        vw_FT = F_test * dL0 + T_test * dphi0
        vw_tau = tau1 * (tau1 * dphi_test * 0.001) + tau4 * (tau4 * dphi_test * 0.001)
        print(f"  虚功原理: F·dL₀+T·dφ₀={vw_FT:.6f}, "
              f"τ₁·dφ₁+τ₄·dφ₄={vw_tau:.6f} {'✓' if abs(vw_FT - vw_tau) < 1e-5 else '✗'}")

        # --- 5. 逆映射: τ₁, τ₄ → F, T ---
        det = J00 * J11 - J01 * J10
        if abs(det) > 1e-8:
            F_inv = (J11 * tau1 - J10 * tau4) / det
            T_inv = (-J01 * tau1 + J00 * tau4) / det
            print(f"  逆映射 (τ → F,T): "
                  f"F_inv={F_inv:.2f} (输入50), T_inv={T_inv:.2f} (输入10) "
                  f"{'✓' if abs(F_inv - F_test) < 0.1 and abs(T_inv - T_test) < 0.1 else '✗'}")

    print("\n" + "=" * 70)


# ============================================================
# 主程序
# ============================================================
xml = "src/unilab/assets/robots/balance_infantry/scene_flat.xml"
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
m.opt.gravity = np.array([0.0, 0.0, 0.0])
print(f">>> 重力: {m.opt.gravity}")

# 关闭全部碰撞，再单独开后腿
# 先保存原始值
orig_contype = m.geom_contype.copy()
orig_conaffinity = m.geom_conaffinity.copy()
for i in range(m.ngeom):
    m.geom_contype[i] = 0
    m.geom_conaffinity[i] = 0

# 全开碰撞
for i in range(m.ngeom):
    m.geom_contype[i] = orig_contype[i]
    m.geom_conaffinity[i] = orig_conaffinity[i]
print(">>> 碰撞: 全开, base高度=1.0")

print(">>> 操作方式：")
print("   1. Space 暂停，Ctrl+J 打开 Joints 面板拖腿")
print("   2. 终端实时打印 joint_pos / VMC φ角 / L₀ / φ₀")
print("   3. V 键导出 home keyframe 到 scene_flat.xml")
print("   4. J 键运行 VMC 雅可比验证")
print("   5. T 键实测力矩 (正反各10步)")
print("   6. Esc 退出")
print()

paused_flag = {"paused": False}

test_mode = {"active": False, "step": 0, "phase_steps": [20, 40, 20], "F": +50.0, "T": +10.0,
              "phase": 0, "L0_before": None, "phi0_before": None, "dof_map": None}

# PD VMC test mode
pd_test = {
    "active": False,
    "step": 0,
    "total_steps": 500,
    "theta_ref_amp": np.deg2rad(10.0),  # θ_ref 振幅 ±10°
    "theta_ref_freq": 2.0,               # 频率 2Hz
    "L0_ref_center": 0.28,               # L0_ref 中心
    "L0_ref_amp": 0.04,                  # L0_ref 振幅 ±4cm
    "L0_ref_freq": 1.5,                  # 频率 1.5Hz
    "kp_theta": 100.0,
    "kd_theta": 5.0,
    "kp_l0": 1500.0,
    "kd_l0": 80.0,
    "ff": 0.0,
    "last_theta": [0.0, 0.0],
    "last_L0": [0.28, 0.28],
    "theta_hist": [],
    "pitch_hist": [],
    "T_hist": [],
    "theta_ref_hist": [],
    "L0_ref_hist": [],
    "L0_hist": [],
    "dof_map": None,
}

def test_vmc_physics(m, d):
    s = compute_vmc_state(m, d)
    F_test = test_mode["F"]
    T_test = test_mode["T"]

    dof_map = {}
    for name in ["left_front_joint", "right_front_joint", "left_rear_joint", "right_rear_joint"]:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_map[name] = m.jnt_dofadr[jid]

    test_mode["active"] = True
    test_mode["step"] = 0
    test_mode["phase"] = 0
    test_mode["L0_before"] = s["L0"]
    test_mode["phi0_before"] = s["phi0"]
    test_mode["L0_origin"] = s["L0"]
    test_mode["phi0_origin"] = s["phi0"]
    test_mode["dof_map"] = dof_map

    print(f"\n{'='*70}")
    print(f"实测力矩: 正10步 → 反20步 → 正10步")


def apply_phase_torque(m, d, phase):
    """施加当前 phase 的力矩: phase 0 (+), phase 1 (-), phase 2 (+)"""
    s = compute_vmc_state(m, d)
    sign = 1 if phase == 0 or phase == 2 else -1
    F_test = sign * test_mode["F"]
    T_test = sign * test_mode["T"]
    label = ["正①", "反", "正②"][phase]

    sigma2 = [np.sin(s["phi3"][i] - s["phi4"][i]) for i in range(2)]
    sigma3 = [np.sin(s["phi1"][i] - s["phi2"][i]) for i in range(2)]
    eps = 1e-8

    taus = []
    for leg in range(2):
        inv_s1 = 1.0 / (s["sigma1"][leg] + eps)
        inv_L0 = 1.0 / (s["L0"][leg] + eps)
        J00 = L1 * np.sin(s["phi0"][leg] - s["phi3"][leg]) * sigma3[leg] * inv_s1
        J01 = L1 * np.cos(s["phi0"][leg] - s["phi3"][leg]) * sigma3[leg] * inv_L0 * inv_s1
        J10 = L4 * np.sin(s["phi0"][leg] - s["phi2"][leg]) * sigma2[leg] * inv_s1
        J11 = L4 * np.cos(s["phi0"][leg] - s["phi2"][leg]) * sigma2[leg] * inv_L0 * inv_s1
        tau1 = J00 * F_test + J01 * T_test
        tau4 = J10 * F_test + J11 * T_test
        taus.append((tau1, tau4))

    dm = test_mode["dof_map"]
    d.qfrc_applied[dm["left_front_joint"]] = taus[0][1]
    d.qfrc_applied[dm["right_front_joint"]] = taus[1][1]
    d.qfrc_applied[dm["left_rear_joint"]] = taus[0][0]
    d.qfrc_applied[dm["right_rear_joint"]] = taus[1][0]

    print(f"\n  [{label}向] F={F_test:+.0f}N, T={T_test:+.0f}Nm")
    print(f"    τ₁=[{taus[0][0]:+.2f}, {taus[1][0]:+.2f}], τ₄=[{taus[0][1]:+.2f}, {taus[1][1]:+.2f}]")
    print(f"    施加前: L₀=[{s['L0'][0]:.4f}, {s['L0'][1]:.4f}], "
          f"φ₀=[{np.rad2deg(s['phi0'][0]):.1f}°, {np.rad2deg(s['phi0'][1]):.1f}°]")

    test_mode["L0_before"] = s["L0"]
    test_mode["phi0_before"] = s["phi0"]


def apply_pd_vmc(m, d, step):
    """单步 PD VMC 控制（周期性目标）"""
    s = compute_vmc_state(m, d)

    dt = 0.002
    t = step * dt

    theta_ref = pd_test["theta_ref_amp"] * np.sin(2 * np.pi * pd_test["theta_ref_freq"] * t)
    L0_ref = pd_test["L0_ref_center"] + pd_test["L0_ref_amp"] * np.sin(2 * np.pi * pd_test["L0_ref_freq"] * t)

    theta_l = np.pi / 2 - s["phi0"][0]
    theta_r = np.pi / 2 - s["phi0"][1]

    if step == 0:
        pd_test["last_theta"] = [theta_l, theta_r]
        pd_test["last_L0"] = [s["L0"][0], s["L0"][1]]
    theta_dot_l = (theta_l - pd_test["last_theta"][0]) / dt
    theta_dot_r = (theta_r - pd_test["last_theta"][1]) / dt
    L0_dot_l = (s["L0"][0] - pd_test["last_L0"][0]) / dt
    L0_dot_r = (s["L0"][1] - pd_test["last_L0"][1]) / dt
    pd_test["last_theta"] = [theta_l, theta_r]
    pd_test["last_L0"] = [s["L0"][0], s["L0"][1]]

    T_l = pd_test["kp_theta"] * (theta_l - theta_ref) + pd_test["kd_theta"] * theta_dot_l
    T_r = pd_test["kp_theta"] * (theta_r - theta_ref) + pd_test["kd_theta"] * theta_dot_r

    F_l = pd_test["kp_l0"] * (L0_ref - s["L0"][0]) + pd_test["kd_l0"] * (-L0_dot_l) + pd_test["ff"]
    F_r = pd_test["kp_l0"] * (L0_ref - s["L0"][1]) + pd_test["kd_l0"] * (-L0_dot_r) + pd_test["ff"]

    taus = []
    for leg in range(2):
        phi1 = s["phi1"][leg]
        phi4 = s["phi4"][leg]
        phi0 = s["phi0"][leg]
        phi2 = s["phi2"][leg]
        phi3 = s["phi3"][leg]
        sigma1 = s["sigma1"][leg]
        L0 = s["L0"][leg]
        F = [F_l, F_r][leg]
        T = [T_l, T_r][leg]

        sigma2 = np.sin(phi3 - phi4)
        sigma3 = np.sin(phi1 - phi2)
        eps = 1e-8
        inv_sigma1 = 1.0 / (sigma1 + eps)
        inv_L0 = 1.0 / (L0 + eps)

        J00 = L1 * np.sin(phi0 - phi3) * sigma3 * inv_sigma1
        J01 = L1 * np.cos(phi0 - phi3) * sigma3 * inv_L0 * inv_sigma1
        J10 = L4 * np.sin(phi0 - phi2) * sigma2 * inv_sigma1
        J11 = L4 * np.cos(phi0 - phi2) * sigma2 * inv_L0 * inv_sigma1

        tau1 = J00 * F + J01 * T
        tau4 = J10 * F + J11 * T
        taus.append((tau4, tau1))

    dm = pd_test["dof_map"]
    d.qfrc_applied[dm["left_front_joint"]] = taus[0][0]
    d.qfrc_applied[dm["right_front_joint"]] = taus[1][0]
    d.qfrc_applied[dm["left_rear_joint"]] = taus[0][1]
    d.qfrc_applied[dm["right_rear_joint"]] = taus[1][1]

    pitch = np.arcsin(np.clip(-d.qpos[3:7] @ np.array([0, 0, 1, 0]), -1, 1))
    pd_test["theta_hist"].append((theta_l, theta_r))
    pd_test["pitch_hist"].append(pitch)
    pd_test["T_hist"].append((T_l, T_r))
    pd_test["theta_ref_hist"].append(theta_ref)
    pd_test["L0_ref_hist"].append(L0_ref)
    pd_test["L0_hist"].append((s["L0"][0], s["L0"][1]))

    return theta_l, theta_r, theta_ref, L0_ref, T_l, T_r, F_l, F_r, pitch


def add_vmc_visualization(v, m, d):
    """在 viewer 中标注: 髋节点(红) / VMC C点(绿) / 轮子(蓝) / 箭头"""
    s = compute_vmc_state(m, d)

    base_pos = d.qpos[0:3].copy()
    base_quat = d.qpos[3:7].copy()
    R_base_flat = np.zeros(9)
    mujoco.mju_quat2Mat(R_base_flat, base_quat)
    R_base = R_base_flat.reshape(3, 3)

    hip_y = np.array([+0.187, -0.187])
    hip_z = 0.141

    scn = v.user_scn
    scn.ngeom = 0
    idx = 0

    def add_sphere(pos, rgba, radius=0.012):
        nonlocal idx
        if idx >= 200:
            return
        mujoco.mjv_initGeom(
            scn.geoms[idx],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            pos=pos,
            mat=np.eye(3).ravel(),
            rgba=rgba,
        )
        idx += 1

    def add_arrow(fr, to, rgba, width=0.006):
        nonlocal idx
        if idx >= 199:
            return
        mujoco.mjv_initGeom(
            scn.geoms[idx],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[width, 0, 0],
            pos=fr,
            mat=np.eye(3).ravel(),
            rgba=rgba,
        )
        idx += 1
        mujoco.mjv_connector(
            scn.geoms[idx],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            width=width,
            from_=fr.astype(np.float64),
            to=to.astype(np.float64),
        )
        idx += 1

    for leg in range(2):
        hip_local = np.array([0.0, hip_y[leg], hip_z])
        hip_world = base_pos + R_base @ hip_local

        L0_val = s["L0"][leg]
        phi0_val = s["phi0"][leg]
        vmc_c_local = np.array([L0_val * np.cos(phi0_val), hip_y[leg], hip_z - L0_val * np.sin(phi0_val)])
        vmc_c_world = base_pos + R_base @ vmc_c_local

        wheel_name = "left_wheel" if leg == 0 else "right_wheel"
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, wheel_name)
        wheel_world = d.xpos[body_id].copy()

        add_sphere(hip_world, [1, 0, 0, 1], 0.015)
        add_sphere(vmc_c_world, [0, 1, 0, 1], 0.012)
        add_sphere(wheel_world, [0, 0.5, 1, 1], 0.012)
        add_arrow(hip_world, vmc_c_world, [0, 1, 0, 0.8], 0.004)
        add_arrow(hip_world, wheel_world, [0, 0.5, 1, 0.8], 0.004)


def on_key(keycode):
    if keycode == ord(" "):
        paused_flag["paused"] = not paused_flag["paused"]
        print("---", "PAUSED" if paused_flag["paused"] else "RUNNING", "---")
        print_status(m, d)
        return
    if keycode in (ord("J"), ord("j")):
        verify_vmc_jacobian(m, d)
        return
    if keycode in (ord("T"), ord("t")):
        test_vmc_physics(m, d)
        return
    if keycode in (ord("P"), ord("p")):
        dm = {}
        for name in ["left_front_joint", "right_front_joint", "left_rear_joint", "right_rear_joint"]:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            dm[name] = m.jnt_dofadr[jid]
        pd_test.update({"active": True, "step": 0, "dof_map": dm,
                        "theta_hist": [], "pitch_hist": [], "T_hist": [],
                        "theta_ref_hist": [], "L0_ref_hist": [], "L0_hist": []})
        print(f"\n{'='*60}")
        print(f"PD VMC: θ_ref=±{np.rad2deg(pd_test['theta_ref_amp']):.0f}° @{pd_test['theta_ref_freq']}Hz, "
              f"L0_ref={pd_test['L0_ref_center']:.3f}±{pd_test['L0_ref_amp']:.3f}m @{pd_test['L0_ref_freq']}Hz, "
              f"kp_θ={pd_test['kp_theta']}, kd_θ={pd_test['kd_theta']}, "
              f"kp_L0={pd_test['kp_l0']}, kd_L0={pd_test['kd_l0']}")
        print(f"{'='*60}")
        return
    if keycode in (ord("V"), ord("v")):
        qpos_list = ["%.6f" % d.qpos[i] for i in range(len(d.qpos))]
        qpos_str = " ".join(qpos_list)
        new_line = '    <key name="home" qpos="' + qpos_str + '" />'
        print()
        print("=" * 70)
        print(new_line)
        print("=" * 70)
        print_status(m, d)
        try:
            lines = Path(xml).read_text().splitlines()
            new_lines = []
            replaced = False
            for ln in lines:
                if not replaced and ln.strip().startswith("<key ") and "home" in ln:
                    new_lines.append(new_line)
                    replaced = True
                else:
                    new_lines.append(ln)
            if not replaced:
                for i, ln in enumerate(new_lines):
                    if "</keyframe>" in ln:
                        new_lines.insert(i, new_line)
                        replaced = True
                        break
            Path(xml).write_text("\n".join(new_lines) + "\n")
            print("OK 已写入 scene_flat.xml")
        except Exception as exc:
            print("写入失败:", exc)
        print()


with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
    frame = 0
    print_period = 30
    print("--- RUNNING ---")
    d.qpos[0] = 0.0
    d.qpos[1] = 0.0
    d.qpos[2] = 1.0
    d.qpos[3] = 1.0
    d.qpos[4] = 0.0
    d.qpos[5] = 0.0
    d.qpos[6] = 0.0
    d.qvel[0:6] = 0.0
    print(">>> base 初始化: z=1.0 (避免轮子插地)")
    print_status(m, d)
    print()
    while v.is_running():
        if test_mode["active"]:
            if test_mode["step"] < test_mode["phase_steps"][test_mode["phase"]]:
                if test_mode["step"] == 0:
                    apply_phase_torque(m, d, test_mode["phase"])
                mujoco.mj_step(m, d)
                test_mode["step"] += 1
                if test_mode["step"] >= test_mode["phase_steps"][test_mode["phase"]]:
                    s_after = compute_vmc_state(m, d)
                    dm = test_mode["dof_map"]
                    phase = test_mode["phase"]
                    label = ["正①", "反", "正②"][phase]
                    F_cur = (1 if phase == 0 or phase == 2 else -1) * test_mode["F"]
                    T_cur = (1 if phase == 0 or phase == 2 else -1) * test_mode["T"]

                    lf_v = d.qvel[dm["left_front_joint"]]
                    rf_v = d.qvel[dm["right_front_joint"]]
                    lr_v = d.qvel[dm["left_rear_joint"]]
                    rr_v = d.qvel[dm["right_rear_joint"]]

                    print(f"    施加后: L₀=[{s_after['L0'][0]:.4f}, {s_after['L0'][1]:.4f}], "
                          f"φ₀=[{np.rad2deg(s_after['phi0'][0]):.1f}°, {np.rad2deg(s_after['phi0'][1]):.1f}°]")

                    for leg in range(2):
                        dL0 = s_after["L0"][leg] - test_mode["L0_before"][leg]
                        dphi0 = s_after["phi0"][leg] - test_mode["phi0_before"][leg]
                        L0_ok = dL0 * F_cur > 0
                        phi0_ok = dphi0 * T_cur > 0
                        L0_sign = "✓" if L0_ok else "✗"
                        phi0_sign = "✓" if phi0_ok else "✗"
                        leg_name = ["左腿", "右腿"][leg]
                        print(f"    {leg_name}: ΔL₀={dL0:+.4f} (F={F_cur:+.0f}) {L0_sign}  "
                              f"Δφ₀={np.rad2deg(dphi0):+.1f}° (T={T_cur:+.0f}) {phi0_sign}")

                    print(f"    速度级虚功:")
                    for leg in range(2):
                        phi1_dot = lr_v if leg == 0 else rr_v
                        phi4_dot = lf_v if leg == 0 else rf_v
                        idx = leg
                        inv_s1 = 1.0 / (s_after["sigma1"][idx] + 1e-8)
                        inv_L0 = 1.0 / (s_after["L0"][idx] + 1e-8)
                        sigma2 = np.sin(s_after["phi3"][idx] - s_after["phi4"][idx])
                        sigma3 = np.sin(s_after["phi1"][idx] - s_after["phi2"][idx])
                        J00 = L1 * np.sin(s_after["phi0"][idx] - s_after["phi3"][idx]) * sigma3 * inv_s1
                        J01 = L1 * np.cos(s_after["phi0"][idx] - s_after["phi3"][idx]) * sigma3 * inv_L0 * inv_s1
                        J10 = L4 * np.sin(s_after["phi0"][idx] - s_after["phi2"][idx]) * sigma2 * inv_s1
                        J11 = L4 * np.cos(s_after["phi0"][idx] - s_after["phi2"][idx]) * sigma2 * inv_L0 * inv_s1
                        L0_dot = J00 * phi1_dot + J10 * phi4_dot
                        phi0_dot = J01 * phi1_dot + J11 * phi4_dot
                        power_FT = F_cur * L0_dot + T_cur * phi0_dot
                        tau1_cur = J00 * F_cur + J01 * T_cur
                        tau4_cur = J10 * F_cur + J11 * T_cur
                        power_tau = tau1_cur * phi1_dot + tau4_cur * phi4_dot
                        leg_name = ["左腿", "右腿"][leg]
                        sign = "✓" if abs(power_FT - power_tau) < 1e-5 else "✗"
                        print(f"      {leg_name}: power={power_FT:+.6f} = {power_tau:+.6f} {sign}")

                    # 直接切换，不清零速度
                    d.qfrc_applied[:] = 0.0
                    if test_mode["phase"] < 2:
                        test_mode["phase"] += 1
                        test_mode["step"] = 0
                    else:
                        # 全阶段结束，对比起点
                        net_dL0 = [s_after["L0"][i] - test_mode["L0_origin"][i] for i in range(2)]
                        net_dphi0 = [np.rad2deg(s_after["phi0"][i] - test_mode["phi0_origin"][i]) for i in range(2)]
                        print(f"    净位移: ΔL₀=[{net_dL0[0]:+.4f}, {net_dL0[1]:+.4f}], "
                              f"Δφ₀=[{net_dphi0[0]:+.1f}°, {net_dphi0[1]:+.1f}°]")
                        print(f"    净力矩冲量: 0 (10*(+50)+20*(-50)+10*(+50)=0)")
                        test_mode["active"] = False
                        print(f"{'='*70}\n")
        elif pd_test["active"]:
            theta_l, theta_r, theta_ref, L0_ref, T_l, T_r, F_l, F_r, pitch = apply_pd_vmc(m, d, pd_test["step"])
            mujoco.mj_step(m, d)
            pd_test["step"] += 1

            if pd_test["step"] % 50 == 0:
                print(f"  step={pd_test['step']:4d}  "
                      f"θ_ref={np.rad2deg(theta_ref):+.1f}° → θ=({np.rad2deg(theta_l):+.1f}°,{np.rad2deg(theta_r):+.1f}°)  "
                      f"L0_ref={L0_ref:.3f}  "
                      f"VMC_T=({T_l:+.2f},{T_r:+.2f})N·m  "
                      f"VMC_F=({F_l:+.2f},{F_r:+.2f})N")

            if pd_test["step"] >= pd_test["total_steps"]:
                pd_test["active"] = False
                T_hist = np.array(pd_test["T_hist"])
                theta_hist = np.array(pd_test["theta_hist"])
                theta_ref_hist = np.array(pd_test["theta_ref_hist"])
                L0_hist = np.array(pd_test["L0_hist"])
                L0_ref_hist = np.array(pd_test["L0_ref_hist"])
                print(f"\n--- PD VMC 结果 ({pd_test['total_steps']} steps) ---")
                print(f"  VMC_T 均值: ({np.mean(T_hist[:,0]):+.2f}, {np.mean(T_hist[:,1]):+.2f}) N·m")
                print(f"  VMC_T 峰值: ({np.max(np.abs(T_hist[:,0])):.2f}, {np.max(np.abs(T_hist[:,1])):.2f}) N·m")
                print(f"  θ 跟踪误差 RMS: {np.rad2deg(np.sqrt(np.mean((theta_hist[:,0] - theta_ref_hist)**2))):.1f}°")
                print(f"  L0 跟踪误差 RMS: {np.sqrt(np.mean((L0_hist[:,0] - L0_ref_hist)**2))*1000:.1f}mm")
                print(f"{'='*60}\n")
        elif not paused_flag["paused"]:
            mujoco.mj_step(m, d)
            if d.ncon > 0 and frame < 5:
                print(f"  [frame {frame}] 接触数={d.ncon}:")
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1])
                    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2])
                    print(f"    {b1}/{g1} ↔ {b2}/{g2}  dist={c.dist:.4f}")
        d.qpos[0] = 0.0
        d.qpos[1] = 0.0
        d.qpos[2] = 1.0
        d.qpos[3] = 1.0
        d.qpos[4] = 0.0
        d.qpos[5] = 0.0
        d.qpos[6] = 0.0
        d.qvel[0:6] = 0.0
        mujoco.mj_forward(m, d)
        frame += 1
        if frame % print_period == 0:
            print_status(m, d)
        add_vmc_visualization(v, m, d)
        v.sync()
        time.sleep(0.016)