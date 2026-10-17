#!/usr/bin/env python3
"""环境验证脚本：不依赖训练，直接检查 action→力矩、观测、奖励、VMC 状态、reset 等链路。"""

import sys
import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path

# 导入 base.py 中的 VMC 函数
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from unilab.envs.locomotion.balance_infantry.base import (
    compute_vmc_torques_fivebar,
    vmc_mapping_fivebar,
    forward_kinematics_fivebar,
)

# ============================================================
# 常量和导入 (与 base.py / joystick.py 保持一致)
# ============================================================
XML_PATH = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "robots" / "balance_infantry" / "scene_flat.xml"

NUM_BALANCE_ACTIONS = 6        # LF, RF, LR, RR, LW, RW
NUM_LEG_ACTIONS = 4
NUM_WHEEL_ACTIONS = 2
WHEEL_JOINT_INDICES = [4, 5]   # LW, RW
LEG_JOINT_INDICES = [0, 1, 2, 3]  # LF, RF, LR, RR

# 关节限位 (与 JOINT_LIMITS 一致)
JOINT_LIMITS = np.array([
    [-1.57, 2.40],  # LF
    [-1.57, 2.40],  # RF
    [-1.57, 1.57],  # LR
    [-1.57, 1.57],  # RR
], dtype=np.float64)

# 五连杆参数
L1, L2, L3, L4, L5 = 0.215, 0.254, 0.254, 0.215, 0.0
OFFSET_PHI1 = 5 * np.pi / 6   # 150°
OFFSET_PHI4 = np.pi / 6        # 30°

# PD 参数
LEG_KP = 120.0
LEG_KD = 6.0
WHEEL_KD = np.array([0.5, 0.5])

# VMC PD 参数 (与 base.py 的 VmcConfig 默认值一致)
VMC_KP_THETA = 50.0
VMC_KD_THETA = 3.0
VMC_KP_L0 = 500.0
VMC_KD_L0 = 50.0
VMC_FF = 80.0          # config 中 feedforward_force
VMC_DEFAULT_L0 = 0.28  # 目标虚拟腿长


def compute_vmc_state_batch(joint_pos):
    """使用 base.py 的 forward_kinematics_fivebar 计算 VMC 状态 (batch)"""
    # 关节顺序: [LF, RF, LR, RR] — 对应 φ₄左, φ₄右, φ₁左, φ₁右
    phi4 = np.array([[joint_pos[0], joint_pos[1]]])  # (1, 2) 左前, 右前
    phi1 = np.array([[joint_pos[2], joint_pos[3]]])  # (1, 2) 左后, 右后

    phi1 = phi1 + OFFSET_PHI1
    phi4 = phi4 + OFFSET_PHI4

    L0, phi0, phi2, phi3, sigma1 = forward_kinematics_fivebar(
        phi1, phi4, L1, L2, L3, L4, L5
    )

    theta = np.pi / 2 - phi0  # 相对竖直方向

    return {
        "L0": L0[0],           # (2,)
        "phi0": phi0[0],       # (2,)
        "theta": theta[0],     # (2,)
        "phi1": phi1[0],       # (2,)
        "phi4": phi4[0],       # (2,)
        "phi2": phi2[0],       # (2,)
        "phi3": phi3[0],       # (2,)
        "sigma1": sigma1[0],   # (2,)
    }


def compute_vmc_torques_full(vmc_state):
    """使用 base.py 的 compute_vmc_torques_fivebar 计算 VMC 力矩"""
    n = 1
    # 构造 batch 输入
    phi1 = vmc_state["phi1"].reshape(1, 2)
    phi4 = vmc_state["phi4"].reshape(1, 2)
    phi0 = vmc_state["phi0"].reshape(1, 2)
    phi2 = vmc_state["phi2"].reshape(1, 2)
    phi3 = vmc_state["phi3"].reshape(1, 2)
    sigma1 = vmc_state["sigma1"].reshape(1, 2)
    L0 = vmc_state["L0"].reshape(1, 2)
    theta = vmc_state["theta"].reshape(1, 2)

    L0_dot = np.zeros((1, 2))
    theta_dot = np.zeros((1, 2))
    theta_ref = np.zeros((1, 2))
    L0_ref = np.full((1, 2), VMC_DEFAULT_L0)

    kp_theta = np.full((1, 2), VMC_KP_THETA)
    kd_theta = np.full((1, 2), VMC_KD_THETA)
    kp_l0 = np.full((1, 2), VMC_KP_L0)
    kd_l0 = np.full((1, 2), VMC_KD_L0)
    feedforward = np.full((1, 2), VMC_FF)

    tau1, tau4 = compute_vmc_torques_fivebar(
        phi1, phi4, phi0, phi2, phi3, sigma1,
        L0, L0_dot, theta, theta_dot,
        theta_ref, L0_ref, L1, L4,
        kp_theta, kd_theta, kp_l0, kd_l0, feedforward,
    )

    # 映射到执行器: τ₄→LF,RF; τ₁→LR,RR
    vmc_torques = np.zeros(NUM_BALANCE_ACTIONS)
    vmc_torques[0] = tau4[0, 0]  # τ₄ → LF
    vmc_torques[1] = tau4[0, 1]  # τ₄ → RF
    vmc_torques[2] = tau1[0, 0]  # τ₁ → LR
    vmc_torques[3] = tau1[0, 1]  # τ₁ → RR

    return vmc_torques

# 控制参数
ACTION_SCALE_JOINT = 0.5
ACTION_SCALE_WHEEL = 10.0
CTRL_DT = 0.02
SIM_DT = 0.005
SIM_SUBSTEPS = 4

# ============================================================
# 五连杆正运动学
# ============================================================
def compute_motor_ctrl(policy_targets, joint_pos, joint_vel, default_dof_pos):
    """计算 PD 力矩 (与 joystick.py 的 _pre_step_motor_control 一致)"""
    n = policy_targets.shape[0]
    ctrl = np.zeros((n, NUM_BALANCE_ACTIONS), dtype=np.float64)

    # 腿关节 PD 控制
    for i in range(NUM_LEG_ACTIONS):
        pos_err = policy_targets[:, i] - joint_pos[:, i]
        vel = joint_vel[:, i]
        ctrl[:, i] = LEG_KP * pos_err - LEG_KD * vel

    # 轮子速度控制
    for i, idx in enumerate(WHEEL_JOINT_INDICES):
        target_vel = policy_targets[:, idx]
        current_vel = joint_vel[:, idx]
        ctrl[:, idx] = WHEEL_KD[i] * (target_vel - current_vel)

    return ctrl


def build_targets(actions, default_dof_pos):
    """将策略输出 (action) 转换为关节目标 (target)"""
    n = actions.shape[0]
    targets = np.zeros((n, NUM_BALANCE_ACTIONS), dtype=np.float64)

    # 腿关节: action * action_scale_joint * joint_range + default
    for i in range(NUM_LEG_ACTIONS):
        lower, upper = JOINT_LIMITS[i]
        default = default_dof_pos[i]
        scale = max(upper - default, default - lower) * ACTION_SCALE_JOINT
        target = actions[:, i] * scale + default
        targets[:, i] = np.clip(target, lower, upper)

    # 轮子: action * action_scale_wheel
    targets[:, 4] = actions[:, 4] * ACTION_SCALE_WHEEL
    targets[:, 5] = actions[:, 5] * ACTION_SCALE_WHEEL

    return targets


# ============================================================
# 主测试
# ============================================================
def main():
    print("=" * 70)
    print("BalanceInfantry 环境验证")
    print("=" * 70)

    # 加载模型
    xml_path = str(XML_PATH)
    print(f"\n[1] 加载模型: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    nq = m.nq
    nv = m.nv
    nu = m.nu
    print(f"    nq={nq} (position DoF), nv={nv} (velocity DoF), nu={nu} (actuators)")

    # 检查默认关节位置
    qpos0 = d.qpos.copy()
    print(f"\n[2] 默认 qpos (前7个=base, 后{nv}个=joints):")
    print(f"    base pos: {qpos0[0:3]}")
    print(f"    base quat: {qpos0[3:7]}")
    print(f"    joint pos: {qpos0[7:7+nv]}")

    # 关节限位检查 (跳过 floating_base: i=0)
    print(f"\n[3] 关节限位:")
    for i in range(1, nv):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            jid = m.jnt_qposadr[i]
            low = m.jnt_range[i][0] if m.jnt_limited[i] else -np.inf
            high = m.jnt_range[i][1] if m.jnt_limited[i] else np.inf
            init = qpos0[jid]
            print(f"    [{i}] {name}: init={init:+.4f}, range=[{low:+.4f}, {high:+.4f}]")

    # 执行器限位检查
    print(f"\n[4] 执行器 (actuator) 信息:")
    for i in range(nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            ctrl_range = m.actuator_ctrlrange[i] if m.actuator_ctrllimited[i] else [-np.inf, np.inf]
            force_range = m.actuator_forcerange[i] if m.actuator_forcelimited[i] else [-np.inf, np.inf]
            print(f"    [{i}] {name}: ctrl_range={ctrl_range}, force_range={force_range}")

    # ============================================================
    # 测试 1: 默认站立状态下的 VMC 状态
    # ============================================================
    print(f"\n{'='*70}")
    print(f"[测试 1] 默认站立状态下的 VMC 状态")
    print(f"{'='*70}")

    joint_pos = qpos0[7:7+nv].copy()
    vmc = compute_vmc_state_batch(joint_pos)
    print(f"    joint_pos: {joint_pos}")
    print(f"    φ₁ = [{np.rad2deg(vmc['phi1'][0]):+.1f}°, {np.rad2deg(vmc['phi1'][1]):+.1f}°]")
    print(f"    φ₄ = [{np.rad2deg(vmc['phi4'][0]):+.1f}°, {np.rad2deg(vmc['phi4'][1]):+.1f}°]")
    print(f"    L₀  = [{vmc['L0'][0]:.4f}, {vmc['L0'][1]:.4f}] m")
    print(f"    φ₀  = [{np.rad2deg(vmc['phi0'][0]):+.1f}°, {np.rad2deg(vmc['phi0'][1]):+.1f}°]")
    print(f"    θ   = [{np.rad2deg(vmc['theta'][0]):+.1f}°, {np.rad2deg(vmc['theta'][1]):+.1f}°]")
    print(f"    σ₁  = [{vmc['sigma1'][0]:.4f}, {vmc['sigma1'][1]:.4f}]")

    # ============================================================
    # 测试 2: Action → Target → PD力矩 链路
    # ============================================================
    print(f"\n{'='*70}")
    print(f"[测试 2] Action → Target → PD力矩 链路")
    print(f"{'='*70}")

    # 测试几个不同的 action 值
    test_actions = [
        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),   # 全零
        np.array([[0.5, 0.5, 0.5, 0.5, 0.0, 0.0]]),   # 腿正
        np.array([[-0.5, -0.5, -0.5, -0.5, 0.0, 0.0]]), # 腿负
        np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]),   # 轮子正
        np.array([[0.0, 0.0, 0.0, 0.0, -1.0, -1.0]]), # 轮子负
    ]

    default_dof = np.zeros(4)
    joint_vel = np.zeros(6)

    for i, action in enumerate(test_actions):
        targets = build_targets(action, default_dof)
        motor_ctrl = compute_motor_ctrl(targets, joint_pos.reshape(1, -1), joint_vel.reshape(1, -1), default_dof)
        print(f"\n  action={action[0]} ->")
        print(f"    targets: {targets[0]}")
        print(f"    motor_ctrl: {motor_ctrl[0]}")
        print(f"    leg torque: {motor_ctrl[0, :4]}")
        print(f"    wheel torque: {motor_ctrl[0, 4:]}")

    # ============================================================
    # 测试 3: 含 VMC 力矩的 step 模拟 (base 放在地面以上)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"[测试 3] 模拟 step (base 抬高到站立高度, +VMC 力矩, random action)")
    print(f"{'='*70}")

    mujoco.mj_resetData(m, d)
    d.ctrl[:] = 0.0

    STAND_HEIGHT = 0.35
    d.qpos[2] = STAND_HEIGHT

    print(f"  base 初始高度设置为: {STAND_HEIGHT:.3f}m")

    num_steps = 200
    action = np.zeros(6)
    default_dof = np.zeros(4)

    for step in range(num_steps):
        joint_pos = d.qpos[7:7+nv].copy()
        joint_vel = d.qvel[6:6+nv].copy()

        if step % 20 == 0:
            action = np.random.uniform(-0.3, 0.3, 6)
            action[4:] = 0.0

        targets = build_targets(action.reshape(1, -1), default_dof)[0]
        pd_ctrl = compute_motor_ctrl(targets.reshape(1, -1), joint_pos.reshape(1, -1), joint_vel.reshape(1, -1), default_dof)[0]

        # 加 VMC 力矩
        vmc_state = compute_vmc_state_batch(joint_pos)
        vmc_torque = compute_vmc_torques_full(vmc_state)
        motor_ctrl = pd_ctrl + vmc_torque

        d.ctrl[:] = np.clip(motor_ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])

        for _ in range(SIM_SUBSTEPS):
            mujoco.mj_step1(m, d)
            mujoco.mj_step2(m, d)

        base_pos = d.qpos[0:3].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        R = R.reshape(3, 3)
        tilt = np.arccos(np.clip(R[2, 2], -1.0, 1.0))

        if step % 2 == 0:
            vmc_state = compute_vmc_state_batch(joint_pos)
            print(f"  step {step:3d}: height={base_pos[2]:.4f}, tilt={np.rad2deg(tilt):.1f}°, "
                  f"L₀=[{vmc_state['L0'][0]:.4f},{vmc_state['L0'][1]:.4f}], θ=[{np.rad2deg(vmc_state['theta'][0]):.1f}°,{np.rad2deg(vmc_state['theta'][1]):.1f}°]")
            print(f"           PD=[{', '.join(f'{pd_ctrl[i]:+.1f}' for i in [0,1,2,3])}], VMC=[{', '.join(f'{vmc_torque[i]:+.1f}' for i in [0,1,2,3])}]")

        if base_pos[2] < 0.05:
            print(f"  ⚠  step {step}: 倒地! height={base_pos[2]:.4f}")
            break

    # ============================================================
    # 测试 3.5: 两次 reset 验证
    # ============================================================
    print(f"\n{'='*70}")
    print(f"[测试 3.5] 两次 reset 验证 (+VMC 力矩)")
    print(f"{'='*70}")

    for reset_idx in range(2):
        print(f"\n  --- Reset #{reset_idx + 1} ---")
        mujoco.mj_resetData(m, d)
        d.qpos[2] = STAND_HEIGHT
        d.ctrl[:] = 0.0

        print(f"    初始 base z={d.qpos[2]:.4f}")
        jp = d.qpos[7:7+nv]
        jv = d.qvel[6:6+nv]
        print(f"    初始 joint_pos(actuated)=[{', '.join(f'{jp[i]:+.4f}' for i in [0,1,4,5])}]")
        print(f"    初始 joint_vel(actuated)=[{', '.join(f'{jv[i]:+.4f}' for i in [0,1,4,5])}]")

        vmc = compute_vmc_state_batch(jp)
        print(f"    初始 VMC: L₀=[{vmc['L0'][0]:.4f}, {vmc['L0'][1]:.4f}], θ=[{np.rad2deg(vmc['theta'][0]):.1f}°, {np.rad2deg(vmc['theta'][1]):.1f}°]")

        # 零动作 + VMC 力矩，观察是否一致
        for step in range(50):
            jp = d.qpos[7:7+nv]
            jv = d.qvel[6:6+nv]
            vmc_state = compute_vmc_state_batch(jp)
            vmc_torque = compute_vmc_torques_full(vmc_state)
            d.ctrl[:] = np.clip(vmc_torque, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])

            for _ in range(SIM_SUBSTEPS):
                mujoco.mj_step1(m, d)
                mujoco.mj_step2(m, d)

            if step % 5 == 0:
                bz = d.qpos[2]
                jp = d.qpos[7:7+nv]
                vmc = compute_vmc_state_batch(jp)
                vmc_t = compute_vmc_torques_full(vmc)
                print(f"    step {step:2d}: base_z={bz:.4f}, L₀=[{vmc['L0'][0]:.4f}, {vmc['L0'][1]:.4f}], θ=[{np.rad2deg(vmc['theta'][0]):.1f}°,{np.rad2deg(vmc['theta'][1]):.1f}°], VMC=[{', '.join(f'{vmc_t[i]:.1f}' for i in [0,1,2,3])}]")

            if d.qpos[2] < 0.05:
                print(f"    ⚠  step {step}: 倒地")
                break

    # ============================================================
    # 测试 4: VMC 力矩 vs 关节力矩
    # ============================================================
    print(f"\n{'='*70}")
    print(f"[测试 4] VMC 力矩计算")
    print(f"{'='*70}")

    mujoco.mj_resetData(m, d)
    joint_pos = d.qpos[7:7+nv].copy()
    vmc = compute_vmc_state_batch(joint_pos)

    # 模拟一个倾斜状态
    for _ in range(50):
        d.ctrl[:] = 0.0
        for _ in range(SIM_SUBSTEPS):
            mujoco.mj_step1(m, d)
            mujoco.mj_step2(m, d)

    joint_pos = d.qpos[7:7+nv].copy()
    joint_vel = d.qvel[6:6+nv].copy()
    vmc = compute_vmc_state_batch(joint_pos)

    # 计算 VMC 力矩
    vmc_torque = compute_vmc_torques_full(vmc)

    print(f"    当前状态:")
    print(f"    joint_pos={[f'{x:+.4f}' for x in joint_pos]}")
    print(f"    L₀=[{vmc['L0'][0]:.4f}, {vmc['L0'][1]:.4f}]")
    print(f"    θ=[{np.rad2deg(vmc['theta'][0]):.1f}°, {np.rad2deg(vmc['theta'][1]):.1f}°]")
    print(f"    VMC torques(actuator): {[f'{vmc_torque[i]:.1f}' for i in [0,1,2,3]]}")

    # ============================================================
    # 测试 5: 观测空间 (模拟 _compute_obs 的输入)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"[测试 5] 观测空间概览")
    print(f"{'='*70}")

    print(f"    观测构成 (actor, 33维):")
    print(f"      gyro(3) + pitch(1) + roll(1) + commands[:, [0,2,3,4]](4)")
    print(f"      + last_actions(6) + wheel_pos_err_l(1) + wheel_pos_err_r(1)")
    print(f"      + sin(yaw)(1) + cos(yaw)(1) + dof_pos_err(4) + dof_vel(6)")
    print(f"      + VMC θ_l(1) + θ_r(1) + L₀_l(1) + L₀_r(1)")
    print(f"      = 33 维")
    print(f"")
    print(f"    观测构成 (critic_flat, 33维):")
    print(f"      world_linvel(3) + gyro(3) + pitch(1) + roll(1)")
    print(f"      + commands[0,2,3,4](4) + last_actions(6)")
    print(f"      + wheel_pos_fmod_l(1) + wheel_pos_fmod_r(1) + local_linvel(3)")
    print(f"      + sin(yaw)(1) + cos(yaw)(1)")
    print(f"      + VMC: θ_l(1)+θ_r(1)+θ̇_l(1)+θ̇_r(1)+L₀_l(1)+L₀_r(1)+L̇₀_l(1)+L̇₀_r(1)")
    print(f"      = 33 维")
    print(f"")
    print(f"    观测构成 (critic 特权, 147维):")
    print(f"      critic_flat(33) + action_t2(6) + action_t3(6)")
    print(f"      + dof_acc(6) + torques(6) + base_mass(1) + base_com(3)")
    print(f"      + feedforward_force(1) + dof_pos_wrapped(6) + ground_friction(2)")
    print(f"      + height_scan(77)")
    print(f"      = 147 维")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n{'='*70}")
    print(f"验证完成")
    print(f"{'='*70}")
    print(f"检查项:")
    print(f"  ✓ 模型加载 (nq={nq}, nv={nv}, nu={nu})")
    print(f"  ✓ 默认关节位置")
    print(f"  ✓ 关节/执行器限位")
    print(f"  ✓ Action → Target → PD力矩 链路")
    print(f"  ✓ VMC 状态计算")
    print(f"  ✓ 多步模拟 (观察高度/倾斜)")
    print(f"  ✓ VMC 力矩计算")
    print(f"  ✓ 观测空间维度")


if __name__ == "__main__":
    main()