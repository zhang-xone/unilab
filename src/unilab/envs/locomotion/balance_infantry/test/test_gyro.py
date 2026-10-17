"""测试 gyro 三轴方向 + pitch/roll/theta 角度: 手动设角速度/姿态, 看机器人反应。

按键:
  1 → gyro_x = +3.0 (绕 X/前进), 右滚(+) or 左滚(-)?
  2 → gyro_y = +3.0 (绕 Y/左),   低头(+) or 抬头(-)?
  3 → gyro_z = +3.0 (绕 Z/上),   左转(+) or 右转(-)?
  4 → pitch = +20° (绕 Y 低头),  obs.pitch 是正还是负?
  5 → roll  = +20° (绕 X 右滚),  obs.roll  是正还是负?
  6 → 复位 (零姿态, 零速度)
  7 → theta≈0°  (默认关节=0, 腿竖直向下)
  8 → theta偏正  (关节全+0.5)
  9 → theta偏负  (关节全-0.5)
  0 → 停止 gyro 旋转
  Z → 腿关节 dof_vel 全+5 rad/s  (关节正速度方向)
  X → 腿关节 dof_vel 全-5 rad/s
  C → 轮子 dof_vel 全+10 rad/s
  V → 轮子 dof_vel 全-10 rad/s
  B → 停止所有关节速度
  U → LF 单独 +joint_vel    I → RF 单独 +joint_vel
  O → LR 单独 +joint_vel    P → RR 单独 +joint_vel
  F → 切换 base 固定/自由 (固定时 base 不动, 只看关节转)
  M → 注入实物电机位置 (LF/RF/LR/RR + pitch=2°), 对比 theta/L0
  R → 完全复位 (mujoco reset: qpos + qvel 全部归零)
  Q → 退出
"""

import mujoco
import mujoco.viewer
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantize"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lqr"))
from test_quantized import build_observation, debug_print_obs
from test_vmc_lqr import fk_fivebar

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..")
SCENE_DIR = os.path.join(_PARENT_DIR, "assets", "robots", "balance_infantry")
SCENE = os.path.join(SCENE_DIR, "scene_step.xml")

if not os.path.exists(SCENE):
    SCENE = os.path.join(SCENE_DIR, "scene_flat.xml")

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
model.opt.timestep = 0.005

# 关重力, 让机器人在空中自由旋转不摔
model.opt.gravity[:] = 0
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY

# 先改 model.qpos0 默认出生高度, 再 reset, 第一帧就是 1m
model.qpos0[2] = 1.0
mujoco.mj_resetData(model, data)

mujoco.mj_forward(model, data)

test_idx = 0
vel = 3.0
joint_vel = 1.0
leg_torque = 5.0       # Nm, 腿关节力矩
wheel_torque = 0.05    # Nm, 轮子力矩
base_fixed = False

# M 键 PD 目标
pd_active = False
pd_target = np.array([-0.881758, -0.878032, -0.101421, -0.123270])
pd_kp = 80.0   # Nm/rad
pd_kd = 1.0    # Nm/(rad/s)

# MJCF 关节顺序: 所有左侧在前, 右侧在后
# qvel 索引 (free joint 占 0-5)
LF_V, LR_V, LW_V = 6, 10, 12   # left_front, left_rear, left_wheel
RF_V, RR_V, RW_V = 13, 17, 19   # right_front, right_rear, right_wheel
# qpos 索引 (free joint 占 0-6)
LF_P, LR_P, LW_P = 7, 11, 13
RF_P, RR_P, RW_P = 14, 18, 20
# ctrl 索引 (actuator 顺序: LF, RF, LR, RR, LW, RW)
LF_C, RF_C, LR_C, RR_C = 0, 1, 2, 3
LW_C, RW_C = 4, 5

print("=" * 60)
print("  GYRO / PITCH+ROLL / YAW / THETA / DOF_VEL / TORQUE 方向验证")
print("  [1]gyroX [2]gyroY [3]gyroZ [4]pitch [5]roll [6]复位")
print("  [7]theta≈0 [8]theta偏正 [9]theta偏负 [0]停gyro")
print("  --- 姿态 ---")
print("  [W]yaw+30° [E]yaw-30°")
print("  --- 速度注入 ---")
print("  [Z]腿v+ [X]腿v- [C]轮v+ [V]轮v- [B]停关节 [F]固定base")
print("  [U]LF [I]RF [O]LR [P]RR  单独关节速度")
print("  --- 力矩注入 ---")
print("  [T]腿τ+ [Y]腿τ- [G]轮τ+ [H]轮τ-")
print("  [J]LFτ [K]RFτ [L]LRτ [;]RRτ  单独关节力矩")
print(f"  速度: gyro={vel} 关节={joint_vel} rad/s | 力矩: 腿={leg_torque} 轮={wheel_torque} Nm | 倾角=20° yaw=30°")
print("=" * 60)


def compute_obs_pr(quat_mj):
    """用观测代码公式 (scalar-first) 计算 pitch 和 roll"""
    qw, qx, qy, qz = quat_mj[0], quat_mj[1], quat_mj[2], quat_mj[3]
    gx = 2 * (qx * qz - qw * qy)
    gy = 2 * (qw * qx + qy * qz)
    pitch = np.arcsin(gx)
    roll = np.arcsin(-gy)
    return np.degrees(pitch), np.degrees(roll)


def compute_theta():
    """用 VMC 公式: theta = π/2 - phi0 - pitch"""
    q = np.array([data.qpos[LF_P], data.qpos[RF_P], data.qpos[LR_P], data.qpos[RR_P]])
    qw, qx, qy, qz = data.qpos[3:7]
    pitch = np.arcsin(np.clip(2.0*(qw*qy - qx*qz), -1.0, 1.0))

    phi1_l = q[2] + 5*np.pi/6; phi4_l = q[0] + np.pi/6
    phi1_r = q[3] + 5*np.pi/6; phi4_r = q[1] + np.pi/6

    L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
    L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)

    theta_l = np.pi/2 - phi0_l - pitch
    theta_r = np.pi/2 - phi0_r - pitch

    return np.degrees(theta_l), np.degrees(theta_r), L0_l, L0_r, q.copy(), np.degrees(pitch)


def print_state():
    gx, gy, gz = data.qvel[3], data.qvel[4], data.qvel[5]
    obs_p, obs_r = compute_obs_pr(data.qpos[3:7])
    theta_l, theta_r, L0_l, L0_r, qj, pitch_deg = compute_theta()
    jvel = np.array([data.qvel[LF_V], data.qvel[RF_V], data.qvel[LR_V], data.qvel[RR_V],
                       data.qvel[LW_V], data.qvel[RW_V]])
    qw, qx, qy, qz = data.qpos[3:7]
    yaw = np.degrees(np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz)))
    sin_yaw = np.sin(np.radians(yaw))
    cos_yaw = np.cos(np.radians(yaw))
    labels = ['停','gyroX','gyroY','gyroZ','pitch','roll','复位','θ=0°','θ=+15°','θ=-15°',
              '腿v+','腿v-','轮v+','轮v-','停关节','LFv','RFv','LRv','RRv',
              '腿τ+','腿τ-','轮τ+','轮τ-','LFτ','RFτ','LRτ','RRτ',
              'yaw+30°','yaw-30°','实物pose']
    print(f"  gyro=[{gx:+.3f},{gy:+.3f},{gz:+.3f}] rad/s  "
          f"obs_pitch={obs_p:+.2f}° obs_roll={obs_r:+.2f}°  | 模式={labels[test_idx]}")
    print(f"  theta=({theta_l:+.2f},{theta_r:+.2f})°  "
          f"L0=({L0_l:.3f},{L0_r:.3f})m  "
          f"q=[{qj[0]:+.2f},{qj[1]:+.2f},{qj[2]:+.2f},{qj[3]:+.2f}]")
    print(f"  dof_vel=[LF:{jvel[0]:+.2f},RF:{jvel[1]:+.2f},LR:{jvel[2]:+.2f},RR:{jvel[3]:+.2f}  "
          f"LW:{jvel[4]:+.2f},RW:{jvel[5]:+.2f}] rad/s  base={'固定' if base_fixed else '自由'}")
    print(f"  yaw={yaw:+.2f}°  sin(yaw)={sin_yaw:+.4f}  cos(yaw)={cos_yaw:+.4f}")


def print_full_obs():
    """调用 test_quantized.py 的 build_observation + debug_print_obs 打印完整观测"""
    dof_pos = data.qpos[7:].copy()
    dof_vel = data.qvel[6:].copy()
    default_dof_pos = model.qpos0[7:].copy() if len(model.qpos0) > 7 else np.zeros_like(dof_pos)
    last_actions = np.zeros(6, dtype=np.float32)
    wheel_pos_err = np.zeros(2, dtype=np.float32)
    yaw_error = 0.0

    obs = build_observation(
        model, data, dof_pos, dof_vel, last_actions,
        cmd_vel=0.0, cmd_yaw_rate=0.0, cmd_height=0.29,
        wheel_pos_err=wheel_pos_err, yaw_error=yaw_error,
        default_dof_pos=default_dof_pos,
    )
    debug_print_obs(obs, step=0)


def key_callback(keycode):
    global test_idx, vel, pd_active

    def quat_from_pitch(deg):
        h = np.radians(deg / 2)
        return np.array([np.cos(h), 0, np.sin(h), 0])

    def quat_from_roll(deg):
        h = np.radians(deg / 2)
        return np.array([np.cos(h), np.sin(h), 0, 0])

    def quat_from_yaw(deg):
        h = np.radians(deg / 2)
        return np.array([np.cos(h), 0, 0, np.sin(h)])

    if keycode == 49:   # 1
        test_idx = 1
        data.qvel[3:6] = [vel, 0, 0]
        print("\n[gyroX] 绕 X 轴(前进) +3.0 rad/s")
        print("  右手拇指指前 → 正=右滚")
        print("  观察机器人往哪边倒? → R=右滚对, L=左滚对")
        print_state()
    elif keycode == 50:  # 2
        test_idx = 2
        data.qvel[3:6] = [0, vel, 0]
        print("\n[gyroY] 绕 Y 轴(左) +3.0 rad/s")
        print("  右手拇指指左 → 正=低头")
        print("  观察机器人往哪边栽? → F=低头对, B=抬头对")
        print_state()
    elif keycode == 51:  # 3
        test_idx = 3
        data.qvel[3:6] = [0, 0, vel]
        print("\n[gyroZ] 绕 Z 轴(上) +3.0 rad/s")
        print("  右手拇指指上 → 正=左转")
        print("  观察机器人往哪边转? → L=左转对, R=右转对")
        print_state()
    elif keycode == 52:  # 4
        test_idx = 4
        data.qvel[3:6] = 0
        data.qpos[3:7] = quat_from_pitch(20)
        mujoco.mj_forward(model, data)
        obs_p, obs_r = compute_obs_pr(data.qpos[3:7])
        print(f"\n[pitch] 设 pitch=+20° (绕Y-左轴, 右手定则: 低头)")
        print(f"  观察屏幕: 机器人是往前栽(低头) 还是 往后仰(抬头)?")
        print(f"  → obs.pitch = {obs_p:+.2f}°  (正=__ 负=__)")
        print(f"     obs.roll  = {obs_r:+.2f}°")
        print_state()
        print_full_obs()
    elif keycode == 53:  # 5
        test_idx = 5
        data.qvel[3:6] = 0
        data.qpos[3:7] = quat_from_roll(20)
        mujoco.mj_forward(model, data)
        obs_p, obs_r = compute_obs_pr(data.qpos[3:7])
        print(f"\n[roll] 设 roll=+20° (绕X-前轴, 右手定则: 右滚)")
        print(f"  观察屏幕: 机器人是往右倒 还是 往左倒?")
        print(f"  → obs.pitch = {obs_p:+.2f}°")
        print(f"     obs.roll  = {obs_r:+.2f}°  (正=__ 负=__)")
        print_state()
        print_full_obs()
    elif keycode == 54:  # 6
        test_idx = 6
        data.qvel[3:6] = 0
        data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        print("\n[复位] 零姿态, 零速度")
        print_state()
        print_full_obs()
    elif keycode == 55:  # 7  theta≈0
        test_idx = 7
        data.qvel[3:6] = 0
        data.qpos[LF_P], data.qpos[RF_P], data.qpos[LR_P], data.qpos[RR_P] = 0.0, 0.0, 0.0, 0.0
        data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        theta_l, theta_r, L0_l, L0_r, _, _ = compute_theta()
        print(f"\n[theta≈0] 默认关节=0, 姿态归零")
        print(f"  theta=({theta_l:+.2f},{theta_r:+.2f})°  → 腿在身体{('前方' if theta_l>0 else '后方')}")
        print(f"  观察窗口: 腿是竖直向下?")
        print_state()
    elif keycode == 56:  # 8  theta偏正
        test_idx = 8
        data.qvel[3:6] = 0
        data.qpos[LF_P], data.qpos[RF_P], data.qpos[LR_P], data.qpos[RR_P] = 0.5, 0.5, 0.5, 0.5
        data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        theta_l, theta_r, L0_l, L0_r, _, _ = compute_theta()
        print(f"\n[theta偏正] 关节全+0.5 rad")
        print(f"  theta=({theta_l:+.2f},{theta_r:+.2f})°  → 腿在身体{('前方' if theta_l>0 else '后方')}")
        print(f"  观察窗口: 腿往哪边偏?")
        print_state()
    elif keycode == 57:  # 9  theta偏负
        test_idx = 9
        data.qvel[3:6] = 0
        data.qpos[LF_P], data.qpos[RF_P], data.qpos[LR_P], data.qpos[RR_P] = -0.5, -0.5, -0.5, -0.5
        data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        theta_l, theta_r, L0_l, L0_r, _, _ = compute_theta()
        print(f"\n[theta偏负] 关节全-0.5 rad")
        print(f"  theta=({theta_l:+.2f},{theta_r:+.2f})°  → 腿在身体{('前方' if theta_l>0 else '后方')}")
        print(f"  观察窗口: 腿往哪边偏?")
        print_state()
    elif keycode == 48:  # 0
        test_idx = 0
        data.qvel[3:6] = 0
        print("\n[停] 角速度清零")
        print_state()
    elif keycode == 90:  # Z → 腿关节速度 +
        test_idx = 10
        data.qvel[3:6] = 0
        data.qvel[LF_V], data.qvel[RF_V], data.qvel[LR_V], data.qvel[RR_V] = joint_vel, joint_vel, joint_vel, joint_vel
        mujoco.mj_forward(model, data)
        print(f"\n[腿v+] 注入腿关节 dof_vel = +{joint_vel:.1f} rad/s (LF/RF/LR/RR)")
        print("  关节速度为正 → 关节往 dof_pos 增大的方向转")
        print("  观察窗口: 每条腿往哪个方向摆? (前/后?)")
        print_state()
    elif keycode == 88:  # X → 腿关节速度 -
        test_idx = 11
        data.qvel[3:6] = 0
        data.qvel[LF_V], data.qvel[RF_V], data.qvel[LR_V], data.qvel[RR_V] = -joint_vel, -joint_vel, -joint_vel, -joint_vel
        mujoco.mj_forward(model, data)
        print(f"\n[腿v-] 注入腿关节 dof_vel = -{joint_vel:.1f} rad/s (LF/RF/LR/RR)")
        print("  关节速度为负 → 关节往 dof_pos 减小的方向转")
        print("  观察窗口: 每条腿往哪个方向摆? (前/后?)")
        print_state()
    elif keycode == 67:  # C → 轮子速度 +
        test_idx = 12
        data.qvel[3:6] = 0
        data.qvel[LF_V], data.qvel[RF_V], data.qvel[LR_V], data.qvel[RR_V] = 0, 0, 0, 0
        data.qvel[LW_V], data.qvel[RW_V] = 3.0, 3.0
        mujoco.mj_forward(model, data)
        print("\n[轮v+] 注入轮子 dof_vel = +3.0 rad/s (LW/RW)")
        print("  轮子正速度 → 前进方向? 后退方向?")
        print_state()
    elif keycode == 86:  # V → 轮子速度 -
        test_idx = 13
        data.qvel[3:6] = 0
        data.qvel[LF_V], data.qvel[RF_V], data.qvel[LR_V], data.qvel[RR_V] = 0, 0, 0, 0
        data.qvel[LW_V], data.qvel[RW_V] = -3.0, -3.0
        mujoco.mj_forward(model, data)
        print("\n[轮v-] 注入轮子 dof_vel = -3.0 rad/s (LW/RW)")
        print("  轮子负速度 → 前进方向? 后退方向?")
        print_state()
    elif keycode == 66:  # B → 停止所有关节速度
        test_idx = 14
        data.qvel[3:] = 0
        mujoco.mj_forward(model, data)
        print("\n[停关节] 所有关节 + gyro 速度清零")
        print_state()
    elif keycode == 85:  # U → LF 单独
        test_idx = 15
        data.qvel[3:] = 0
        data.qvel[LF_V] = joint_vel
        mujoco.mj_forward(model, data)
        print(f"\n[LF] 注入 LF dof_vel = +{joint_vel:.1f} rad/s")
        print("  观察窗口: LF 腿往哪个方向转?")
        print_state()
    elif keycode == 73:  # I → RF 单独
        test_idx = 16
        data.qvel[3:] = 0
        data.qvel[RF_V] = joint_vel
        mujoco.mj_forward(model, data)
        print(f"\n[RF] 注入 RF dof_vel = +{joint_vel:.1f} rad/s")
        print("  观察窗口: RF 腿往哪个方向转?")
        print_state()
    elif keycode == 79:  # O → LR 单独
        test_idx = 17
        data.qvel[3:] = 0
        data.qvel[LR_V] = joint_vel
        mujoco.mj_forward(model, data)
        print(f"\n[LR] 注入 LR dof_vel = +{joint_vel:.1f} rad/s")
        print("  观察窗口: LR 腿往哪个方向转?")
        print_state()
    elif keycode == 80:  # P → RR 单独
        test_idx = 18
        data.qvel[3:] = 0
        data.qvel[RR_V] = joint_vel
        mujoco.mj_forward(model, data)
        print(f"\n[RR] 注入 RR dof_vel = +{joint_vel:.1f} rad/s")
        print("  观察窗口: RR 腿往哪个方向转?")
        print_state()
    elif keycode == 84:  # T → 腿力矩 +
        test_idx = 19
        data.qvel[:] = 0
        data.ctrl = [leg_torque, leg_torque, leg_torque, leg_torque, 0, 0]
        print(f"\n[腿τ+] 注入腿关节力矩 = +{leg_torque:.1f} Nm (LF/RF/LR/RR)")
        print("  正向力矩 → 关节往 dof_pos 增大的方向加速")
        print("  观察窗口: 每条腿往哪个方向摆? (前/后?)")
        print_state()
    elif keycode == 89:  # Y → 腿力矩 -
        test_idx = 20
        data.qvel[:] = 0
        data.ctrl = [-leg_torque, -leg_torque, -leg_torque, -leg_torque, 0, 0]
        print(f"\n[腿τ-] 注入腿关节力矩 = -{leg_torque:.1f} Nm (LF/RF/LR/RR)")
        print("  负向力矩 → 关节往 dof_pos 减小的方向加速")
        print("  观察窗口: 每条腿往哪个方向摆? (前/后?)")
        print_state()
    elif keycode == 71:  # G → 轮力矩 +
        test_idx = 21
        data.qvel[:] = 0
        data.ctrl[LW_C], data.ctrl[RW_C] = wheel_torque, wheel_torque
        data.ctrl[:4] = [0, 0, 0, 0]
        print(f"\n[轮τ+] 注入轮子力矩 = +{wheel_torque:.1f} Nm (LW/RW)")
        print("  轮子正向力矩 → 前进方向? 后退方向?")
        print_state()
    elif keycode == 72:  # H → 轮力矩 -
        test_idx = 22
        data.qvel[:] = 0
        data.ctrl[LW_C], data.ctrl[RW_C] = -wheel_torque, -wheel_torque
        data.ctrl[:4] = [0, 0, 0, 0]
        print(f"\n[轮τ-] 注入轮子力矩 = -{wheel_torque:.1f} Nm (LW/RW)")
        print("  轮子负向力矩 → 前进方向? 后退方向?")
        print_state()
    elif keycode == 74:  # J → LF 力矩单独
        test_idx = 23
        data.qvel[:] = 0
        data.ctrl = [leg_torque, 0, 0, 0, 0, 0]
        print(f"\n[LFτ] 注入 LF 力矩 = +{leg_torque:.1f} Nm")
        print("  观察窗口: LF 腿往哪个方向转?")
        print_state()
    elif keycode == 75:  # K → RF 力矩单独
        test_idx = 24
        data.qvel[:] = 0
        data.ctrl = [0, leg_torque, 0, 0, 0, 0]
        print(f"\n[RFτ] 注入 RF 力矩 = +{leg_torque:.1f} Nm")
        print("  观察窗口: RF 腿往哪个方向转?")
        print_state()
    elif keycode == 76:  # L → LR 力矩单独
        test_idx = 25
        data.qvel[:] = 0
        data.ctrl = [0, 0, leg_torque, 0, 0, 0]
        print(f"\n[LRτ] 注入 LR 力矩 = +{leg_torque:.1f} Nm")
        print("  观察窗口: LR 腿往哪个方向转?")
        print_state()
    elif keycode == 59:  # ; → RR 力矩单独
        test_idx = 26
        data.qvel[:] = 0
        data.ctrl = [0, 0, 0, leg_torque, 0, 0]
        print(f"\n[RRτ] 注入 RR 力矩 = +{leg_torque:.1f} Nm")
        print("  观察窗口: RR 腿往哪个方向转?")
        print_state()
    elif keycode == 87:  # W → yaw +30°
        test_idx = 27
        data.qvel[3:6] = 0
        data.qpos[3:7] = quat_from_yaw(30)
        mujoco.mj_forward(model, data)
        print("\n[yaw+30°] 绕 Z 轴(上) +30°")
        print("  右手拇指指上 → 正=左转")
        print("  观察窗口: 机器人是往左转 还是 往右转?")
        print_state()
    elif keycode == 69:  # E → yaw -30°
        test_idx = 28
        data.qvel[3:6] = 0
        data.qpos[3:7] = quat_from_yaw(-30)
        mujoco.mj_forward(model, data)
        print("\n[yaw-30°] 绕 Z 轴(上) -30°")
        print("  右手拇指指上 → 负=右转")
        print("  观察窗口: 机器人是往左转 还是 往右转?")
        print_state()
    elif keycode == 77:  # M → 切换 PD 控制
        test_idx = 29
        if pd_active:
            pd_active = False
            theta_l, theta_r, L0_l, L0_r, qj, pitch_deg = compute_theta()
            print(f"\n[实物位姿] PD 停止, 冻结当前位置")
            print(f"  关节: LF={np.rad2deg(qj[0]):+.1f}° RF={np.rad2deg(qj[1]):+.1f}° "
                  f"LR={np.rad2deg(qj[2]):+.1f}° RR={np.rad2deg(qj[3]):+.1f}°")
            print(f"  theta=({theta_l:+.2f},{theta_r:+.2f})°  L0=({L0_l:.3f},{L0_r:.3f})m")
            print(f"  实物 theta≈(-56.7°,+35.1°)  L0≈(0.149,0.162)m")
        else:
            pd_active = True
            print(f"\n[实物位姿] 开启 PD 控制")
            print(f"  目标: LF={np.rad2deg(pd_target[0]):+.1f}° RF={np.rad2deg(pd_target[1]):+.1f}° "
                  f"LR={np.rad2deg(pd_target[2]):+.1f}° RR={np.rad2deg(pd_target[3]):+.1f}°")
            print(f"  Kp={pd_kp} Nm/rad  Kd={pd_kd} Nm/(rad/s)")
    elif keycode == 70:  # F → 切换 base 固定
        global base_fixed
        base_fixed = not base_fixed
        print(f"\n[Base] {'固定' if base_fixed else '自由'} (base {'不动' if base_fixed else '可动'})")
        print_state()
    elif keycode == 82:  # R → 完全复位
        test_idx = 0
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        print("\n[复位] 全部状态归零 (qpos / qvel)")
        print_state()
    elif keycode == 81:  # Q
        print("\n退出")
        sys.exit(0)


with mujoco.viewer.launch_passive(model, data,
                                   key_callback=key_callback) as viewer:
    viewer.sync()
    step = 0
    while viewer.is_running():
        # 持续注入速度（mj_step 会因阻尼吃掉注入值，需每步刷新）
        if test_idx == 1:
            data.qvel[3:6] = [vel, 0, 0]
        elif test_idx == 2:
            data.qvel[3:6] = [0, vel, 0]
        elif test_idx == 3:
            data.qvel[3:6] = [0, 0, vel]
        elif test_idx == 10:
            data.qvel[LF_V], data.qvel[RF_V], data.qvel[LR_V], data.qvel[RR_V] = joint_vel, joint_vel, joint_vel, joint_vel
        elif test_idx == 11:
            data.qvel[LF_V], data.qvel[RF_V], data.qvel[LR_V], data.qvel[RR_V] = -joint_vel, -joint_vel, -joint_vel, -joint_vel
        elif test_idx == 12:
            data.qvel[LW_V], data.qvel[RW_V] = 3.0, 3.0
        elif test_idx == 13:
            data.qvel[LW_V], data.qvel[RW_V] = -3.0, -3.0
        elif test_idx == 15:  # LF
            data.qvel[LF_V] = joint_vel
        elif test_idx == 16:  # RF
            data.qvel[RF_V] = joint_vel
        elif test_idx == 17:  # LR
            data.qvel[LR_V] = joint_vel
        elif test_idx == 18:  # RR
            data.qvel[RR_V] = joint_vel
        # 力矩注入: 通过 data.ctrl 持续施加力矩
        elif test_idx == 19:  # 腿τ+
            data.ctrl = [leg_torque, leg_torque, leg_torque, leg_torque, 0, 0]
        elif test_idx == 20:  # 腿τ-
            data.ctrl = [-leg_torque, -leg_torque, -leg_torque, -leg_torque, 0, 0]
        elif test_idx == 21:  # 轮τ+
            data.ctrl = [0, 0, 0, 0, wheel_torque, wheel_torque]
        elif test_idx == 22:  # 轮τ-
            data.ctrl = [0, 0, 0, 0, -wheel_torque, -wheel_torque]
        elif test_idx == 23:  # LFτ
            data.ctrl = [leg_torque, 0, 0, 0, 0, 0]
        elif test_idx == 24:  # RFτ
            data.ctrl = [0, leg_torque, 0, 0, 0, 0]
        elif test_idx == 25:  # LRτ
            data.ctrl = [0, 0, leg_torque, 0, 0, 0]
        elif test_idx == 26:  # RRτ
            data.ctrl = [0, 0, 0, leg_torque, 0, 0]
        # 固定 base: 测速/测力矩时把 base 速度清零
        if base_fixed and test_idx >= 10 and test_idx <= 26:
            data.qvel[0:6] = 0
        # 实物pose模式: PD 控制驱到目标关节角
        if test_idx == 29 and pd_active:
            cur = np.array([data.qpos[LF_P], data.qpos[RF_P],
                            data.qpos[LR_P], data.qpos[RR_P]])
            vel_j = np.array([data.qvel[LF_V], data.qvel[RF_V],
                              data.qvel[LR_V], data.qvel[RR_V]])
            tau = pd_kp * (pd_target - cur) - pd_kd * vel_j
            data.ctrl[LF_C] = tau[0]
            data.ctrl[RF_C] = tau[1]
            data.ctrl[LR_C] = tau[2]
            data.ctrl[RR_C] = tau[3]
            data.ctrl[LW_C] = 0
            data.ctrl[RW_C] = 0
            # 每 200 步打印一次当前状态
            if step % 200 == 0:
                theta_l, theta_r, L0_l, L0_r, qj, pitch_deg = compute_theta()
                err = np.max(np.abs(pd_target - cur))
                print(f"  [PD] step={step} err_max={np.rad2deg(err):+.2f}°  "
                      f"tau=[{tau[0]:+.1f},{tau[1]:+.1f},{tau[2]:+.1f},{tau[3]:+.1f}]Nm  "
                      f"theta=({theta_l:+.2f},{theta_r:+.2f})°  L0=({L0_l:.3f},{L0_r:.3f})m")
            mujoco.mj_step(model, data)
        elif test_idx == 29 and not pd_active:
            # PD 已停, 冻结观察
            data.qvel[:] = 0
            data.ctrl = np.zeros(model.nu)
            mujoco.mj_forward(model, data)
        else:
            mujoco.mj_step(model, data)

        # 每 200 步 (~1s) 打印实际状态, 用于观察阻尼影响
        if test_idx >= 10 and test_idx <= 26 and step % 200 == 0:
            jv = [data.qvel[LF_V], data.qvel[RF_V], data.qvel[LR_V], data.qvel[RR_V],
                  data.qvel[LW_V], data.qvel[RW_V]]
            jp = [data.qpos[LF_P], data.qpos[RF_P], data.qpos[LR_P], data.qpos[RR_P]]
            ct = data.ctrl.copy() if hasattr(data, 'ctrl') and len(data.ctrl) > 0 else None
            print(f"  [step={step}] qpos=[{jp[0]:+.3f},{jp[1]:+.3f},{jp[2]:+.3f},{jp[3]:+.3f}]  "
                  f"qvel=[{jv[0]:+.3f},{jv[1]:+.3f},{jv[2]:+.3f},{jv[3]:+.3f}  LW:{jv[4]:+.3f},RW:{jv[5]:+.3f}]")
        step += 1
        viewer.sync()