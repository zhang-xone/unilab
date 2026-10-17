"""测试 box 基元拼成的金字塔台阶地形."""
import mujoco
import mujoco.viewer
import numpy as np
import os, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..")
SCENE = os.path.join(_PARENT_DIR, "assets", "robots", "balance_infantry", "scene_box_pyramid.xml")

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)

print("=" * 60)
print("  Box 金字塔台阶地形测试")
print("  [W]上坡前进 [S]下坡后退 [A/D]转向")
print("  上金字塔在 x=-5 (中心最高, 外层最低)")
print("  [R]复位 [0]停止 [Q]退出")
print("=" * 60)

joint_names = [
    "left_front_joint", "right_front_joint", "left_rear_joint", "right_rear_joint",
    "left_wheel_joint", "right_wheel_joint",
]
joint_qpos = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
              for n in joint_names]
act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_actuator")
           for n in joint_names]

try:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
except Exception:
    pass

data.ctrl[:] = 0.0

LEG_KP = 100.0
LEG_KD = 5.0
WHEEL_KD = 2.0

cmd_vel = np.array([0.0, 0.0], dtype=np.float64)  # [forward, turn]
cmd_leg = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def controller(m, d):
    q = np.array([d.qpos[jp] for jp in joint_qpos], dtype=np.float64)
    qvel = np.array([d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]]
                     for n in joint_names], dtype=np.float64)

    tau = np.zeros(6, dtype=np.float64)
    tau[:4] = LEG_KP * (cmd_leg - q[:4]) - LEG_KD * qvel[:4]
    tau[4:] = -WHEEL_KD * (qvel[4:] - cmd_vel[0])
    tau[4] -= cmd_vel[1]
    tau[5] += cmd_vel[1]
    d.ctrl[:] = tau


def key_callback(keycode):
    global cmd_vel
    v, w = cmd_vel
    if keycode == 87:    # W
        v += 3.0
    elif keycode == 83:  # S
        v -= 3.0
    elif keycode == 65:  # A
        w += 5.0
    elif keycode == 68:  # D
        w -= 5.0
    elif keycode == 82:  # R
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        data.ctrl[:] = 0.0
    elif keycode == 48:  # 0
        v, w = 0.0, 0.0
    cmd_vel[:] = [v, w]


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    viewer.sync()
    while viewer.is_running():
        controller(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()