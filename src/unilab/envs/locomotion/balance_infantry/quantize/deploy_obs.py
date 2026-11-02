#!/usr/bin/env python3
"""部署侧 29 维观测构建 —— 量化 / 导出 / 仿真测试的唯一真源.

与训练环境保持一致 (``src/unilab/envs/locomotion/balance_infantry/joystick.py``):

    obs = [gyro(3), pitch, roll, cmd(vx, vyaw, height, jump)(4), last_actions(6),
           dof_pos_err(4), dof_vel(6), theta_l, theta_r, L0_l, L0_r]

关键约定（这几条曾经写错过，改动前先对照 ``joystick.py`` / ``joystick_vmc.py``）:

* ``pitch = asin(+gx)``、``roll = asin(-gy)``，``gx = 2(xz-wy)``、``gy = 2(yz+wx)``；
* ``dof_pos_err = dof_pos - default_dof_pos``（不是 ``default - dof_pos``）；
* ``theta = pi/2 - phi0 + pitch``（不是 ``- pitch``）；
* ``phi1 = (q_LR - default_LR) + 5*pi/6``、``phi4 = (q_LF - default_LF) + pi/6``；
* ``cmd[3]`` 是 jump 标志，不是 standing。

所有需要构建部署观测的脚本都应调用 :func:`build_observation`，不要在各自文件里
再复制一份公式。
"""

import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LQR_DIR = os.path.join(_SCRIPT_DIR, "..", "lqr")
for _p in (_SCRIPT_DIR, _LQR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_vmc_lqr import fk_fivebar  # noqa: E402

# 观测维度 / clip / 归一化 eps / 训练噪声全部来自训练产物。
# 训练后运行 quantize/sync_trained_params.py 覆盖 quantize/trained_params.py。
from trained_params import (  # noqa: E402
    NORM_EPS,
    NUM_ACTIONS,
    NUM_OBS,
    OBS_CLIP,
    TRAINING_OBS_NOISE,
)


def quat_pitch_roll(quat):
    """四元数 (w, x, y, z) → (pitch, roll)，与 env ``_compute_obs`` 同式同号。"""
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    gx = 2.0 * (x * z - w * y)
    gy = 2.0 * (y * z + w * x)
    pitch = float(np.arcsin(np.clip(gx, -1.0, 1.0)))
    roll = float(np.arcsin(np.clip(-gy, -1.0, 1.0)))
    return pitch, roll


def leg_kinematics(dof_pos, default_dof_pos):
    """五连杆正解 → ``(L0_l, phi0_l, L0_r, phi0_r)``，与 env ``_update_vmc_state`` 一致。"""
    q = np.asarray(dof_pos, dtype=np.float64).reshape(-1)
    d = np.asarray(default_dof_pos, dtype=np.float64).reshape(-1)
    cal_lf, cal_rf, cal_lr, cal_rr = (q[0] - d[0], q[1] - d[1], q[2] - d[2], q[3] - d[3])

    phi1_l = cal_lr + 5.0 * np.pi / 6.0
    phi4_l = cal_lf + np.pi / 6.0
    phi1_r = cal_rr + 5.0 * np.pi / 6.0
    phi4_r = cal_rf + np.pi / 6.0

    L0_l, phi0_l, _, _, _ = fk_fivebar(phi1_l, phi4_l)
    L0_r, phi0_r, _, _, _ = fk_fivebar(phi1_r, phi4_r)
    return float(L0_l), float(phi0_l), float(L0_r), float(phi0_r)


# 训练侧实际生效的观测噪声来自 trained_params.TRAINING_OBS_NOISE。
# 均匀分布 ±(level × scale)，加在**归一化之前**；与 env `_compute_obs` 同语义。
# 注意: θ̇ / L̇₀ 不在 29 维观测里，因此没有对应的噪声项。
_NOISE_SLICES = {
    "gyro": slice(0, 3),
    "gravity": slice(3, 5),        # pitch, roll
    "joint_angle": slice(15, 19),  # dof_pos_err
    "joint_vel": slice(19, 25),    # dof_vel
    "vmc_phi0": slice(25, 27),     # theta_l, theta_r
    "vmc_L0": slice(27, 29),       # L0_l, L0_r
}


def add_observation_noise(obs, level=1.0, scales=None, rng=None):
    """给 29 维原始观测加观测噪声，返回新数组（不修改入参）。

    Args:
        obs: build_observation() 的输出 (29,)
        level: 噪声倍率，1.0 = 训练同分布，0 = 关闭
        scales: 覆盖默认 scale（默认用 TRAINING_OBS_NOISE）
        rng: numpy Generator，便于复现
    """
    out = np.asarray(obs, dtype=np.float32).copy()
    if level <= 0.0:
        return out
    scales = dict(TRAINING_OBS_NOISE if scales is None else scales)
    rng = np.random.default_rng() if rng is None else rng
    for key, sl in _NOISE_SLICES.items():
        n = out[sl].size
        out[sl] += rng.uniform(-1.0, 1.0, n).astype(np.float32) * float(level) * scales[key]
    return out


def build_observation(
    gyro,
    quat,
    dof_pos,
    dof_vel,
    last_actions,
    cmd_vel,
    cmd_yaw_rate,
    cmd_height,
    default_dof_pos,
    cmd_jump=0.0,
):
    """构建 29 维原始观测（未归一化），与训练 env 逐项对齐。"""
    pitch, roll = quat_pitch_roll(quat)
    L0_l, phi0_l, L0_r, phi0_r = leg_kinematics(dof_pos, default_dof_pos)
    theta_l = (np.pi / 2.0 - phi0_l + pitch + np.pi) % (2.0 * np.pi) - np.pi
    theta_r = (np.pi / 2.0 - phi0_r + pitch + np.pi) % (2.0 * np.pi) - np.pi

    dof_pos_arr = np.asarray(dof_pos, dtype=np.float64).reshape(-1)
    default_arr = np.asarray(default_dof_pos, dtype=np.float64).reshape(-1)
    dof_pos_err = dof_pos_arr[:4] - default_arr[:4]

    obs = np.concatenate([
        np.asarray(gyro, dtype=np.float64).reshape(3),
        np.array([pitch, roll], dtype=np.float64),
        np.array([cmd_vel, cmd_yaw_rate, cmd_height, cmd_jump], dtype=np.float64),
        np.asarray(last_actions, dtype=np.float64).reshape(6),
        dof_pos_err,
        np.asarray(dof_vel, dtype=np.float64).reshape(6),
        np.array([theta_l, theta_r, L0_l, L0_r], dtype=np.float64),
    ]).astype(np.float32)

    if obs.shape != (NUM_OBS,):
        raise ValueError(f"观测维度应为 {NUM_OBS}, 实际 {obs.shape}")
    return obs