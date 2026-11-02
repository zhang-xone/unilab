"""腿长跟踪频率特性：在仿真里对 L 方向做正弦扫频，输出增益/相位。

用法（仓库根目录）:
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
        src/unilab/envs/locomotion/balance_infantry/test/test1/leg_frf_sim.py

为什么单独写：实机那组 `vofalegsin+` 是对 φ0 做 ±0.05 rad 的扫频，幅度正好等于
静摩擦死区（F/kp = 2.0/40 = 0.05 rad），腿全程没动，测不出频率特性。要测腿的
动态，必须在 **L 方向**激励且幅度 ≥ 3 倍死区（L 方向死区 ≈ F/kp * dL/ddof ≈ 10 mm）。
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402

import sim_replay_id as S  # noqa: E402

L1, L2, L3, L4, L5 = S.L1, S.L2, S.L3, S.L4, S.L5
ORDER = ["left_front_joint", "right_front_joint", "left_rear_joint", "right_rear_joint"]
FRONT, REAR = [0, 1], [2, 3]
L0, FREQS = 0.28, (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


def dof_to_L(a: np.ndarray) -> np.ndarray:
    """纯 L 位形：前关节 +a、后关节 -a（dL/da = 0.302 m/rad，φ0 不变）。

    (a, -a) 伸缩腿，(a, a) 只摆 φ0 —— 这是五杆机构的解耦方向，别搞混。
    """
    return np.asarray(
        S.forward_kinematics_fivebar(
            -a + 5 * np.pi / 6, a + np.pi / 6, L1, L2, L3, L4, L5
        )[0]
    ).reshape(-1)


def build_L_map() -> tuple[np.ndarray, np.ndarray]:
    d = np.linspace(-0.40, 0.40, 400)
    return dof_to_L(d), d  # (L, dof) 单调递增


def sim_leg_sine(freq: float, amp: float, dur: float, fs_ctrl: float = 500.0) -> tuple:
    """跑一段 L 正弦，返回 (t, L_cmd, L_act)。吊装夹具 + 重力与实机一致。"""
    model = mujoco.MjModel.from_xml_path(S.SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    model.opt.timestep = S.CTRL_DT / S.SUBSTEPS
    p0, q0 = data.qpos[0:3].copy(), data.qpos[3:7].copy()

    dof = {
        n: model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ORDER
    }
    qadr = {
        n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ORDER
    }
    wheel_dof = [
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in S.WHEELS
    ]

    Lmap, dmap = build_L_map()
    n = int(dur * fs_ctrl)
    t = np.arange(n) / fs_ctrl
    L_cmd = L0 + amp * np.sin(2 * np.pi * freq * t)
    dof_cmd = np.interp(L_cmd, Lmap, dmap)

    out = np.zeros((n, 2))
    out[:, 0] = L_cmd
    for i in range(n):
        tau = np.empty(4)
        for k, name in enumerate(ORDER):
            tgt = dof_cmd[i] if k in FRONT else -dof_cmd[i]
            tau[k] = S.LEG_KP * (tgt - data.qpos[qadr[name]]) - S.LEG_KD * data.qvel[dof[name]]
        tau = np.clip(tau, S.LEG_LO, S.LEG_HI)
        for k, name in enumerate(ORDER):
            data.qfrc_applied[dof[name]] = tau[k]
        for wd in wheel_dof:
            data.qfrc_applied[wd] = -S.WHEEL_KD * data.qvel[wd]

        q = np.array([data.qpos[qadr[name]] for name in ORDER])
        out[i, 1] = np.asarray(
            S.forward_kinematics_fivebar(
                np.array([q[2] + 5 * np.pi / 6]), np.array([q[0] + np.pi / 6]), L1, L2, L3, L4, L5
            )[0]
        ).reshape(-1)[0]

        for _ in range(S.SUBSTEPS):
            S.hold_base(model, data, p0, q0)
            mujoco.mj_step(model, data)
    return t, out[:, 0], out[:, 1]


def main() -> None:
    amp = 0.03  # L 死区约 15 mm 的 2 倍；再小会被静摩擦吃掉
    print(f"L 正弦激励 {L0}±{amp} m（L 方向死区约 0.01 m）")
    print(" f(Hz)   增益   相位(deg)   幅值(mm)   偏差均值(mm)")
    rows = []
    for f in FREQS:
        t, x, y = sim_leg_sine(f, amp, dur=max(6.0 / f, 3.0))
        mask = t > t[-1] * 0.5
        tt, xx, yy = t[mask], x[mask], y[mask]
        # 单频最小二乘取幅值/相位（比 FFT 分箱稳）
        A = np.column_stack([np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt), np.ones_like(tt)])
        cx = np.linalg.lstsq(A, xx, rcond=None)[0]
        cy = np.linalg.lstsq(A, yy, rcond=None)[0]
        ax, ay = np.hypot(cx[0], cx[1]), np.hypot(cy[0], cy[1])
        ph = np.degrees(np.arctan2(cy[0], cy[1]) - np.arctan2(cx[0], cx[1]))
        ph = (ph + 180) % 360 - 180
        rows.append((f, ay / ax, ph))
        print(f"{f:6.2f}  {ay/ax:5.2f}  {ph:8.1f}   {ay*1000:7.2f}   {(yy-xx).mean()*1000:+7.2f}")

    # -3 dB / -90° 交点
    g = np.array([r[1] for r in rows])
    p = np.array([r[2] for r in rows])
    fr = np.array([r[0] for r in rows])
    i3 = np.where(g < g[0] / np.sqrt(2))[0]
    np.savetxt(os.path.join(HERE, "leg_frf_sim.csv"),
               np.column_stack([fr, g, p]), delimiter=",", header="freq_hz,gain,phase_deg", comments="")
    i90 = np.where(p < -90)[0]
    print(
        "-3dB 带宽 ≈ %.2f Hz ；-90° 相位滞后频率 ≈ %.2f Hz"
        % (fr[i3[0]] if len(i3) else float("nan"), fr[i90[0]] if len(i90) else float("nan"))
    )


if __name__ == "__main__":
    main()
