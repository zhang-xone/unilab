"""复刻实机测试链路：VMC 目标 (L, φ0) → 逆解 → 关节目标 → PD。

实机那几组测试（vofagas / vofalegsin+ / vofasin）都是先在任务空间给 (腿长 L, 摆角 φ0)，
再用五杆逆解换成四个关节目标。本脚本在仿真里做同一件事：

  1. 验证逆解：用录波的 (L, φ0) 反解，能否还原录波的关节目标（误差应 ~1e-14）；
  2. 用 VMC 链路重放 vofalegsin+，与直接喂关节目标的结果对照（应完全一致）；
  3. 把"φ0 正弦扫频"当成干净的测试信号重跑，扫描不同幅度，看多大的激励才推得动
     腿（实机 ±0.05 rad 那组全程没动）。

用法（仓库根目录）:
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
        src/unilab/envs/locomotion/balance_infantry/test/test1/vmc_replay.py
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
FS = 1.0 / S.CTRL_DT


def fk_dof(qf: float, qr: float) -> tuple[float, float]:
    o = S.forward_kinematics_fivebar(
        np.array([qr + 5 * np.pi / 6]), np.array([qf + np.pi / 6]), L1, L2, L3, L4, L5
    )
    return float(np.asarray(o[0]).reshape(-1)[0]), float(np.asarray(o[1]).reshape(-1)[0])


def ik(Lt: float, p0t: float, seed: tuple[float, float] = (0.0, 0.0), iters: int = 40):
    """五杆逆解：(L, φ0) → (前关节, 后关节)。工作空间内从 (0,0) 起解唯一。"""
    qf, qr = seed
    h = 1e-6
    for _ in range(iters):
        L0, p0 = fk_dof(qf, qr)
        err = np.array([L0 - Lt, p0 - p0t])
        if np.abs(err).max() < 1e-12:
            break
        jac = np.array(
            [
                [
                    (fk_dof(qf + h, qr)[0] - fk_dof(qf - h, qr)[0]) / (2 * h),
                    (fk_dof(qf, qr + h)[0] - fk_dof(qf, qr - h)[0]) / (2 * h),
                ],
                [
                    (fk_dof(qf + h, qr)[1] - fk_dof(qf - h, qr)[1]) / (2 * h),
                    (fk_dof(qf, qr + h)[1] - fk_dof(qf, qr - h)[1]) / (2 * h),
                ],
            ]
        )
        dq = np.linalg.solve(jac, -err)
        qf, qr = qf + dq[0], qr + dq[1]
    return qf, qr


def ik_traj(L_cmd: np.ndarray, p0_cmd: np.ndarray) -> np.ndarray:
    """整段逆解，逐帧以上一帧为种子（保证不走分支）。"""
    out = np.zeros((len(L_cmd), 2))
    seed = ik(float(L_cmd[0]), float(p0_cmd[0]))
    for i, (lo, p0) in enumerate(zip(L_cmd, p0_cmd)):
        seed = ik(float(lo), float(p0), seed)
        out[i] = seed
    return out


def sim_run(targets: np.ndarray, soft: np.ndarray | None = None, gas: float = 0.0,
            play: float = 0.0, delay_steps: int = 0) -> dict:
    """targets: (n, 2) 的 [前关节, 后关节] 目标。吊装夹具 + 重力，与实机吊装一致。

    soft: 力矩软启动系数（对应固件 SOFT_START_MS），None = 1。
    """
    model = mujoco.MjModel.from_xml_path(S.SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    model.opt.timestep = S.CTRL_DT / S.SUBSTEPS
    p0, q0 = data.qpos[0:3].copy(), data.qpos[3:7].copy()

    dof = {n: model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ORDER}
    qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ORDER}
    wheel_dof = [
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in S.WHEELS
    ]

    n = len(targets)
    out = np.zeros((n, 2))
    pstate = np.array([targets[0, 0], targets[0, 0], targets[0, 1], targets[0, 1]], dtype=float)
    # 纯传输延时：目标 FIFO 延迟 delay_steps 个控制周期（2 ms/步），模拟 CAN+驱动器延时
    fifo = np.repeat(targets[:1, :], max(delay_steps, 1), axis=0)
    q_hist = np.zeros((n, 4))
    qd_hist = np.zeros((n, 4))
    tau_hist = np.zeros((n, 4))
    for i in range(n):
        if delay_steps > 0:
            fifo = np.vstack([fifo, targets[i:i + 1, :]])
            src = fifo[0]
            fifo = fifo[1:]
        else:
            src = targets[i]
        tgt = np.array([src[0], src[0], src[1], src[1]])  # LF,RF 前 / LR,RR 后
        if play > 0.0:
            # 齿隙（回差）：目标必须先走完 ±play/2 的自由行程，输出才跟上。
            # 这是"响应更快但尖峰被削"的经典成因：反向时先留一段不动，
            # 之后以目标的速度追上，等效相位超前 + 幅值压缩。
            d = tgt - pstate
            over = np.abs(d) - play / 2
            pstate = np.where(over > 0, tgt - np.sign(d) * play / 2, pstate)
            tgt = pstate.copy()
        tau = np.empty(4)
        for k, name in enumerate(ORDER):
            tau[k] = S.LEG_KP * (tgt[k] - data.qpos[qadr[name]]) - S.LEG_KD * data.qvel[dof[name]]
        if gas:
            tau = tau + np.array([gas, gas, -gas, -gas])  # 前 +g / 后 -g（伸腿方向）
        if soft is not None:
            tau = tau * soft[i]
        tau = np.clip(tau, S.LEG_LO, S.LEG_HI)
        for k, name in enumerate(ORDER):
            data.qfrc_applied[dof[name]] = tau[k]
        for wd in wheel_dof:
            data.qfrc_applied[wd] = -S.WHEEL_KD * data.qvel[wd]

        q = np.array([data.qpos[qadr[name]] for name in ORDER])
        out[i] = fk_dof(q[0], q[2])
        q_hist[i] = q
        qd_hist[i] = [data.qvel[dof[name]] for name in ORDER]
        tau_hist[i] = tau

        for _ in range(S.SUBSTEPS):
            S.hold_base(model, data, p0, q0)
            mujoco.mj_step(model, data)
    return {"L": out[:, 0], "phi0": out[:, 1], "q": q_hist, "qd": qd_hist, "tau": tau_hist}


def check_ik_roundtrip(csv: str) -> None:
    """录波的关节目标 → FK 得 (L, φ0) → IK 回关节目标，验证逆解与固件一致。"""
    raw = S.load_csv(os.path.join(HERE, csv))
    tm, keep = S.monotonic_time(raw[:, 0])
    a = raw[keep]
    qf, qr = a[:, 1], a[:, 3]  # 目标列 I1=LF(front), I3=LR(rear)
    L = np.empty(len(a))
    p0 = np.empty(len(a))
    for i in range(len(a)):
        L[i], p0[i] = fk_dof(qf[i], qr[i])
    back = ik_traj(L, p0)
    print(
        "  %-18s 逆解回代误差 max %.2e / %.2e rad"
        % (os.path.basename(csv), np.abs(back[:, 0] - qf).max(), np.abs(back[:, 1] - qr).max())
    )


def main() -> None:
    print("== 1. 逆解一致性（录波关节目标 ↔ IK）==")
    for f in ("vofagas.csv", "vofalegsin+.csv"):
        check_ik_roundtrip(f)

    print("\n== 2. VMC 链路重放 vs 直接喂关节目标（vofalegsin+）==")
    raw = S.load_csv(os.path.join(HERE, "vofalegsin+.csv"))
    tm, keep = S.monotonic_time(raw[:, 0])
    a = raw[keep]
    tu = S.uniform_time(tm)
    qf = np.interp(tu, tm, a[:, 1])
    qr = np.interp(tu, tm, a[:, 3])
    L_cmd = np.array([fk_dof(x, y)[0] for x, y in zip(qf, qr)])
    p_cmd = np.array([fk_dof(x, y)[1] for x, y in zip(qf, qr)])
    targets = ik_traj(L_cmd, p_cmd)
    r_vmc = sim_run(targets)
    print(
        "  IK 反解与录波目标的最大差: %.2e rad ；VMC 重放 L 范围 %.4f..%.4f"
        % (np.abs(targets[:, 0] - qf).max(), r_vmc["L"].min(), r_vmc["L"].max())
    )

    print("\n== 3. φ0 扫频：多大激励才推得动腿 ==")
    print("   L=0.2987, φ0 = π/2 + A·sin(2πft)，看实际摆角幅值（rad）")
    print("   幅度A   0.5Hz   1Hz    2Hz    5Hz    10Hz")
    L0, p_mid = 0.2987, np.pi / 2
    for amp in (0.05, 0.10, 0.20):
        row = []
        for f in (0.5, 1.0, 2.0, 5.0, 10.0):
            dur = max(6.0 / f, 3.0)
            t = np.arange(int(dur * FS)) / FS
            p_cmd = p_mid + amp * np.sin(2 * np.pi * f * t)
            tg = ik_traj(np.full_like(t, L0), p_cmd)
            r = sim_run(tg)
            m = t > t[-1] * 0.5
            tt, yy = t[m], r["phi0"][m]
            A = np.column_stack([np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt), np.ones_like(tt)])
            c = np.linalg.lstsq(A, yy, rcond=None)[0]
            row.append(np.hypot(c[0], c[1]))
        print("   %5.2f  " % amp + "  ".join("%.3f" % v for v in row))


if __name__ == "__main__":
    main()
