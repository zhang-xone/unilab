"""sim_replay_id.py — 把实机辨识录波的**指令**在 MuJoCo 里重放，产出同格式仿真录波并画对比图。

用法（VS Code 终端里）::

    cd /home/xxx/ubuntu_software/unilab/UniLab
    export UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl
    uv run --no-sync python src/unilab/envs/locomotion/balance_infantry/test/test1/sim_replay_id.py wheel_slope.csv
    uv run --no-sync python src/unilab/envs/locomotion/balance_infantry/test/test1/sim_replay_id.py vofasine.csv --frf
    uv run --no-sync python src/unilab/envs/locomotion/balance_infantry/test/test1/sim_replay_id.py vofagas.csv

产出（与输入 CSV 同目录）::

    sim_<name>.csv       同列格式的仿真录波（8 列轮速帧 / 25 列腿帧）
    cmp_<name>.png       实机 vs 仿真 时域对比（--frf 时多一张频域图）

重放约定（与实机一一对应）:
    * 腿关节 PD  : kp=40, kd=2，力矩钳位 前[-32.7,47.3] / 后[-47.3,32.7]（含气弹簧偏置）
    * 轮关节 PD  : tau = 0.1*(w_cmd - w)，钳位 ±4 N·m
    * 控制周期   : 2 ms（=实机 500 Hz 录波），物理子步 0.5 ms
    * 吊起工况   : 关重力，四条腿用 PD 保持在标称位形（= 实机轮子测试的吊装状态）
    * 仿真**不施加**气弹簧恒定力矩（实机是 gas - PD），这正是要对比出来的差异之一
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402

from unilab.envs.locomotion.balance_infantry.base import forward_kinematics_fivebar  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 7))   # -> 仓库根 UniLab/
SCENE = os.path.join(ROOT, "src/unilab/assets/robots/balance_infantry/scene_flat.xml")

LEG_KP, LEG_KD = 40.0, 2.0
WHEEL_KD = 0.1
CTRL_DT = 0.002          # 500 Hz，与实机录波一致
SUBSTEPS = 4             # 物理步长 0.5 ms
# 与 robot.xml 一致：电机 ±40 叠加气弹簧后的非对称饱和
LEG_LO = np.array([-32.7, -32.7, -47.3, -47.3])
LEG_HI = np.array([47.3, 47.3, 32.7, 32.7])
GAS_BIAS = 7.3           # 只用于力矩钳位，不施加

LEGS = ["left_front_joint", "left_rear_joint", "right_front_joint", "right_rear_joint"]
WHEELS = ["left_wheel_joint", "right_wheel_joint"]
L1, L2, L3, L4, L5 = 0.215, 0.254, 0.254, 0.215, 0.0


# ----------------------------------------------------------------------------
HOLD_KP, HOLD_KD, HOLD_KR, HOLD_KDR = 4000.0, 400.0, 200.0, 20.0


def hold_base(model, data, p0: np.ndarray, q0: np.ndarray) -> None:
    """模拟"吊起来"的夹具：在 root 的 6 个 dof 上加一个弹簧-阻尼，把机体近似吊住。

    这样重力照常作用，腿和轮子能像实机吊装时一样自由摆动（零重力是做不到的）。
    """
    rot = np.zeros(3)
    mujoco.mju_subQuat(rot, q0, data.qpos[3:7])
    data.qfrc_applied[0:3] = HOLD_KP * (p0 - data.qpos[0:3]) - HOLD_KD * data.qvel[0:3]
    data.qfrc_applied[3:6] = HOLD_KR * rot - HOLD_KDR * data.qvel[3:6]


def load_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def monotonic_time(t: np.ndarray):
    """把录波时间轴修成严格单调，并丢掉重复帧。

    三种坏帧分别处理（否则会把时间轴拉长，重放出来就像"仿真有固定延时"）：
      * 重复帧 (|dt| < 半周期)  —— 丢帧，不加时间（此前的实现给它们各补一个周期，
        于是 16399 帧的轮速斜坡被拉长成 32.8 s，而真实只有 29.9 s，看图像是延时）；
      * 回绕 (dt < 0)          —— 分段重新计时，接一个周期；
      * 大缺口 (dt > 20 周期)  —— 同样接一个周期。
    """
    dt_med = float(np.median(np.diff(t)))
    d = np.diff(t)
    dup = np.abs(d) < 0.5 * dt_med
    bridge = (d < -0.5 * dt_med) | (d > 20.0 * dt_med)
    step = d.copy()
    step[bridge] = dt_med
    step[dup] = 0.0
    tm_all = np.concatenate([[t[0]], t[0] + np.cumsum(step)])
    keep = np.concatenate([[True], ~dup])
    return tm_all[keep], keep


def uniform_time(tm: np.ndarray, fs: float = 1.0 / CTRL_DT) -> np.ndarray:
    n = int(round((tm[-1] - tm[0]) * fs)) + 1
    return tm[0] + np.arange(n) / fs


# ----------------------------------------------------------------------------
def replay_wheel(csv_path: str, w_real: np.ndarray, tm: np.ndarray, tu: np.ndarray) -> dict:
    """8 列轮速帧：I1=w_cmd, I2/I5=w_act, I3/I6=tau。"""
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    m.opt.timestep = CTRL_DT / SUBSTEPS
    p0, q0 = d.qpos[0:3].copy(), d.qpos[3:7].copy()

    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEGS + WHEELS}
    dof = {n: m.jnt_dofadr[j] for n, j in jid.items()}
    qadr = {n: m.jnt_qposadr[j] for n, j in jid.items()}
    nom = {n: float(d.qpos[qadr[n]]) for n in LEGS}

    w_cmd = np.interp(tu, tm, w_real[:, 1])
    n = len(tu)
    out = np.zeros((n, 8))
    out[:, 0] = tu
    out[:, 1] = w_cmd

    for i in range(n):
        # 腿：PD 保持标称位形（吊装时腿不动）
        for name in LEGS:
            sign = 0.0
            d.qfrc_applied[dof[name]] = -LEG_KP * (d.qpos[qadr[name]] - nom[name]) - LEG_KD * d.qvel[dof[name]] + sign
        # 轮：速度环
        tau = np.empty(2)
        for k, name in enumerate(WHEELS):
            tau[k] = WHEEL_KD * (w_cmd[i] - d.qvel[dof[name]])
        tau = np.clip(tau, -4.0, 4.0)
        for k, name in enumerate(WHEELS):
            d.qfrc_applied[dof[name]] = tau[k]

        out[i, 2] = d.qvel[dof[WHEELS[0]]]
        out[i, 3] = tau[0]
        out[i, 4] = -tau[0]
        out[i, 5] = d.qvel[dof[WHEELS[1]]]
        out[i, 6] = tau[1]
        out[i, 7] = -tau[1]

        for _ in range(SUBSTEPS):
            hold_base(m, d, p0, q0)
            mujoco.mj_step(m, d)

    return {"sim": out, "real": w_real}


def replay_leg(csv_path: str, a_real: np.ndarray, tm: np.ndarray, tu: np.ndarray,
               gas: float = 0.0) -> dict:
    """25 列腿帧：I1..I4=target_pos(LF,RF,LR,RR), I5..I8=dof_pos。"""
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    m.opt.timestep = CTRL_DT / SUBSTEPS
    p0, q0 = d.qpos[0:3].copy(), d.qpos[3:7].copy()

    order = ["left_front_joint", "right_front_joint", "left_rear_joint", "right_rear_joint"]
    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in order + WHEELS}
    dof = {n: m.jnt_dofadr[j] for n, j in jid.items()}
    qadr = {n: m.jnt_qposadr[j] for n, j in jid.items()}
    nom = {n: float(d.qpos[qadr[n]]) for n in order}


    tgt = np.column_stack([np.interp(tu, tm, a_real[:, 1 + k]) for k in range(4)])
    n = len(tu)
    out = np.zeros((n, 25))
    out[:, 0] = tu
    out[:, 1:5] = tgt

    for i in range(n):
        tau = np.empty(4)
        for k, name in enumerate(order):
            tau[k] = LEG_KP * (tgt[i, k] - d.qpos[qadr[name]]) - LEG_KD * d.qvel[dof[name]]
        # gas=0 时与当前 env 一致：PD 直接按非对称范围钳位（气弹簧只体现为钳位偏置）。
        # gas=7.3 时额外加恒定的气弹簧前馈（前 +、后 −，符号已用失重实验确认是"伸腿"），
        # 并按实机的 ±40 钳位，用来 A/B 对比"补上气弹簧"能贴近多少。
        if gas != 0.0:
            bias = gas * np.array([1.0, 1.0, -1.0, -1.0])
            tau = np.clip(tau + bias, -40.0, 40.0)
        else:
            tau = np.clip(tau, LEG_LO, LEG_HI)
        for k, name in enumerate(order):
            d.qfrc_applied[dof[name]] = tau[k]
        for name in WHEELS:
            d.qfrc_applied[dof[name]] = -WHEEL_KD * d.qvel[dof[name]]

        q = np.array([d.qpos[qadr[name]] for name in order])
        qd = np.array([d.qvel[dof[name]] for name in order])
        out[i, 5:9] = q
        out[i, 9:13] = qd
        out[i, 13:17] = tau
        # L / phi0（左腿：前=LF, 后=LR；右腿：前=RF, 后=RR）
        for k, (front, rear) in enumerate([(0, 2), (1, 3)]):
            o = forward_kinematics_fivebar(
                np.array([[q[rear] + 5 * np.pi / 6]]), np.array([[q[front] + np.pi / 6]]),
                L1, L2, L3, L4, L5,
            )
            out[i, 17 + k] = o[0][0, 0]
            out[i, 19 + k] = o[1][0, 0]
        # 气弹簧：仿真**未施加**，这里记录固件模型会算出的常值 ±7.3（仅作参考列）
        out[i, 21], out[i, 23] = -GAS_BIAS, -GAS_BIAS
        out[i, 22], out[i, 24] = GAS_BIAS, GAS_BIAS

        for _ in range(SUBSTEPS):
            hold_base(m, d, p0, q0)
            mujoco.mj_step(m, d)
    return {"sim": out, "real": a_real}


# ----------------------------------------------------------------------------
def plot_time(sim: np.ndarray, real: np.ndarray, kind: str, out_png: str, title: str) -> None:
    """每个信号单独一个子图 + 残差子图，避免多条曲线挤在一起看不清。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_r, t_s = real[:, 0], sim[:, 0]
    C_REAL, C_SIM, C_CMD = "#1f77b4", "#d62728", "#7f7f7f"

    def three(ax, c_cmd, c_real, c_sim, ylabel, names):
        # 仿真先画成粗的浅色底线，实机再细线覆盖：两条重合时也能同时看见
        ax.plot(t_s, sim[:, c_sim], color=C_SIM, lw=2.6, alpha=0.55, label=f"sim {names}")
        if c_cmd is not None:
            ax.plot(t_r, real[:, c_cmd], color=C_CMD, lw=1.0, ls=":", label=f"cmd {names}")
        ax.plot(t_r, real[:, c_real], color=C_REAL, lw=1.2, label=f"real {names}")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    if kind == "wheel":
        rows = [(1, 2, "wheel speed [rad/s]", "LW", "w_cmd"),
                (1, 5, "wheel speed [rad/s]", "RW", "w_cmd"),
                (3, 3, "torque [N·m]", "LW", None),
                (6, 6, "torque [N·m]", "RW", None)]
    else:
        rows = [(1, 5, "LF [rad]", "LF", "target"), (2, 6, "RF [rad]", "RF", "target"),
                (3, 7, "LR [rad]", "LR", "target"), (4, 8, "RR [rad]", "RR", "target"),
                (13, 13, "PD torque [N·m]", "LF", None), (15, 15, "PD torque [N·m]", "LR", None)]

    n = len(rows) + 2
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.9 * n), sharex=True)
    for ax, (c_cmd, c_real, ylabel, nm, cname) in zip(axes, rows):
        three(ax, c_cmd if cname else None, c_real, c_real, ylabel, nm)

    if kind == "leg":
        ax = axes[-2]
        ax.clear()
        ax.plot(t_r, real[:, 17], color=C_REAL, lw=1.3, label="real L_left")
        ax.plot(t_s, sim[:, 17], color=C_SIM, lw=1.3, ls="--", label="sim L_left")
        ax.set_ylabel("leg length L [m]", fontsize=8)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=7, loc="lower right")
        ax = axes[-1]
        ax.clear()
        ax.plot(t_r, real[:, 17] - np.interp(t_r, t_s, sim[:, 17]),
                color="#2ca02c", lw=1.2, label="real − sim")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("ΔL [m]", fontsize=8)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=7, loc="lower right")
    else:
        ax = axes[-2]
        ax.clear()
        ax.plot(t_r, real[:, 2] - np.interp(t_r, t_s, sim[:, 2]), color="#2ca02c", lw=1.2, label="real − sim  (LW)")
        ax.plot(t_r, real[:, 5] - np.interp(t_r, t_s, sim[:, 5]), color="#9467bd", lw=1.2, label="real − sim  (RW)")
        ax.set_ylabel("Δw [rad/s]", fontsize=8)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=7, loc="lower right")
        ax = axes[-1]
        ax.clear()
        ax.plot(t_r, real[:, 3] - np.interp(t_r, t_s, sim[:, 3]), color="#2ca02c", lw=1.2, label="Δτ (LW)")
        ax.plot(t_r, real[:, 6] - np.interp(t_r, t_s, sim[:, 6]), color="#9467bd", lw=1.2, label="Δτ (RW)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("Δτ [N·m]", fontsize=8)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=7, loc="lower right")

    axes[-1].set_xlabel("t [s]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("  ->", out_png)


def plot_frf(sim: np.ndarray, real: np.ndarray, kind: str, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import id_analyze as ia

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    if kind == "wheel":
        pairs = [("left", 1, 2), ("right", 1, 5)]
    else:
        pairs = [("phi0", None, 19)]
    for name, ci, ai in pairs:
        for label, arr, style in (("real", real, "-o"), ("sim", sim, "--s")):
            if ci is None:
                # 腿：激励 = 指令 target 的 φ0（前=LF 列1/2，后=LR 列3/4），响应 = 实测/仿真 φ0
                tgt_front, tgt_rear = (1, 3) if label == "real" else (1, 3)
                p0c = np.asarray(forward_kinematics_fivebar(
                    arr[:, tgt_rear] + 5 * np.pi / 6, arr[:, tgt_front] + np.pi / 6,
                    L1, L2, L3, L4, L5)[1]).reshape(-1)
                y = arr[:, ai]
            else:
                p0c, y = arr[:, ci], arr[:, ai]
            fs = 1.0 / np.median(np.diff(arr[:, 0]))
            f, G, _, _ = ia.frf_sweep_vec(p0c, y, arr[:, 0], fs)
            if f.size < 5:
                continue
            ax1.semilogx(f, np.abs(G), style, ms=3, lw=1.0, label=f"{name}-{label}")
            ax2.semilogx(f, np.degrees(np.angle(G)), style, ms=3, lw=1.0, label=f"{name}-{label}")
    for ax, yl in ((ax1, "|G|"), (ax2, "phase [deg]")):
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print("  ->", out_png)


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="test1/ 下的实机录波文件名或路径")
    ap.add_argument("--frf", action="store_true", help="额外输出频域对比图")
    ap.add_argument("--gas", type=float, default=0.0,
                    help="腿气弹簧前馈力矩幅值 N·m（默认 0 = 当前仿真模型；实机约 7.3）")
    args = ap.parse_args()

    path = args.csv if os.path.isabs(args.csv) or os.path.exists(args.csv) else os.path.join(HERE, args.csv)
    name = os.path.splitext(os.path.basename(path))[0]
    raw = load_csv(path)
    kind = "wheel" if raw.shape[1] == 8 else "leg"
    tm, keep = monotonic_time(raw[:, 0])
    a = raw[keep]
    tu = uniform_time(tm)
    print(f"重放 {name}: {raw.shape[0]} 行 -> 去重 {len(a)} 行 -> {len(tu)} 点 @500 Hz, 模式={kind}")

    res = replay_wheel(path, a, tm, tu) if kind == "wheel" else replay_leg(path, a, tm, tu, args.gas)
    tag = name if args.gas == 0.0 else f"{name}_gas{args.gas:g}"
    out_csv = os.path.join(os.path.dirname(path), f"sim_{tag}.csv")
    np.savetxt(out_csv, res["sim"], delimiter=",", fmt="%.6f",
               header=",".join(f"c{i}" for i in range(res["sim"].shape[1])), comments="")
    print("  ->", out_csv)

    plot_time(res["sim"], a, kind, os.path.join(os.path.dirname(path), f"cmp_{tag}.png"), tag)
    if args.frf:
        plot_frf(res["sim"], a, kind, os.path.join(os.path.dirname(path), f"cmp_{tag}_frf.png"))

    # 实机帧率与仿真不同（125~500 Hz），对比前把实机插值到仿真时间轴上
    s_ = res["sim"]
    r_ = np.column_stack([np.interp(s_[:, 0], tm, a[:, c]) for c in range(a.shape[1])])
    n = len(s_)
    if kind == "wheel":
        print("\n稳态对比 (末 20%):")
        for tag_, arr in (("real", r_), ("sim ", s_)):
            sl = slice(int(0.8 * n), None)
            print(f"  {tag_}: w_L={arr[sl,2].mean():6.2f}  tau_L={arr[sl,3].mean():.4f}  "
                  f"w_R={arr[sl,5].mean():6.2f}  tau_R={arr[sl,6].mean():.4f}")
    else:
        print("\n整段对比:")
        for tag_, arr in (("real", r_), ("sim ", s_)):
            L = arr[:, 17:19]
            print(f"  {tag_}: L={L.min():.4f}..{L.max():.4f} (mean {L.mean():.4f})  "
                  f"max|dof|={np.abs(arr[:,5:9]).max():.3f} rad  max|tau|={np.abs(arr[:,13:17]).max():.1f}")
        print(f"  dof rms(real-sim) = {np.sqrt(((r_[:,5:9]-s_[:,5:9])**2).mean()):.3f} rad   "
              f"L rms = {np.sqrt(((r_[:,17:19]-s_[:,17:19])**2).mean()):.4f} m")


if __name__ == "__main__":
    main()
