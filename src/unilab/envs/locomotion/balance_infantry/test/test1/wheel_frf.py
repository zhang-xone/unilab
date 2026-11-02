"""轮速环频率特性：仿真扫频 + 实机 5 Hz 正弦对照。

实机只有一组定频正弦（`vofasin.csv`，5 Hz ±5 rad/s），`wheel_slope.csv` 是斜坡（0→30），
拿不到全频段。所以这里在仿真里扫 0.5–20 Hz，再把实机那一个点叠上去。

注意：本机 `mujoco` 与 `matplotlib` 同进程会 segfault（先 mujoco 后 mpl 必崩），
所以 `sim_replay_id`（会 import mujoco）在绘图模式下必须完全不导入，见 `_s()`。

用法（仓库根目录）:
    # 1) 扫频（只跑 mujoco，写 CSV）
    MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
        src/unilab/envs/locomotion/balance_infantry/test/test1/wheel_frf.py
    # 2) 出图（只跑 matplotlib，读 CSV）
    MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
        src/unilab/envs/locomotion/balance_infantry/test/test1/wheel_frf.py --plot
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CTRL_DT = 0.002
SUBSTEPS = 4
FS = 1.0 / CTRL_DT
FREQS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
AMP = 5.0
LEGS = ["left_front_joint", "right_front_joint", "left_rear_joint", "right_rear_joint"]
WHEELS = ["left_wheel_joint", "right_wheel_joint"]
LEG_KP, LEG_KD, WHEEL_KD = 40.0, 2.0, 0.1

_S = None


def _s():
    """延迟导入 sim_replay_id（它 import mujoco，绘图进程里不能碰）。"""
    global _S
    if _S is None:
        import sim_replay_id  # noqa: PLC0415

        _S = sim_replay_id
    return _S


def fit(t: np.ndarray, x: np.ndarray, f: float) -> tuple[float, float]:
    a = np.column_stack(
        [np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)]
    )
    c = np.linalg.lstsq(a, x, rcond=None)[0]
    return float(np.hypot(c[0], c[1])), float(np.arctan2(c[0], c[1]))


def sim_point(m, d, dof, qadr, nom, f: float, amp: float = AMP) -> tuple[float, float]:
    import mujoco  # noqa: PLC0415

    S = _s()
    mujoco.mj_resetDataKeyframe(m, d, 0)
    p0, q0 = d.qpos[0:3].copy(), d.qpos[3:7].copy()
    dur = max(8.0 / f, 4.0)
    n = int(dur * FS)
    t = np.arange(n) / FS
    w_cmd = amp * np.sin(2 * np.pi * f * t)
    wl = np.empty(n)
    for i in range(n):
        for name in LEGS:
            d.qfrc_applied[dof[name]] = (
                -LEG_KP * (d.qpos[qadr[name]] - nom[name]) - LEG_KD * d.qvel[dof[name]]
            )
        tau = np.clip(WHEEL_KD * (w_cmd[i] - d.qvel[dof[WHEELS[0]]]), -4.0, 4.0)
        d.qfrc_applied[dof[WHEELS[0]]] = tau
        d.qfrc_applied[dof[WHEELS[1]]] = tau
        wl[i] = d.qvel[dof[WHEELS[0]]]
        for _ in range(SUBSTEPS):
            S.hold_base(m, d, p0, q0)
            mujoco.mj_step(m, d)
    msk = t > dur * 0.5
    ac, pc = fit(t[msk], w_cmd[msk], f)
    ar, pr = fit(t[msk], wl[msk], f)
    return ar / ac, (np.degrees(pr - pc) + 180) % 360 - 180


def real_point() -> tuple[float, float, float]:
    """实机 5 Hz 正弦（有效窗 6–19 s，前后是无力状态的零值）。"""
    S = _s()
    a = S.load_csv(os.path.join(HERE, "vofasin.csv"))
    t = a[:, 0]
    m = (t >= 6.0) & (t <= 19.0)
    f = 5.0
    ac, pc = fit(t[m], a[m, 1], f)
    ar, pr = fit(t[m], a[m, 2], f)
    return f, ar / ac, (np.degrees(pr - pc) + 180) % 360 - 180


def main() -> None:
    import mujoco  # noqa: PLC0415

    S = _s()
    m = mujoco.MjModel.from_xml_path(S.SCENE)
    d = mujoco.MjData(m)
    m.opt.timestep = CTRL_DT / SUBSTEPS
    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEGS + WHEELS}
    dof = {n: m.jnt_dofadr[j] for n, j in jid.items()}
    qadr = {n: m.jnt_qposadr[j] for n, j in jid.items()}
    mujoco.mj_resetDataKeyframe(m, d, 0)
    nom = {n: float(d.qpos[qadr[n]]) for n in LEGS}

    f_s, g_s, p_s = [], [], []
    for f in FREQS:
        g, p = sim_point(m, d, dof, qadr, nom, f)
        f_s.append(f)
        g_s.append(g)
        p_s.append(p)
        print("  %5.1f Hz  增益 %.2f  相位 %+6.1f°" % (f, g, p), flush=True)

    fr, gr, pr = real_point()
    print("  实机 %4.1f Hz  增益 %.2f  相位 %+6.1f°" % (fr, gr, pr), flush=True)

    np.savetxt(
        os.path.join(HERE, "wheel_frf_sim.csv"),
        np.column_stack([f_s, g_s, p_s]),
        delimiter=",",
        header="freq_hz,gain,phase_deg",
        comments="",
        fmt="%.4f",
    )
    np.savetxt(
        os.path.join(HERE, "wheel_frf_real.csv"),
        np.array([[fr, gr, pr]]),
        delimiter=",",
        header="freq_hz,gain,phase_deg",
        comments="",
        fmt="%.4f",
    )
    print("-> wheel_frf_sim.csv / wheel_frf_real.csv（再用 --plot 出图）")


def plot() -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    d = np.loadtxt(os.path.join(HERE, "wheel_frf_sim.csv"), delimiter=",", skiprows=1)
    r = np.loadtxt(os.path.join(HERE, "wheel_frf_real.csv"), delimiter=",", skiprows=1)
    f_s, g_s, p_s = d[:, 0], d[:, 1], d[:, 2]
    fr, gr, pr = float(r[0]), float(r[1]), float(r[2])

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].semilogx(f_s, 20 * np.log10(g_s), "o-", label="sim")
    ax[0].semilogx([fr], [20 * np.log10(gr)], "s", ms=11, color="k", label="real (5 Hz sine)")
    ax[0].axhline(-3.0, ls="--", c="gray", lw=1)
    ax[0].set_ylabel("gain (dB)")
    ax[0].grid(which="both", alpha=0.3)
    ax[0].legend()
    ax[0].set_title("wheel speed loop FRF: sim sweep vs real (+-5 rad/s)")
    ax[1].semilogx(f_s, p_s, "o-", label="sim")
    ax[1].semilogx([fr], [pr], "s", ms=11, color="k", label="real")
    ax[1].set_ylabel("phase (deg)")
    ax[1].set_xlabel("freq (Hz)")
    ax[1].grid(which="both", alpha=0.3)
    ax[1].legend()
    fig.tight_layout()
    out = os.path.join(HERE, "cmp_wheel_frf.png")
    fig.savefig(out, dpi=110)
    print("-> " + out)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    else:
        main()
