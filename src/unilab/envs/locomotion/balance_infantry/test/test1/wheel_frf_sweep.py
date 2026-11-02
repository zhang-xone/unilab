"""轮速环扫频对比：`vofasine.csv`（实机 1→24 Hz 对数扫频）vs 仿真的同指令重放。

之前只用了 `vofasin.csv`（5 Hz 定频）一个点，`vofasine.csv` 才是扫频：
指令 ±5 rad/s，频率从 1.08 Hz 扫到 23.8 Hz。

做法：滑窗局部正弦拟合（窗长 4 个周期），得到每个频点的增益/相位；实机和仿真
用**完全相同**的处理，所以两边的差别只来自被控对象。

注意：本机 mujoco 与 matplotlib 同进程会 segfault，所以拆成
默认（只跑 mujoco，落 CSV）和 `--plot`（只跑 matplotlib）两段。

用法（仓库根目录）:
    MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
        src/unilab/envs/locomotion/balance_infantry/test/test1/wheel_frf_sweep.py
    ... --plot
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_S = None


def _s():
    global _S
    if _S is None:
        import sim_replay_id  # noqa: PLC0415

        _S = sim_replay_id
    return _S


def local_freq(t: np.ndarray, c: np.ndarray) -> np.ndarray:
    """由指令过零间隔插值出瞬时频率。"""
    zc = np.where(np.diff(np.sign(c)) > 0)[0]
    fz = 1.0 / np.diff(t[zc])
    tm = 0.5 * (t[zc[:-1]] + t[zc[1:]])
    return np.interp(t, tm, fz)


def sweep_frf(t: np.ndarray, cmd: np.ndarray, resp: np.ndarray) -> np.ndarray:
    """滑窗正弦拟合，返回 [f, gain, phase_deg]。"""
    f_inst = local_freq(t, cmd)
    rows = []
    step = int(0.1 * 500)
    for i in range(0, len(t), step):
        f = f_inst[i]
        if not (0.5 < f < 30):
            continue
        w = max(1, int(4.0 / f * 500))
        lo, hi = max(0, i - w // 2), min(len(t), i + w // 2)
        tt = t[lo:hi]
        if len(tt) < 20 or len(tt) < 3 * 500 / f:
            continue
        A = np.column_stack([np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt), np.ones_like(tt)])
        cx = np.linalg.lstsq(A, cmd[lo:hi], rcond=None)[0]
        cy = np.linalg.lstsq(A, resp[lo:hi], rcond=None)[0]
        ac, ay = np.hypot(cx[0], cx[1]), np.hypot(cy[0], cy[1])
        if ac < 0.5:
            continue
        g = ay / ac
        ph = (np.degrees(np.arctan2(cy[0], cy[1]) - np.arctan2(cx[0], cx[1])) + 180) % 360 - 180
        rows.append((f, g, ph))
    a = np.array(rows)
    # 按频率分箱平均
    out = []
    for lo in (0.8, 1.2, 1.8, 2.6, 3.8, 5.5, 8.0, 11.0, 16.0, 22.0):
        m = (a[:, 0] >= lo) & (a[:, 0] < lo * 1.35)
        if m.sum():
            out.append((a[m, 0].mean(), a[m, 1].mean(), a[m, 2].mean()))
    return np.array(out)


def main() -> None:
    S = _s()
    p = os.path.join(HERE, "vofasine.csv")
    raw = S.load_csv(p)
    t = raw[:, 0]
    act = np.abs(raw[:, 1]) > 0.5
    print(
        "vofasine: %d 帧  t %.2f..%.2f s  有效 %.1f%%  指令 ±%.1f rad/s"
        % (len(t), t[0], t[-1], 100 * act.mean(), np.abs(raw[:, 1]).max())
    )
    tm, keep = S.monotonic_time(raw[:, 0])
    a = raw[keep]
    tu = S.uniform_time(tm)
    r = S.replay_wheel(p, a, tm, tu)

    for nm in ("real", "sim"):
        z = np.asarray(r[nm])
        tz, cz, yz = z[:, 0], z[:, 1], z[:, 2]
        m = np.abs(cz) > 0.5
        res = sweep_frf(tz[m], cz[m], yz[m])
        np.savetxt(
            os.path.join(HERE, "wheel_sweep_%s.csv" % nm),
            res,
            delimiter=",",
            header="freq_hz,gain,phase_deg",
            comments="",
            fmt="%.4f",
        )
        print("  %s: %d 个频点" % (nm, len(res)))
        for f, g, p_ in res:
            print("     f=%5.2f Hz  增益 %.2f  相位 %+6.1f°" % (f, g, p_))
    print("-> wheel_sweep_real.csv / wheel_sweep_sim.csv")


def plot() -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    re = np.loadtxt(os.path.join(HERE, "wheel_sweep_real.csv"), delimiter=",", skiprows=1)
    si = np.loadtxt(os.path.join(HERE, "wheel_sweep_sim.csv"), delimiter=",", skiprows=1)

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].semilogx(si[:, 0], 20 * np.log10(si[:, 1]), "o-", label="sim")
    ax[0].semilogx(re[:, 0], 20 * np.log10(re[:, 1]), "s-", ms=7, color="k", label="real")
    ax[0].axhline(-3.0, ls="--", c="gray", lw=1)
    ax[0].set_ylabel("gain (dB)")
    ax[0].grid(which="both", alpha=0.3)
    ax[0].legend()
    ax[0].set_title("wheel speed loop FRF: vofasine log sweep 1->24 Hz (+-5 rad/s)")
    ax[1].semilogx(si[:, 0], si[:, 2], "o-", label="sim")
    ax[1].semilogx(re[:, 0], re[:, 2], "s-", ms=7, color="k", label="real")
    ax[1].set_ylabel("phase (deg)")
    ax[1].set_xlabel("freq (Hz)")
    ax[1].grid(which="both", alpha=0.3)
    ax[1].legend()
    fig.tight_layout()
    out = os.path.join(HERE, "cmp_wheel_frf_sweep.png")
    fig.savefig(out, dpi=110)
    print("-> " + out)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    else:
        main()
