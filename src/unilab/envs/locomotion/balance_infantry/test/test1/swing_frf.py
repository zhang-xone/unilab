"""摆动（φ0）方向的频率特性：仿真正弦扫频 + Bode。

实机那组 `vofalegsin+` 是 φ0 ±0.05 rad，正好等于静摩擦死区，腿全程没动，
测不出摆动特性。这里在仿真里用足够大的幅度（±0.1 / ±0.2 rad）扫频，
给出增益/相位，作为下一轮实机测试的参照。

用法（仓库根目录）:
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
        src/unilab/envs/locomotion/balance_infantry/test/test1/swing_frf.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vmc_replay as V  # noqa: E402

FS = 1.0 / V.S.CTRL_DT
FREQS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


def fit_sine(t: np.ndarray, y: np.ndarray, f: float) -> tuple[float, float]:
    a = np.column_stack(
        [np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)]
    )
    c = np.linalg.lstsq(a, y, rcond=None)[0]
    return float(np.hypot(c[0], c[1])), float(np.arctan2(c[0], c[1]))


def sweep(amp: float, L0: float = 0.28, p_mid: float = np.pi / 2) -> list[tuple]:
    rows = []
    for f in FREQS:
        dur = max(6.0 / f, 3.0)
        t = np.arange(int(dur * FS)) / FS
        p_cmd = p_mid + amp * np.sin(2 * np.pi * f * t)
        tg = V.ik_traj(np.full_like(t, L0), p_cmd)
        r = V.sim_run(tg)
        m = t > t[-1] * 0.5
        ax, phx = fit_sine(t[m], p_cmd[m], f)
        ay, phy = fit_sine(t[m], r["phi0"][m], f)
        dphi = (np.degrees(phy - phx) + 180) % 360 - 180
        rows.append((f, ay / ax, dphi, ay, r["L"][m].mean()))
    return rows


def main() -> None:
    print("摆动方向 φ0 扫频（L 锁 0.28，吊装 + 重力）")
    print(" 幅值(rad)  f(Hz)   增益   相位(deg)  实测幅值(rad)  L均值(m)")
    out = []
    for amp in (0.10, 0.20):
        for f, g, p, a, Lm in sweep(amp):
            print("   %.2f     %5.2f  %5.2f  %8.1f   %.3f       %.4f" % (amp, f, g, p, a, Lm))
            out.append((amp, f, g, p))
    np.savetxt(
        os.path.join(HERE, "swing_frf.csv"),
        np.array(out),
        delimiter=",",
        header="amp_rad,freq_hz,gain,phase_deg",
        comments="",
        fmt="%.4f",
    )
    print("-> swing_frf.csv")


if __name__ == "__main__":
    main()
