"""把 AI_policy.c 的辨识测试信号在仿真里原样跑一遍。

镜像的宏（`test1/AI_policy.c:145-232`，改这里就等于改固件）：

    TRAJ_MODE      1        0=摆角 1=腿长 2=两者 3=轮速
    TRAJ_SWEEP     0        0=定频正弦 1=对数扫频
    TRAJ_FREQ_HZ   0.05     定频频率
    TRAJ_F0/F1_HZ  1/20     扫频起止
    TRAJ_DURATION_S 40
    TRAJ_DELAY_S   1.0      先在默认位形站 1 s（t 钳 0 => s=0 => L=BASE_HEIGHT）
    TRAJ_BASE_HEIGHT 0.28
    TRAJ_BASE_PHI0   pi/2
    TRAJ_AMP_LEN   0.08     L = 0.28 ± 0.08  =>  0.20 ~ 0.36
    TRAJ_AMP_ANGLE 0.0
    TRAJ_TRIANGLE  1        (2/pi)*asin(sin(phase))
    TRAJ_VOFA_DECIM 4       125 Hz
    SOFT_START_MS  300      力矩 0->1 软启动

信号链与固件一致：(φ0, L) → 精确逆解 → 关节目标 → PD(40/2) → 饱和 → 吊装仿真。

用法（仓库根目录）:
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
        src/unilab/envs/locomotion/balance_infantry/test/test1/firmware_test.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vmc_replay as V  # noqa: E402

FS = 1.0 / 0.002  # 固件 500 Hz


def traj_signal(
    mode: int = 1,
    sweep: bool = False,
    freq: float = 0.05,
    f0: float = 1.0,
    f1: float = 20.0,
    duration: float = 40.0,
    delay: float = 1.0,
    base_h: float = 0.28,
    base_phi0: float = np.pi / 2,
    amp_len: float = 0.08,
    amp_ang: float = 0.0,
    triangle: bool = True,
    t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (t, L_cmd, phi0_cmd)，与 AI_policy.c 生成器逐行对应。"""
    if t is None:
        t = np.arange(int((duration + delay) * FS)) / FS
    tc = np.clip(t - delay, 0.0, None)  # TRAJ_DELAY_S：前 1 s 钳在 0
    tc = np.minimum(tc, duration)
    if sweep:
        r = f1 / f0
        w0 = 2 * np.pi * f0 * duration / np.log(r)
        phase = w0 * (np.power(r, tc / duration) - 1.0)
    else:
        phase = 2 * np.pi * freq * tc
    s = (2 / np.pi) * np.arcsin(np.sin(phase)) if triangle else np.sin(phase)
    s = np.where(tc >= duration, 0.0, s)  # traj_done: 撤回 0
    zero = np.zeros_like(s)
    L = base_h + (amp_len * s if mode in (1, 2) else zero)
    p0 = base_phi0 + (amp_ang * s if mode in (0, 2) else zero)
    return t, L, p0


def targets_from_signal(L: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return V.ik_traj(L, p0)


def run(name: str, **kw) -> dict:
    t, L, p0 = traj_signal(**kw)
    tg = targets_from_signal(L, p0)
    soft = np.clip(t / 0.3, 0.0, 1.0)   # SOFT_START_MS = 300
    r = V.sim_run(tg, soft=soft)
    m = t > kw.get("delay", 1.0)
    use_phi = (kw.get("mode", 1) == 0)
    c = p0[m] if use_phi else L[m]
    a = r["phi0"][m] if use_phi else r["L"][m]
    amp_cmd = (c.max() - c.min()) / 2
    amp_act = (a.max() - a.min()) / 2
    unit = "rad" if use_phi else "m"
    print(
        "%-28s 指令幅 %.3f %s → 实测 %.3f %s (增益 %.2f)  均值 %.4f"
        % (name, amp_cmd, unit, amp_act, unit, amp_act / max(amp_cmd, 1e-9), a.mean())
    )
    return {"t": t, "L_cmd": L, "L": r["L"], "phi0": r["phi0"], "amp_cmd": amp_cmd, "amp": amp_act}


def compare_with_real(mode: int = 1, freq: float = 0.05) -> dict:
    """用固件信号跑仿真，和 vofagas.csv 画同一张对比图。

    仿真的 25 列帧格式与固件 VOFA 腿模式帧一致：
      [t, tgt(4), dof(4), dof_vel(4), tau_pd(4), L(2), phi0(2), gas(4)]
    """
    import sim_replay_id as S

    t, L, p0 = traj_signal(mode=mode, freq=freq)
    tg = targets_from_signal(L, p0)
    soft = np.clip(t / 0.3, 0.0, 1.0)
    r = V.sim_run(tg, soft=soft)

    sim = np.zeros((len(t), 25))
    sim[:, 0] = t
    sim[:, 1], sim[:, 2] = tg[:, 0], tg[:, 0]  # LF / RF 目标
    sim[:, 3], sim[:, 4] = tg[:, 1], tg[:, 1]  # LR / RR 目标
    sim[:, 5:9] = r["q"]
    sim[:, 9:13] = r["qd"]
    sim[:, 13:17] = r["tau"]
    sim[:, 17], sim[:, 18] = r["L"], r["L"]
    sim[:, 19], sim[:, 20] = r["phi0"], r["phi0"]

    raw = S.load_csv(os.path.join(HERE, "vofagas.csv"))
    tm, keep = S.monotonic_time(raw[:, 0])
    real = raw[keep].copy()
    real[:, 0] = tm

    # 时间对齐：按 L 互相关找最匹配的平移（实机起始时刻不一定正好 0）
    m = real[:, 0] < real[:, 0].max() - 0.5
    xr = real[m, 17] - real[m, 17].mean()
    sr = real[m, 0]
    best, best_lag = -1e18, 0.0
    for lag in np.arange(-3.0, 3.0, 0.02):
        xs = np.interp(sr + lag, sim[:, 0], sim[:, 17])
        xs = xs - xs.mean()
        c = float(np.dot(xr, xs))
        if c > best:
            best, best_lag = c, lag
    sim[:, 0] += best_lag
    print("  时间对齐平移 %+0.2f s" % best_lag)
    real_amp = (real[:, 17].max() - real[:, 17].min()) / 2
    print("  实机 L 幅值 %.3f m ；仿真 %.3f m" % (real_amp, (sim[:, 17].max() - sim[:, 17].min()) / 2))

    out = os.path.join(HERE, "cmp_firmware_vofagas.png")
    S.plot_time(sim, real, "leg", out, "firmware test mirrored: vofagas (L triangle 0.28+-0.08 @0.05 Hz)")
    sim_csv = os.path.join(HERE, "sim_firmware_vofagas.csv")
    np.savetxt(sim_csv, sim, delimiter=",", fmt="%.6f",
               header=",".join(str(i) for i in range(25)), comments="")
    print("  -> %s\n  -> %s" % (out, sim_csv))
    return {"sim": sim, "real": real}


def main() -> None:
    print("== 当前固件设置：腿长三角波 0.28±0.08 @0.05 Hz，40 s ==")
    cur = compare_with_real(mode=1, freq=0.05)

    print("\n== 改宏之后仿真预测（同样的吊装条件）==")
    run("L 三角 ±0.08 @0.2 Hz", mode=1, freq=0.2, duration=10.0)
    run("L 正弦 ±0.08 @1 Hz", mode=1, freq=1.0, triangle=False, duration=6.0)
    run("L 扫频 1-20 Hz ±0.08", mode=1, sweep=True, duration=20.0)
    run("摆角正弦 ±0.05 @1 Hz", mode=0, amp_len=0.0, amp_ang=0.05, triangle=False, duration=6.0)
    run("摆角正弦 ±0.20 @1 Hz", mode=0, amp_len=0.0, amp_ang=0.20, triangle=False, duration=6.0)
    run("摆角扫频 1-20 Hz ±0.20", mode=0, amp_len=0.0, amp_ang=0.20, sweep=True, duration=20.0)


if __name__ == "__main__":
    main()
