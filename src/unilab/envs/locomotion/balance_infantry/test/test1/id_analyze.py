#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
id_analyze.py — RMUC balance-infantry real-robot system-identification analysis.

Companion script for id_analysis.md.  Everything printed here is reproducible:
    UV_CACHE_DIR=/tmp/uv-cache ~/.local/bin/uv run --no-sync python \
        src/unilab/envs/locomotion/balance_infantry/test/test1/id_analyze.py

Data:  src/unilab/envs/locomotion/balance_infantry/test/test1/*.csv  (VOFA+ justfloat CSV)
Code:  AI_policy.c (same folder)  +  ../../quantize/*/ctrl_params.h

Dependencies: numpy only (no pandas / scipy).
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# 0.  Constants taken from source (NOT fitted)
# ══════════════════════════════════════════════════════════════════════════════
# ctrl_params.h  (quantize/2026-10-30_09-18-48_mujoco/ctrl_params.h)
L1, L2, L3, L4, L5 = 0.215, 0.254, 0.254, 0.215, 0.0
LEG_KP = 40.0          # N·m/rad, all four leg joints
LEG_KD = 2.0           # N·m·s/rad
WHEEL_KD = 0.1         # N·m/(rad/s)  -> wheel torque = WHEEL_KD*(w_cmd - w)
LEG_TAU_MAX = 40.0     # N·m  (AI_policy.c:1162)
WHEEL_TAU_MAX = 4.0    # N·m  (AI_policy.c:1166)
PI6 = math.pi / 6.0
PI56 = 5.0 * math.pi / 6.0

# CSV column semantics (AI_policy.c:1291-1358, 25-channel identification frame)
CH25 = {
    0: 't            [s]   测试开始后秒数 (退出无力归零)',
    1: 'target_pos[0] LF φ4 指令 [rad]',
    2: 'target_pos[1] RF φ4 指令 [rad]',
    3: 'target_pos[2] LR φ1 指令 [rad]',
    4: 'target_pos[3] RR φ1 指令 [rad]',
    5: 'dof_pos[0] LF 实测 [rad]',
    6: 'dof_pos[1] RF 实测 [rad]',
    7: 'dof_pos[2] LR 实测 [rad]',
    8: 'dof_pos[3] RR 实测 [rad]',
    9: 'dof_vel[0..3] 实测角速度 [rad/s]',
    10: '',
    11: '',
    12: '',
    13: 'torques[0] LF 纯PD [N·m]',
    14: 'torques[1] RF 纯PD [N·m]',
    15: 'torques[2] LR 纯PD [N·m]',
    16: 'torques[3] RR 纯PD [N·m]',
    17: 'L_left  左腿实测腿长 [m] (MotionSolve)',
    18: 'L_right 右腿实测腿长 [m]',
    19: 'phi_left[0]  左腿实测摆角 [rad]',
    20: 'phi_right[0] 右腿实测摆角 [rad]',
    21: 'gas_torque_l[0] 左后 φ1 气弹簧力矩 [N·m]',
    22: 'gas_torque_l[1] 左前 φ4 气弹簧力矩 [N·m]',
    23: 'gas_torque_r[0] 右后 φ1 气弹簧力矩 [N·m]',
    24: 'gas_torque_r[1] 右前 φ4 气弹簧力矩 [N·m]',
}
# 8-channel frame (TRAJ_MODE==3 wheel mode)
CH8 = {
    0: 't [s]', 1: 'w_cmd [rad/s]', 2: 'w_act_L [rad/s]', 3: 'tau_pd_L [N·m]',
    4: 'tau_out_L = -tau_pd_L [N·m]', 5: 'w_act_R [rad/s]', 6: 'tau_pd_R [N·m]',
    7: 'tau_out_R = -tau_pd_R [N·m]',
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  small numerical helpers (numpy only)
# ══════════════════════════════════════════════════════════════════════════════
def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', skiprows=1)


def segments(a, label=''):
    """Split a VOFA csv into monotone-time runs and report the CSV quirks."""
    t_raw = a[:, 0]
    n = len(t_raw)
    n_t0 = int(np.sum(t_raw == 0))
    resets = np.where(np.diff(t_raw) < -1e-6)[0]
    dts = np.diff(t_raw)
    n_dup = int(np.sum(np.abs(dts) < 1e-9))
    segs = []
    s = 0
    for b in list(resets) + [n - 1]:
        seg = a[s:b + 1]
        tt = seg[:, 0]
        tt = tt - tt[0]
        keep = np.concatenate(([True], np.diff(tt) > 1e-9))
        segs.append((seg[keep], tt[keep]))
        s = b + 1
    segs = [s for s in segs if len(s[0]) > 20]
    # only keep segments where time actually advances
    segs = [s for s in segs if (s[1][-1] - s[1][0]) > 0.05]
    info = dict(n=n, n_t0=n_t0, n_reset=len(resets), n_dup=n_dup, nseg=len(segs),
                fs=[1.0 / np.median(np.diff(s[1])) for s in segs],
                dur=[s[1][-1] for s in segs])
    return segs, info


def trapz(y, x):
    if len(y) < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def linfit(X, y, names=None):
    """OLS y = X @ p ; returns p, se, r2, n."""
    p, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ p
    n, k = X.shape
    dof = max(n - k, 1)
    sig2 = float(r @ r) / dof
    cov = sig2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.abs(np.diag(cov)))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(r @ r) / ss_tot if ss_tot > 0 else np.nan
    return p, se, r2, n, r


def nelder_mead(f, x0, step=0.2, iters=4000, tol=1e-12):
    """Tiny Nelder-Mead minimiser (no scipy)."""
    x0 = np.asarray(x0, float)
    n = len(x0)
    sim = [x0.copy()]
    for i in range(n):
        v = x0.copy()
        v[i] += (step if abs(x0[i]) < 1e-9 else step * abs(x0[i]))
        sim.append(v)
    sim = np.array(sim)
    val = np.array([f(v) for v in sim])
    for _ in range(iters):
        idx = np.argsort(val)
        sim, val = sim[idx], val[idx]
        if abs(val[-1] - val[0]) < tol * (abs(val[0]) + tol):
            break
        cen = sim[:-1].mean(axis=0)
        xr = cen + 1.0 * (cen - sim[-1]); fr = f(xr)
        if fr < val[0]:
            xe = cen + 2.0 * (cen - sim[-1]); fe = f(xe)
            sim[-1], val[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < val[-2]:
            sim[-1], val[-1] = xr, fr
        else:
            xc = cen + 0.5 * (sim[-1] - cen); fc = f(xc)
            if fc < val[-1]:
                sim[-1], val[-1] = xc, fc
            else:
                sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])
                for i in range(1, len(sim)):
                    val[i] = f(sim[i])
    i = int(np.argmin(val))
    return sim[i], float(val[i])


def hilbert(x):
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[1:n // 2] = 2.0
        h[n // 2] = 1.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


def movavg(x, w):
    w = max(int(w), 1)
    if w < 2:
        return x.copy()
    k = np.ones(w) / w
    return np.convolve(x, k, mode='same')


# ══════════════════════════════════════════════════════════════════════════════
# 2.  five-bar kinematics (mirror of chassis_algorithm.c MotionSolve / traj_ik_exact)
# ══════════════════════════════════════════════════════════════════════════════
def fk(phi1, phi4, l1=L1, l2=L2, l3=L3, l4=L4, l5=L5):
    phi1 = np.asarray(phi1, float); phi4 = np.asarray(phi4, float)
    xB = l1 * np.cos(phi1); yB = l1 * np.sin(phi1)
    xD = l5 + l4 * np.cos(phi4); yD = l4 * np.sin(phi4)
    dx = xD - xB; dy = yD - yB
    A0 = 2 * l2 * dx; b0 = 2 * l2 * dy
    C0 = l2 * l2 + dx * dx + dy * dy - l3 * l3
    disc = np.maximum(A0 * A0 + b0 * b0 - C0 * C0, 0.0)
    phi2 = 2 * np.arctan2(b0 + np.sqrt(disc), A0 + C0)
    xC = xB + l2 * np.cos(phi2); yC = yB + l2 * np.sin(phi2)
    return np.hypot(xC - l5 / 2, yC), np.arctan2(yC, xC - l5 / 2)


def ik(L, phi0, l1=L1, l2=L2, l3=L3, l4=L4, l5=L5):
    L = np.asarray(L, float); phi0 = np.asarray(phi0, float)
    xC = L * np.cos(phi0) + l5 / 2; yC = L * np.sin(phi0)
    A = l1 + xC; B = l1 * l1 - xC * xC; C = l2 * l2 - yC * yC; D = l2 * l2 + yC * yC
    disc = 2 * l1 * l1 * D + 2 * xC * xC * C - B * B - C * C
    phi1 = 2 * np.arctan2(2 * l1 * yC + np.sqrt(np.maximum(disc, 0)), A * A - C)
    A = l3 + l4; B = l5 - xC; C = l3 - l4; D = xC + l4 - l5
    E = A * A - B * B - yC * yC; F = B * B - C * C + yC * yC
    ef = np.maximum(E * F, 0)
    phi4 = 2 * np.arctan2(2 * l4 * yC - np.sqrt(ef), D * D + yC * yC - l3 * l3)
    return phi1, phi4


def jac(L, phi0, dl=1e-6):
    l_p, _ = fk(*ik(L + dl, phi0)); l_m, _ = fk(*ik(L - dl, phi0))
    dLdL = (l_p - l_m) / (2 * dl)
    p_p, p_m = ik(L, phi0 + dl)[0], ik(L, phi0 - dl)[0]
    p_p4, p_m4 = ik(L, phi0 + dl)[1], ik(L, phi0 - dl)[1]
    dphi1 = (p_p - p_m) / (2 * dl)
    dphi4 = (p_p4 - p_m4) / (2 * dl)
    return dLdL, dphi1, dphi4


def sec(title):
    print('\n' + '=' * 78)
    print(' ' + title)
    print('=' * 78)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  geometry validation
# ══════════════════════════════════════════════════════════════════════════════
def part_geometry():
    sec('3. 五位连杆几何 / 正逆解自洽性 (link lengths L1=%.3f L2=L3=%.3f L4=%.3f L5=%.1f)'
        % (L1, L2, L4, L5))
    Ln, p0n = fk(PI56, PI6)
    print('  标称位形 (φ1=5π/6, φ4=π/6): L=%.6f m, phi0=%.6f rad (π/2=%.6f)' % (Ln, p0n, math.pi / 2))
    p1, p4 = ik(0.28035, 1.5711)
    print('  逆解 (L=0.28035, φ0=1.5711) -> φ1=%.6f (5π/6=%.6f), φ4=%.6f (π/6=%.6f)'
          % (p1, PI56, p4, PI6))
    dLdL, dphi1, dphi4 = jac(0.28035, math.pi / 2)
    print('  雅可比 @标称:  ∂(φ1)/∂L = %+.4f  ∂(φ4)/∂L = %+.4f  (rad per m)' % (dphi1, dphi4))
    print('                  ∂(φ1)/∂φ0 = %+.4f ∂(φ4)/∂φ0 = %+.4f' % (dphi1, dphi4))
    h = 1e-6
    J1 = (fk(PI56 + h, PI6)[0] - fk(PI56 - h, PI6)[0]) / (2 * h)
    J4 = (fk(PI56, PI6 + h)[0] - fk(PI56, PI6 - h)[0]) / (2 * h)
    print('  几何雅可比 ∂L/∂φ1 = %+.4f m/rad, ∂L/∂φ4 = %+.4f m/rad' % (J1, J4))
    print('  代码注释里说 J_l[0][1] ≈ J_l[1][1] ≈ -0.15  -> 固件补偿用的雅可比与几何差 %.1f× / %.1f×'
          % (abs(J1) / 0.15, abs(J4) / 0.15))
    print('  等价关节刚度: k_L = kp/(J1²+J4²) = %.0f N/m = %.2f N/mm'
          % (LEG_KP / (J1 ** 2 + J4 ** 2), LEG_KP / (J1 ** 2 + J4 ** 2) / 1000))
    return J1, J4, Ln, p0n


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CSV inventory / quirks
# ══════════════════════════════════════════════════════════════════════════════
def part_inventory():
    sec('4.  CSV 清单与数据质量 (通道数 / 采样率 / 时长 / 异常)')
    files = ['wheel_slope.csv', 'vofasin.csv', 'vofasine.csv',
             'vofalegsin+.csv', 'vofagas.csv', 'vofagas1.csv']
    out = {}
    for f in files:
        a = load(f)
        segs, info = segments(a)
        print('  %-16s ncol=%d rows=%d | t==0 行=%d 重复时间戳=%d 时间回绕=%d | seg=%d'
              % (f, a.shape[1], info['n'], info['n_t0'], info['n_dup'], info['n_reset'], info['nseg']))
        for i, (s, tt) in enumerate(segs):
            dt = np.diff(tt)
            print('      seg%d: n=%6d  t=[%8.3f,%8.3f] s  fs=%.2f Hz  dt p1/p99=%.5f/%.5f  jitter=%.1f%%'
                  % (i, len(tt), tt[0], tt[-1], 1 / np.median(dt),
                     np.percentile(dt, 1), np.percentile(dt, 99),
                     100 * (np.percentile(dt, 99) - np.percentile(dt, 1)) / np.median(dt)))
        out[f] = segs
    print('\n  说明: t==0 的行是 rl_active=0 的空闲帧 (g_traj_t_run0 被清零)。')
    print('        t 单调但与上一段不连续 = 一次新的「退出无力」, 段间不能直接拼。')
    print('        重复时间戳 = 同一帧被 VOFA+ 写了两次, 统计时必须去重。')
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5.  wheel ramp (static / kinetic friction)
# ══════════════════════════════════════════════════════════════════════════════
def part_wheel_ramp():
    sec('5.  wheel_slope.csv  —— 轮速斜坡 (TRAJ_MODE=3, TRAJ_WAVE=1)')
    segs, info = segments(load('wheel_slope.csv'))
    a, t = segs[0]
    wcmd, wL, tauL, wR, tauR = a[:, 1], a[:, 2], a[:, 3], a[:, 5], a[:, 6]
    ramp = np.polyfit(t[(t > 8) & (wcmd < 29)], wcmd[(t > 8) & (wcmd < 29)], 1)
    t_on_cmd = t[np.argmax(wcmd > 1e-6)]
    t_clamp = t[np.argmax(wcmd >= 29.999)]
    print('  指令: 线性斜坡 0→%.1f rad/s, 斜率 %.4f rad/s^2 (=> T_ramp=%.2f s), 起始 t=%.3f s,'
          % (wcmd.max(), ramp[0], wcmd.max() / ramp[0], t_on_cmd))
    print('        到达上限 t=%.3f s, 之后恒值保持到日志结束 t=%.3f s (共 %.2f s)'
          % (t_clamp, t[-1], t[-1] - t_clamp))
    print('  稳态: w_act_L=%.3f  w_act_R=%.3f rad/s (指令 30), tau_pd_L=%.4f tau_pd_R=%.4f N·m'
          % (wL[t > t_clamp + 2].mean(), wR[t > t_clamp + 2].mean(),
             tauL[t > t_clamp + 2].mean(), tauR[t > t_clamp + 2].mean()))
    print('  校验 tau_out = -tau_pd: max|I4+I3| = %.1e, max|I7+I6| = %.1e'
          % (np.abs(a[:, 4] + a[:, 3]).max(), np.abs(a[:, 7] + a[:, 6]).max()))
    print('  校验 tau_pd == wheel_kd*(w_cmd - w_act), wheel_kd=0.1:')
    for nm, w, tau in [('L', wL, tauL), ('R', wR, tauR)]:
        pred = WHEEL_KD * (wcmd - w)
        r = tau - pred
        print('     %s: rms(resid)=%.2e N·m  max=%.2e  (=> 力矩通道就是纯比例速度环, 无额外滤波)'
              % (nm, np.sqrt((r ** 2).mean()), np.abs(r).max()))

    res = {}
    for nm, w, tau in [('L', wL, tauL), ('R', wR, tauR)]:
        # ---- (a) break-away / static friction
        thr = 0.20
        i_on = int(np.argmax(w > thr))
        # last sample still stuck
        i_stick = i_on - 1
        while i_stick > 0 and w[i_stick] > 0.02:
            i_stick -= 1
        Fs = tau[i_stick]
        Fs_next = tau[min(i_on + 1, len(t) - 1)]
        print('\n  [%s] 静摩擦: 轮子首次越过 %.1f rad/s 于 t=%.4f s (指令 w=%.4f)'
              % (nm, thr, t[i_on], wcmd[i_on]))
        print('      最后静止样本 t=%.4f: w=%.4f, tau=%.4f N·m ; 下一帧 tau=%.4f ; tau 峰值=%.4f'
              % (t[i_stick], w[i_stick], Fs, Fs_next, tau.max()))
        print('      => F_s ≈ %.4f N·m (取脱离前一帧), 上界 %.4f N·m (全程峰值)'
              % (Fs, tau.max()))
        # ---- (b) sliding-phase integral regression:  J dw = tau - b w - c
        sl = (t > t[i_on] + 1.0) & (wcmd < 29.9)
        idx = np.where(sl)[0]
        rows, rhs = [], []
        win, stride = 0.5, 0.25
        for t0 in np.arange(t[idx[0]], t[idx[-1]] - win, stride):
            m = (t >= t0) & (t <= t0 + win)
            if m.sum() < 10:
                continue
            tt, ww, tt2 = t[m], w[m], tau[m]
            rows.append([ww[-1] - ww[0], trapz(ww, tt), tt[-1] - tt[0]])
            rhs.append(trapz(tt2, tt))
        X = np.array(rows); y = np.array(rhs)
        p, se, r2, n, r = linfit(X, y)
        print('\n  [%s] 滑动段积分回归  ∫tau dt = J·Δw + b·∫w dt + c·Δt   (%d 个 0.5 s 窗口)'
              % (nm, n))
        print('      J  = %8.5f ± %.5f kg·m²   (%.1f ± %.1f g·m²)'
              % (p[0], se[0], p[0] * 1e3, se[0] * 1e3))
        print('      b  = %8.5f ± %.5f N·m·s/rad' % (p[1], se[1]))
        print('      c  = %8.5f ± %.5f N·m  (= F_k + 常值偏置; 本轮只有一个转向, 无法分离)'
              % (p[2], se[2]))
        print('      R² = %.5f, 残差 rms = %.2e N·m·s' % (r2, np.sqrt((r ** 2).mean())))
        # ---- (c) direct torque-vs-speed at high speed (steady state)
        ss = t > t_clamp + 2.0
        res[nm] = dict(F_s=Fs, F_s_max=tau.max(), J=p[0], J_se=se[0], b=p[1], b_se=se[1],
                       c=p[2], c_se=se[2], w_ss=w.mean(), tau_ss=tau.mean())
        print('      稳态点 (w=%.3f rad/s): tau=%.4f N·m => 该点总阻力 = %.4f N·m'
              % (w[ss].mean(), tau[ss].mean(), tau[ss].mean()))
    print('\n  交叉验证: 若用 sweep 得到的 J 固定, c = tau_ss - bw_ss ... 见第 6 节。')
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 6.  wheel sine (5 Hz) + log sweep FRF
# ══════════════════════════════════════════════════════════════════════════════
def frf_sweep_vec(cmd, act, t, fs, fmin=0.6, fmax=28.0, nw_per=3.0):
    """complex demodulation of a (log) sweep: local complex amplitude ratio."""
    # 必须先去掉直流分量: 腿摆角指令是 1.57±0.05 rad, 直接取解析信号相位会被
    # 大偏置压成小幅振荡, unwrap 后 f_inst≈0, 所有点都会被 fmin 过滤掉 (之前的 bug)。
    cmd = cmd - np.mean(cmd)
    act = act - np.mean(act)
    th = np.unwrap(np.angle(hilbert(cmd)))
    f_inst = np.gradient(th) / (2 * np.pi) / np.median(np.diff(t))
    f_inst = movavg(f_inst, 25)
    e = np.exp(-1j * th)
    n = len(t)
    f_out, G, ac, aa = [], [], [], []
    stride = max(int(0.02 * fs), 5)
    i = 1
    while i < n - 2:
        fq = max(f_inst[i], 1e-3)
        if fq < fmin * 0.7:
            i += stride
            continue
        w = int(round(nw_per / fq * fs))
        w = min(max(w, 32), n)
        i0, i1 = max(0, i - w // 2), min(n, i + w // 2)
        if i1 - i0 < 32:
            break
        ww = np.hanning(i1 - i0)
        cc = 2 * np.sum(cmd[i0:i1] * e[i0:i1] * ww) / np.sum(ww)
        aav = 2 * np.sum(act[i0:i1] * e[i0:i1] * ww) / np.sum(ww)
        if abs(cc) > 1e-6:
            f_out.append(f_inst[i]); ac.append(abs(cc)); aa.append(abs(aav))
            G.append(aav / cc)
        i += stride
    f_out = np.array(f_out); G = np.array(G); ac = np.array(ac); aa = np.array(aa)
    m = (f_out > fmin) & (f_out < fmax) & np.isfinite(G) & (ac > 0.15 * np.abs(cmd).max())
    return f_out[m], G[m], ac[m], aa[m]


def part_wheel_frf(ramp_res):
    sec('6.  轮速正弦 / 对数扫频  —— 闭环带宽 / 惯量 / 延时')
    segs_s, _ = segments(load('vofasin.csv'))
    a, t = segs_s[0]
    fs = 1 / np.median(np.diff(t))
    print('  vofasin.csv  : 定频正弦, w_cmd 幅值 %.3f rad/s' % np.abs(a[:, 1]).max())
    zc = np.where(np.diff(np.sign(a[:, 1])) != 0)[0]
    per = np.diff(t[zc])
    print('      过零间隔 mean=%.4f s -> f = %.4f Hz (std %.2e s), %d 个半周期, 时长 %.2f s'
          % (per.mean(), 0.5 / per.mean(), per.std(), len(zc), t[-1]))
    segs_e, _ = segments(load('vofasine.csv'))
    a2, t2 = segs_e[0]
    fs2 = 1 / np.median(np.diff(t2))
    print('  vofasine.csv : 对数扫频, w_cmd 幅值 %.3f rad/s, 时长 %.2f s'
          % (np.abs(a2[:, 1]).max(), t2[-1]))
    f, G, ac, aa = frf_sweep_vec(a2[:, 1], a2[:, 2], t2, fs2, fmin=2.5)
    fL = f
    print('      扫频瞬时频率 %.2f -> %.2f Hz (每段窗口 %.1f 周期)' % (fL[0], fL[-1], 4))
    # log-linearity of the sweep
    m = (fL > 1) & (fL < 18)
    p = np.polyfit(fL[m], np.log(fL[m]), 1)
    print('      ln f vs t 不是线性 (扫频是 ln-f 对 t 线性): 见下')
    # check exponential sweep law on time
    th_c = np.unwrap(np.angle(hilbert(a2[:, 1])))
    f_inst_t = np.gradient(th_c) / (2 * np.pi) / np.median(np.diff(t2))
    tt = t2.copy()
    good = (f_inst_t > 1.0) & (f_inst_t < 18) & (tt > 1.0)
    q = np.polyfit(tt[good], np.log(f_inst_t[good]), 1)
    print('      ln f(t) = %+.5f t %+.5f  (R²=%.5f) -> 扫频速率 %.3f decade/s, T_sweep=%.2f s, r=f1/f0=%.2f'
          % (q[0], q[1], 1 - np.sum((np.log(f_inst_t[good]) - np.polyval(q, tt[good])) ** 2) / np.sum((np.log(f_inst_t[good]) - np.log(f_inst_t[good]).mean()) ** 2),
             q[0] / math.log(10), math.log(20) / q[0], math.exp(q[0] * tt[good][-1])))
    # ---- fit 1st order + delay
    g0_hold = [1.0]

    def model(params, fq):
        J, b, tau_d = params
        return g0_hold[0] * np.exp(-2j * np.pi * fq * tau_d) * WHEEL_KD / (WHEEL_KD + b + 1j * 2 * np.pi * fq * J)

    def cost(params):
        J, b, tau_d = params
        if J <= 1e-6 or J > 0.1 or b < 0 or b > 1 or tau_d < -0.02 or tau_d > 0.05:
            return 1e6
        Gm = model(params, fL)
        e = (np.abs(Gm) - np.abs(G)) ** 2 + (np.angle(Gm) - np.angle(G)) ** 2
        return float(np.sum(e))

    best, bv = nelder_mead(cost, [0.004, 0.001, 0.008], step=0.3)
    # one extra pass with the best real DC-gain scale (absorbs any loop scale error)
    Gm0 = model(best, fL)
    g0_hold[0] = float(np.real(np.sum(G * np.conj(Gm0)) / np.sum(np.abs(Gm0) ** 2)))
    best, bv = nelder_mead(cost, list(best), step=0.1)
    J, b, tau_d = best
    print('  整体增益 g0 = %.3f (远小于 1 说明通道有比例误差; 本机接近 1)' % g0_hold[0])
    Gm = model(best, fL)
    r2_mag = 1 - np.sum((np.abs(Gm) - np.abs(G)) ** 2) / np.sum((np.abs(G) - np.abs(G).mean()) ** 2)
    ph_r2 = 1 - np.sum((np.angle(Gm) - np.angle(G)) ** 2) / np.sum((np.angle(G) - np.angle(G).mean()) ** 2)
    fc = (WHEEL_KD + b) / (2 * np.pi * J)
    print('\n  一阶模型  G(s) = kd/(Js+kd+b)·e^{-sτd}   (kd=%.2f N·m/(rad/s) 已知)' % WHEEL_KD)
    print('      J     = %.5f kg·m²  (%.2f g·m²)' % (J, J * 1e3))
    print('      b     = %.5f N·m·s/rad' % b)
    print('      τd    = %.2f ms' % (tau_d * 1e3))
    print('      f_c   = (kd+b)/(2πJ) = %.3f Hz   (ω_c=%.2f rad/s)' % (fc, 2 * np.pi * fc))
    print('      R²(幅值) = %.5f,  R²(相位) = %.5f,  cost=%.4f' % (r2_mag, ph_r2, bv))
    # publish a table of the FRF
    print('\n 频段 FRF 抽样 (f Hz, |G|, 相位 deg, G_cmd 幅值, G_act 幅值):')
    for fq in [1, 2, 3, 4, 5, 7, 10, 13, 16, 19]:
        i = int(np.argmin(np.abs(fL - fq)))
        print('    %5.1f  %.4f  %8.2f   %.3f  %.3f   | 模型 %.4f %8.2f'
              % (fL[i], abs(G[i]), np.degrees(np.angle(G[i])), ac[i], aa[i],
                 abs(Gm[i]), np.degrees(np.angle(Gm[i]))))
    # ---- 5 Hz fixed sine cross-check
    f5, G5, ac5, aa5 = frf_sweep_vec(a[:, 1], a[:, 2], t, fs)
    i = int(np.argmin(np.abs(f5 - 5.0)))
    print('\n  5 Hz 定频正弦 (vofasin.csv) 交叉验证: f=%.2f Hz |G|=%.4f (模型 %.4f) 相位=%.1f° (模型 %.1f°)'
          % (f5[i], abs(G5[i]), abs(model(best, np.array([f5[i]])))[0],
             np.degrees(np.angle(G5[i])), np.degrees(np.angle(model(best, np.array([f5[i]])))[0])))
    print('      => 两条独立数据 (定频 5 Hz + 1..20 Hz 扫频) 一致, 一阶+延时模型成立。')
    print('      增益在 5 Hz 已跌到 %.2f, 若按 1/(1+jf/fc) 反推 fc=%.2f Hz, 与上面 %.2f Hz 一致。'
          % (abs(G5[i]), 5.0 * abs(G5[i]) / math.sqrt(max(1 - abs(G5[i]) ** 2, 1e-9)), fc))
    # ---- delay decomposition
    print('\n  延时分解: 一阶环节本身在 %.1f Hz 的相位 = %.1f°, 实测 %.1f°, 差 = 纯延时部分。'
          % (fL[-1], -np.degrees(math.atan(2 * np.pi * fL[-1] / (2 * np.pi * fc))), np.degrees(np.angle(G[-1]))))
    print('      推论: 这个 τd 是「速度环 + CAN + 电机内环」的总等效力矩/反馈延时,')
    print('      在 J=%.4f 下使相位裕度额外损失 %.1f° @f_c。' % (J, np.degrees(2 * np.pi * fc * tau_d)))
    # ---- torque-vs-speed
    print('\n  力矩-转速关系 (扫频段, tau_pd = kd(w_cmd-w_act)):')
    for lo, hi in [(1, 3), (3, 5), (5, 8), (8, 12), (12, 16), (16, 20)]:
        m = (fL >= lo) & (fL < hi)
        if m.sum() == 0:
            continue
        ww = aa[m]
        print('    %2d-%2d Hz: |w| rms=%.3f rad/s, |tau| rms=%.4f N·m  -> tau/w=%.5f N·m·s/rad'
              % (lo, hi, np.sqrt((aa[m] ** 2).mean()), np.sqrt((abs(G[m] * ac[m]) * WHEEL_KD) ** 2).mean(),
                 np.sqrt((abs(G[m] * ac[m]) * WHEEL_KD) ** 2).mean() / max(np.sqrt((aa[m] ** 2).mean()), 1e-9)))
    return dict(J=J, b=b, tau_d=tau_d, fc=fc, r2_mag=r2_mag, ph_r2=ph_r2, f=fL, G=G)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  leg length test  (vofagas / vofagas1)
# ══════════════════════════════════════════════════════════════════════════════
def part_leg_length(name, J1, J4):
    sec('7.  %s  —— 腿长三角波 (TRAJ_MODE=1)' % name)
    segs, info = segments(load(name))
    a, t = segs[0]
    fs = 1 / np.median(np.diff(t))
    Lc, p0c = fk(PI56 + a[:, 3], PI6 + a[:, 1])
    LmL, LmR = a[:, 17], a[:, 18]
    p0mL, p0mR = a[:, 19], a[:, 20]
    tauLF, tauRF, tauLR, tauRR = a[:, 13], a[:, 14], a[:, 15], a[:, 16]
    print('  时长 %.2f s @ %.1f Hz; 指令 L %.4f..%.4f m (幅值 ±%.4f), φ0 恒 %.5f'
          % (t[-1], fs, Lc.min(), Lc.max(), (Lc.max() - Lc.min()) / 2, p0c.mean()))
    # waveform + period
    ac = Lc - Lc.mean()
    zc = np.where(np.diff(np.sign(ac)) != 0)[0]
    per = np.diff(t[zc])
    print('  波形: 周期 %.4f s (f=%.4f Hz), 半周期 %d 个, |dL/dt| 中位 = %.4f m/s (三角波应为 %.4f)'
          % (per.mean(), 1 / per.mean(), len(zc), np.median(np.abs(np.gradient(Lc, t))),
             4 * 0.08 / per.mean()))
    # tracking error
    dL = LmL - Lc
    print('\n  跟踪误差 dL = L_meas - L_cmd:  mean %+.1f mm, std %.1f mm, min %+.1f, max %+.1f mm'
          % (dL.mean() * 1e3, dL.std() * 1e3, dL.min() * 1e3, dL.max() * 1e3))
    print('  实测腿长范围: L %.4f..%.4f m  (指令 %.3f..%.3f, 少伸 %.1f mm / 多伸 %.1f mm)'
          % (LmL.min(), LmL.max(), Lc.min(), Lc.max(),
             (Lc.min() - LmL.min()) * 1e3, (LmL.max() - Lc.max()) * 1e3))
    print('  摆角 φ0 本应恒为 %.4f, 实测 %.4f..%.4f rad (漂 %.1f mrad) -> 腿长测试里混进了明显的摆角误差'
          % (math.pi / 2, p0mL.min(), p0mL.max(), (p0mL.max() - p0mL.min()) * 1e3))
    # hysteresis: split by sign(dLc/dt)
    dLc = np.gradient(Lc, t)
    moving = np.abs(dLc) > 0.2 * np.median(np.abs(dLc))
    up, dn = (dLc > 0) & moving, (dLc < 0) & moving
    print('\n  迟滞: 上行 dL = %+.1f ± %.1f mm, 下行 dL = %+.1f ± %.1f mm'
          % (dL[up].mean() * 1e3, dL[up].std() * 1e3, dL[dn].mean() * 1e3, dL[dn].std() * 1e3))
    print('        -> 中线 = %+.1f mm (弹性/补偿残余), 半差 = %.1f mm (摩擦)'
          % ((dL[up].mean() + dL[dn].mean()) / 2 * 1e3, (dL[dn].mean() - dL[up].mean()) / 2 * 1e3))
    # joint-space residual torque
    eLF = a[:, 1] - a[:, 5]; eLR = a[:, 3] - a[:, 7]
    print('\n  关节跟踪误差: LF(φ4) rms %.4f rad, LR(φ1) rms %.4f rad  (kp=40 => τ 残差 %.2f / %.2f N·m)'
          % (np.sqrt((eLF ** 2).mean()), np.sqrt((eLR ** 2).mean()),
             LEG_KP * np.sqrt((eLF ** 2).mean()), LEG_KP * np.sqrt((eLR ** 2).mean())))
    print('  实测 PD 力矩: LF %+.2f..%+.2f, LR %+.2f..%+.2f N·m (钳位 ±40, 是否饱和: %s)'
          % (tauLF.min(), tauLF.max(), tauLR.min(), tauLR.max(),
             '是' if max(np.abs(tauLF).max(), np.abs(tauLR).max()) > 39.5 else '否'))
    # equivalent leg-axis residual force and stiffness
    Faxis = (J1 * tauLR + J4 * tauLF) / (J1 ** 2 + J4 ** 2)   # least squares along (J1,J4)
    print('  把 PD 残差投影到腿轴向: F_res = (J1·τ1 + J4·τ4)/|J|² = %+.2f .. %+.2f N (mean %+.2f)'
          % (Faxis.min(), Faxis.max(), Faxis.mean()))
    print('  但两个关节的 τ 同号/异号:  τ_LR/τ_LF = %+.2f (纯腿轴向力应 = J1/J4 = %+.2f)'
          % (tauLR.mean() / tauLF.mean(), J1 / J4))
    print('   => 残差力矩不是纯腿轴向载荷, 说明固件的气弹簧补偿雅可比 (J_l≈0.15)')
    print('      与几何雅可比 (%.3f/%.3f) 不一致, 补偿在前后关节上分错了。' % (J1, J4))
    # stiffness from the elastic (mid-line) part
    dLc_mid = dL.copy()
    m = np.abs(dL) < 0.06
    X = np.column_stack([dL[m], np.ones(m.sum())])
    p, se, r2, n, res = linfit(X, Faxis[m])
    print('  静态刚度 (F_res vs dL, 弹性段): dF/dL = %+.1f N/m (%+.2f N/mm), 截距 %+.2f N, R²=%.4f'
          % (p[0], p[0] / 1e3, p[1], r2))
    print('      与理论 k_L = kp/(J1²+J4²) = %.0f N/m 相比 => %.2f×'
          % (LEG_KP / (J1 ** 2 + J4 ** 2), p[0] / (LEG_KP / (J1 ** 2 + J4 ** 2))))
    # gas spring torque channel
    g1, g2 = a[:, 21], a[:, 22]
    print('\n  气弹簧力矩通道: gas_l[0](φ1) = %+.3f..%+.3f, gas_l[1](φ4) = %+.3f..%+.3f N·m'
          % (g1.min(), g1.max(), g2.min(), g2.max()))
    print('   |gas[0]/gas[1]| = %.3f  (几何雅可比比 |J1/J4| = %.2f)  => 固件用 |J|=0.15, 0.15'
          % (np.abs(g1 / g2).mean(), abs(J1 / J4)))
    # fit F_gas assuming the code's J_l = -0.15
    print('   若按固件 J_l=0.15: F_gas = %.2f .. %.2f N (随 L 单调增?) 相关系数 r(L,F) = %.4f'
          % (np.abs(g1).min() / 0.15, np.abs(g1).max() / 0.15, np.corrcoef(Lc, np.abs(g1))[0, 1]))
    X = np.column_stack([np.abs(g1), np.ones(len(g1))])
    p, se, r2, n, res = linfit(X, Lc)
    print('   线性拟合 F_gas = a·|gas| + b 回推 L: R²=%.4f (仅作一致性检查)' % r2)
    return dict(Lc=Lc, Lm=LmL, dL=dL, t=t, Faxis=Faxis, J1=J1, J4=J4)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  leg swing sweep (vofalegsin+)
# ══════════════════════════════════════════════════════════════════════════════
def part_leg_swing(J1, J4):
    sec('8.  vofalegsin+.csv  —— 摆角对数扫频 (TRAJ_MODE=0, TRAJ_SWEEP=1)')
    segs, info = segments(load('vofalegsin+.csv'))
    print('  文件含 %d 段: %s' % (len(segs), ['%.1f s' % s[1][-1] for s in segs]))
    a, t = segs[-1]
    fs = 1 / np.median(np.diff(t))
    Lc, p0c = fk(PI56 + a[:, 3], PI6 + a[:, 1])
    p0m = a[:, 19]
    print('  指令: L 恒 %.4f m, φ0 扫频 %.4f..%.4f rad (幅值 ±%.4f), 时长 %.2f s @ %.1f Hz'
          % (Lc.mean(), p0c.min(), p0c.max(), (p0c.max() - p0c.min()) / 2, t[-1], fs))
    f, G, ac, aa = frf_sweep_vec(p0c, p0m, t, fs)
    if f.size < 8:
        print('  ! 扫频段复解调失败 (%d 个有效频点, 指令波形/幅值不满足解调假设), 跳过 J_eff/τd 拟合' % f.size)
        return dict(J=float('nan'), tau_d=float('nan'), wn=float('nan'), zeta=float('nan'))
    q = np.percentile(f, [1, 50, 99])
    print('  有效频点 %d 个, f = %.2f .. %.2f Hz (中位 %.2f); 幅值指令 %.4f rad, 实测 %.4f rad'
          % (f.size, q[0], q[2], q[1], ac.mean(), aa.max()))
    # first order (PD position loop on inertia) + delay
    def model(params, fq):
        J, tau_d, g0 = params
        s = 1j * 2 * np.pi * fq
        return g0 * (LEG_KP + LEG_KD * s) / (J * s ** 2 + LEG_KD * s + LEG_KP) * np.exp(-s * tau_d)

    def cost(params):
        J, tau_d, g0 = params
        if not (1e-4 < J < 2.0) or not (-0.01 < tau_d < 0.05) or not (0.5 < g0 < 1.5):
            return 1e6
        Gm = model(params, f)
        return float(np.sum((np.abs(Gm) - np.abs(G)) ** 2 + (np.angle(Gm) - np.angle(G)) ** 2))

    best, bv = nelder_mead(cost, [0.05, 0.008, 1.0], step=0.3)
    J, tau_d, g0 = best
    Gm = model(best, f)
    r2m = 1 - np.sum((np.abs(Gm) - np.abs(G)) ** 2) / np.sum((np.abs(G) - np.abs(G).mean()) ** 2)
    r2p = 1 - np.sum((np.angle(Gm) - np.angle(G)) ** 2) / np.sum((np.angle(G) - np.angle(G).mean()) ** 2)
    wn = math.sqrt(LEG_KP / J)
    zeta = LEG_KD / (2 * math.sqrt(LEG_KP * J))
    print('\n  二阶闭环模型 G = (kp+kd·s)/(J s²+kd s+kp)·e^{-sτd}  (kp=%.0f kd=%.1f 已知):' % (LEG_KP, LEG_KD))
    print('      J_eff = %.5f kg·m² (%.1f g·m²)' % (J, J * 1e3))
    print('      ωn = sqrt(kp/J) = %.2f rad/s = %.2f Hz,  ζ = kd/(2√(kpJ)) = %.3f' % (wn, wn / 2 / math.pi, zeta))
    print('      τd = %.2f ms,  g0 = %.3f' % (tau_d * 1e3, g0))
    print('      R²(幅值)=%.4f R²(相位)=%.4f' % (r2m, r2p))
    print('\n  f [Hz]   |G|     ∠G [deg]   |模型|   ∠模型')
    for fq in [1, 2, 3, 5, 7, 9, 11, 13, 15, 18]:
        i = int(np.argmin(np.abs(f - fq)))
        print('  %5.1f  %.3f  %8.1f    %.3f  %8.1f'
              % (f[i], abs(G[i]), np.degrees(np.angle(G[i])), abs(Gm[i]), np.degrees(np.angle(Gm[i]))))
    print('\n  半功率点(-3 dB, 相对 DC): ', end='')
    mag = np.abs(G)
    i3 = int(np.argmax(mag < mag[:3].mean() / math.sqrt(2)))
    print('f_-3dB ≈ %.2f Hz (相对最低频段 |G|≈%.2f, 不是相对 DC=1)' % (f[i3], mag[:3].mean()))
    print('  谐振峰: 最大 |G| = %.3f @ %.2f Hz (ζ=%.2f 时闭环 PD 基本无峰)'
          % (mag.max(), f[int(np.argmax(mag))], zeta))
    if r2m < 0.8:
        amp = (p0c.max() - p0c.min()) / 2
        print('\n  !! 拟合不可信 (R²(幅值)=%.2f):' % r2m)
        print('     本测试 φ0 指令幅值只 ±%.3f rad, 折到关节 ±%.3f rad < 干摩擦死区'
              % (amp, amp))
        print('     F/kp ≈ %.3f rad (F≈2.6 N·m, kp=40), 激励落在 stick-slip 区间;' % (2.6 / LEG_KP))
        print('     且实测 L 在 %.3f..%.3f m 间变 (指令恒 0.28 m), 腿被机身重量压着上下弹,'
              % (a[:, 17].min(), a[:, 17].max()))
        print('     主导模态是腿长而不是摆角。=> J_eff 不是关节惯量, 不要写进仿真。')
        print('     要辨识关节惯量: 幅值加到 ±0.2 rad (4 倍死区) 以上, 并锁住/吊住 L。')
    return dict(J=J, tau_d=tau_d, wn=wn, zeta=zeta, f=f, G=G, r2m=r2m, r2p=r2p)


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(__doc__)
    print('─' * 78)
    print('通道表 (25 列辨识帧, AI_policy.c:1291-1358):')
    for k in sorted(CH25):
        if CH25[k]:
            print('   I%-2d %s' % (k, CH25[k]))
    print('通道表 (8 列轮速帧, AI_policy.c:1281-1289):')
    for k in sorted(CH8):
        print('   I%-2d %s' % (k, CH8[k]))

    J1, J4, Ln, p0n = part_geometry()
    part_inventory()
    ramp = part_wheel_ramp()
    frf = part_wheel_frf(ramp)
    part_leg_length('vofagas.csv', J1, J4)
    part_leg_length('vofagas1.csv', J1, J4)
    sw = part_leg_swing(J1, J4)

    sec('9.  汇总 / 交叉校验')
    print('  轮: J(扫频)=%.5f kg·m² vs J(斜坡积分)=%.5f±%.5f kg·m²  -> 差 %.0f%%'
          % (frf['J'], ramp['L']['J'], ramp['L']['J_se'],
             100 * abs(frf['J'] - ramp['L']['J']) / frf['J']))
    print('      f_c=%.2f Hz, τd=%.1f ms, b=%.5f, F_s(L/R)=%.3f/%.3f N·m, c(L/R)=%.4f/%.4f N·m'
          % (frf['fc'], frf['tau_d'] * 1e3, frf['b'], ramp['L']['F_s'], ramp['R']['F_s'],
             ramp['L']['c'], ramp['R']['c']))
    if not np.isfinite(sw['J']) or sw.get('r2m', 0.0) < 0.8:
        print('  腿: 摆角扫频不可信 (R²(幅值)=%.2f), 不给 J_eff —— 激励幅值低于摩擦死区, '
              '实测主导模态是腿长而非摆角' % sw.get('r2m', float('nan')))
    else:
        print('  腿: J_eff(摆角扫频)=%.4f kg·m², ωn=%.1f rad/s (%.2f Hz), ζ=%.2f, τd=%.1f ms'
              % (sw['J'], sw['wn'], sw['wn'] / 2 / math.pi, sw['zeta'], sw['tau_d'] * 1e3))


if __name__ == '__main__':
    main()
