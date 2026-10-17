"""测试阻尼 / 摩擦系数：VMC PD 控制 φ₀(phi0) 摆到极限 → 零力自由下摆 → 下落时间和停止角度。

物理含义:
  θ = π/2 - φ₀ (往前为正, 与 test_vmc_lqr.py 一致)
  PD 把 φ₀ 推到目标 → 释放 → 重力拉回 90°(垂直)
  → 阻尼/摩擦决定下落快慢和最终卡在哪个角度。

操作:
  1 2 3 4 → 选择参数 (damping / armature / frictionloss / phi0_kp)
  ↑ ↓     → 参数 ×2 / ÷2
  Space   → 单次下摆: PD摆腿→零力释放→打印 θ 过程
  A       → 参数扫描: 遍历6档, 对比 fall_time + theta_final
  S       → 切换 base 固定 / 自由
  R       → 复位
  Q       → 退出
"""

import mujoco
import mujoco.viewer
import numpy as np
import os, sys, csv
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "..", "..")
SCENE_DIR = os.path.join(_PARENT_DIR, "assets", "robots", "balance_infantry")
SCENE = os.path.join(SCENE_DIR, "scene_flat_pillar.xml")

sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "lqr"))
from test_vmc_lqr import fk_fivebar, compute_jacobian_partials

ACT_JOINT_NAMES = [
    "left_front_joint", "right_front_joint",
    "left_rear_joint", "right_rear_joint",
    "left_wheel_joint", "right_wheel_joint",
]
# 五连杆子关节（无 actuator，但 armature/frictionloss 默认 0.01 会阻碍重力下摆）
CHILD_JOINT_NAMES = [
    "left_front_child1_joint", "left_front_child2_joint",
    "left_front_child3_joint", "left_rear_child1_joint",
    "right_front_child1_joint", "right_front_child2_joint",
    "right_front_child3_joint", "right_rear_child1_joint",
]

DEFAULT_DAMPING  = np.array([0.05, 0.05, 0.05, 0.05, 0.01, 0.01], dtype=np.float64)
DEFAULT_ARMATURE = np.array([0.01, 0.01, 0.01, 0.01, 0.003, 0.003], dtype=np.float64)
DEFAULT_FRICLOSS = np.array([0.50, 0.50, 0.50, 0.50, 0.01, 0.01], dtype=np.float64)
DEFAULT_KP       = 100.0
DEFAULT_KD       = 10.0
DEFAULT_WHEEL_KD = 2.0

BASE_HEIGHT = 0.47

PHI0_EXTREME = np.deg2rad(180)
L0_TARGET    = 0.37  # 实测时腿长 0.37m
KP_LEG       = 500.0
KD_LEG       = 50.0
HOLD_STEPS   = 300
DROP_STEPS   = 1200
SETTLE_THRESH = 0.1  # °/s, 库仑摩擦下蠕变很慢, 0.001 太严永远达不到

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
model.opt.timestep = 0.005
model.opt.gravity[:] = np.array([0, 0, -9.81])

dof_adr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                    for n in ACT_JOINT_NAMES], dtype=np.int32)
qpos_adr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in ACT_JOINT_NAMES], dtype=np.int32)
# 子关节 DOF 地址（用于清零 armature/frictionloss）
child_dof_adr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                          for n in CHILD_JOINT_NAMES], dtype=np.int32)

CSV_DIR = os.path.join(_SCRIPT_DIR, "drop_data")
os.makedirs(CSV_DIR, exist_ok=True)

cur_damping  = DEFAULT_DAMPING.copy()
cur_armature = DEFAULT_ARMATURE.copy()
cur_fricloss = DEFAULT_FRICLOSS.copy()
cur_kp       = DEFAULT_KP
cur_kd       = DEFAULT_KD
wheel_kd     = DEFAULT_WHEEL_KD
base_fixed   = True
param_idx    = 0

PARAM_NAMES = ["damping", "armature", "frictionloss", "phi0_kp"]

auto_mode   = None
auto_step   = 0
sweep_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
sweep_idx   = 0
drop_trace  = []
drop_trace_g = None
drop_csv_data = []  # [(step, time_s, theta_deg, phi0_deg, L0_m, qvel*4, ctrl*4), ...]

_phi0_unwrap = {"L": 0.0, "R": 0.0}
_phi0_offset = {"L": 0.0, "R": 0.0}


def _unwrap_phi0(raw: float, side: str) -> float:
    global _phi0_unwrap, _phi0_offset
    diff = raw - _phi0_unwrap[side]
    if diff > np.pi:
        _phi0_offset[side] -= 2.0 * np.pi
    elif diff < -np.pi:
        _phi0_offset[side] += 2.0 * np.pi
    _phi0_unwrap[side] = raw
    return raw + _phi0_offset[side]


def _reset_unwrap() -> None:
    global _phi0_unwrap, _phi0_offset
    _phi0_unwrap = {"L": 0.0, "R": 0.0}
    _phi0_offset = {"L": 0.0, "R": 0.0}


def sync() -> None:
    for i, adr in enumerate(dof_adr):
        model.dof_damping[adr]    = cur_damping[i]
        model.dof_armature[adr]   = cur_armature[i]
        model.dof_frictionloss[adr] = cur_fricloss[i]
    # 子关节是被动转轴, 摩擦远小于主动关节 (否则会阻碍重力下摆)
    for adr in child_dof_adr:
        model.dof_damping[adr]    = 0.01
        model.dof_armature[adr]   = 0.001
        model.dof_frictionloss[adr] = 0.05


def lock_base() -> None:
    pass


def zero_ctrl() -> None:
    data.ctrl[:] = 0.0


def get_vmc_state() -> dict:
    q = np.array([data.qpos[a] for a in qpos_adr[:4]], dtype=np.float64)
    phi1_l = q[2] + 5.0 * np.pi / 6.0
    phi4_l = q[0] + np.pi / 6.0
    L0_l, phi0_raw, phi2_l, phi3_l, sigma1_l = fk_fivebar(phi1_l, phi4_l)
    phi0_l = _unwrap_phi0(float(phi0_raw), "L")
    J00, J01, J10, J11 = compute_jacobian_partials(
        phi1_l, phi4_l, phi0_l, phi2_l, phi3_l, sigma1_l, L0_l)
    pitch = float(data.qpos[4])
    theta = np.pi / 2.0 - phi0_l - pitch
    return {
        "L0": float(L0_l), "phi0": float(phi0_l), "theta": float(theta),
        "phi2": float(phi2_l), "phi3": float(phi3_l), "sigma1": float(sigma1_l),
        "phi1": float(phi1_l), "phi4": float(phi4_l),
        "J00": float(J00), "J01": float(J01), "J10": float(J10), "J11": float(J11),
    }


def apply_vmc_hold(phi0_target: float, L0_target: float) -> None:

    def _vmc_one_side(i_front: int, i_rear: int, side: str) -> tuple:
        q = np.array([data.qpos[a] for a in qpos_adr[:4]], dtype=np.float64)
        phi1 = q[i_rear] + 5.0 * np.pi / 6.0
        phi4 = q[i_front] + np.pi / 6.0
        L0, phi0_raw, phi2, phi3, sigma1 = fk_fivebar(phi1, phi4)
        phi0 = _unwrap_phi0(float(phi0_raw), side)
        J00, J01, J10, J11 = compute_jacobian_partials(phi1, phi4, phi0, phi2, phi3, sigma1, L0)

        qv = np.array([data.qvel[a] for a in dof_adr], dtype=np.float64)
        omega_front = qv[i_front]
        omega_rear = qv[i_rear]
        L0_dot = J00 * omega_rear + J01 * omega_front
        phi0_dot = J10 * omega_rear + J11 * omega_front

        F = KP_LEG * (L0_target - L0) - KD_LEG * L0_dot
        T = cur_kp * (phi0_target - phi0) - cur_kd * phi0_dot

        tau_front = J01 * F + J11 * T
        tau_rear = J00 * F + J10 * T
        return L0, phi0, np.clip(tau_front, -40.0, 40.0), np.clip(tau_rear, -40.0, 40.0)

    L0_l, phi0_l, tau_lf, tau_lr = _vmc_one_side(0, 2, "L")
    L0_r, phi0_r, tau_rf, tau_rr = _vmc_one_side(1, 3, "R")

    qv = np.array([data.qvel[a] for a in dof_adr], dtype=np.float64)
    tau = np.zeros(6, dtype=np.float64)
    tau[0] = tau_lf; tau[1] = tau_rf; tau[2] = tau_lr; tau[3] = tau_rr
    tau[4:] = -wheel_kd * qv[4:]
    data.ctrl[:] = np.clip(tau, -40.0, 40.0)


def set_parameter_to(val: float, scale_all: bool = False) -> None:
    global cur_kp
    if param_idx == 0:
        if scale_all: cur_damping[:4] *= val
        else:         cur_damping[:4] = val
    elif param_idx == 1:
        if scale_all: cur_armature[:4] *= val
        else:         cur_armature[:4] = val
    elif param_idx == 2:
        if scale_all: cur_fricloss[:4] *= val
        else:         cur_fricloss[:4] = val
    elif param_idx == 3:
        if scale_all: cur_kp = max(1.0, cur_kp * val)
        else:         cur_kp = max(1.0, val)
    sync()


def reset_all() -> None:
    global cur_kp, cur_kd
    cur_damping[:4]  = DEFAULT_DAMPING[:4]
    cur_armature[:4] = DEFAULT_ARMATURE[:4]
    cur_fricloss[:4] = DEFAULT_FRICLOSS[:4]
    cur_kp  = DEFAULT_KP
    cur_kd  = DEFAULT_KD
    sync()
    _reset_unwrap()
    mujoco.mj_resetData(model, data)
    print("\n[Reset] all params → default")


def print_header() -> None:
    print(f"\n{'='*50}")
    print(f"  param=[{PARAM_NAMES[param_idx]}]  base={'FIX' if base_fixed else 'FREE'}  grav={model.opt.gravity[2]:.2f}")
    pv = {0: cur_damping[0], 1: cur_armature[0], 2: cur_fricloss[0], 3: cur_kp}[param_idx]
    print(f"  current value: {pv:.4f}")
    st = get_vmc_state()
    print(f"  θ={np.degrees(st['theta']):.1f}°  φ₀={np.degrees(st['phi0']):.1f}°  L0={st['L0']:.4f}m  target_φ₀={np.degrees(PHI0_EXTREME):.0f}°  target_L0={L0_TARGET:.2f}m")
    print(f"{'='*50}")


def start_drop_test() -> None:
    global auto_mode, auto_step, drop_trace, drop_csv_data
    mujoco.mj_resetData(model, data)
    _reset_unwrap()
    auto_mode = "drop"
    auto_step = 0
    drop_trace = []
    drop_csv_data = []
    print(f"\n[Drop] VMC PD φ₀→{np.degrees(PHI0_EXTREME):.0f}°(θ={np.degrees(np.pi/2-PHI0_EXTREME):.0f}°) → hold → release ...")


def start_param_sweep() -> None:
    global auto_mode, auto_step, sweep_idx, drop_trace
    set_parameter_to(sweep_values[0], scale_all=False)
    mujoco.mj_resetData(model, data)
    _reset_unwrap()
    auto_mode = "sweep"
    auto_step = 0
    sweep_idx = 0
    drop_trace = []
    print(f"\n[Sweep] gravity drop across {sweep_values}")
    print(f"  param=[{PARAM_NAMES[param_idx]}]")
    print(f"  {'param':>10s}  {'fall_time(s)':>12s}  {'theta_start':>10s}  {'theta_final':>10s}  {'delta_deg':>9s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*9}")


CSV_COLS = ["step", "time_s", "theta_rad", "phi0_rad", "L0_m",
            "qvel_lf", "qvel_rf", "qvel_lr", "qvel_rr",
            "ctrl_lf", "ctrl_rf", "ctrl_lr", "ctrl_rr"]


def _record_csv_row(step: int) -> None:
    st = get_vmc_state()
    qv = data.qvel[dof_adr[:4]]
    ct = data.ctrl[:4]
    drop_csv_data.append((
        step, step * model.opt.timestep,
        round(float(st["theta"]), 6),
        round(float(st["phi0"]), 6),
        round(st["L0"], 6),
        round(float(qv[0]), 6), round(float(qv[1]), 6),
        round(float(qv[2]), 6), round(float(qv[3]), 6),
        round(float(ct[0]), 6), round(float(ct[1]), 6),
        round(float(ct[2]), 6), round(float(ct[3]), 6),
    ))


def _export_csv() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pn = PARAM_NAMES[param_idx]
    pv = {0: cur_damping[0], 1: cur_armature[0], 2: cur_fricloss[0], 3: cur_kp}[param_idx]
    fname = f"drop_{pn}_{pv:.4f}_{ts}.csv"
    fpath = os.path.join(CSV_DIR, fname)
    with open(fpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        w.writerows(drop_csv_data)
    print(f"  [CSV] saved → {fpath}")
    return fpath


def _drop_tick() -> None:
    global auto_step, drop_trace, drop_trace_g
    if auto_step < HOLD_STEPS:
        apply_vmc_hold(PHI0_EXTREME, L0_TARGET)
    elif auto_step == HOLD_STEPS:
        zero_ctrl()
        st = get_vmc_state()
        grav = data.qfrc_bias[dof_adr]
        passive = data.qfrc_passive[dof_adr]
        drop_trace_g = grav[:4].copy()
        print(f"\n  ══ RELEASE step={auto_step} ══")
        print(f"  model.opt.gravity = {model.opt.gravity}")
        print(f"  model.opt.density = {model.opt.density}  viscosity={model.opt.viscosity}")
        print(f"  θ={np.degrees(st['theta']):.1f}°  φ₀={np.degrees(st['phi0']):.1f}°  L0={st['L0']:.4f}")
        print(f"  data.ctrl = {np.array2string(data.ctrl, precision=4, suppress_small=True)}")
        print(f"  qfrc_bias(leg) = {np.array2string(grav[:4], precision=4, suppress_small=True)}")
        print(f"  qfrc_passive   = {np.array2string(passive[:4], precision=4, suppress_small=True)}")
        print(f"  qvel(leg)      = {np.array2string(data.qvel[dof_adr], precision=4, suppress_small=True)}")
        print(f"  qacc(leg)      = {np.array2string(data.qacc[dof_adr], precision=4, suppress_small=True)}")
        print(f"  dof_damping    = {np.array2string(model.dof_damping[dof_adr[:4]], precision=4)}")
        print(f"  dof_armature   = {np.array2string(model.dof_armature[dof_adr[:4]], precision=4)}")
        print(f"  dof_fricloss   = {np.array2string(model.dof_frictionloss[dof_adr[:4]], precision=4)}")
        print(f"  child_damping  = {np.array2string(model.dof_damping[child_dof_adr], precision=4)}")
        print(f"  child_armature = {np.array2string(model.dof_armature[child_dof_adr], precision=4)}")
        print(f"  child_fricloss = {np.array2string(model.dof_frictionloss[child_dof_adr], precision=4)}")
        print(f"  actuator_gain  = {np.array2string(model.actuator_gainprm[:4], precision=4)}")
        print(f"  ───────────────────────────────")
    elif HOLD_STEPS < auto_step <= HOLD_STEPS + DROP_STEPS and (auto_step - HOLD_STEPS) % 50 == 1:
        st = get_vmc_state()
        print(f"  step={auto_step:4d}  θ={np.degrees(st['theta']):7.2f}°  φ₀={np.degrees(st['phi0']):7.2f}°  "
              f"ctrl={np.array2string(data.ctrl[:4], precision=4, suppress_small=True)}  "
              f"qvel={np.array2string(data.qvel[dof_adr[:4]], precision=4, suppress_small=True)}")
    if auto_step >= HOLD_STEPS:
        _record_csv_row(auto_step)
    drop_trace.append(float(np.degrees(get_vmc_state()["theta"])))
    auto_step += 1


def _drop_done(is_sweep: bool) -> None:
    arr = np.array(drop_trace)
    theta_start = arr[HOLD_STEPS]
    theta_final = arr[-1]
    delta_deg = theta_final - theta_start

    theta_dot = np.abs(np.diff(arr[HOLD_STEPS:]) / model.opt.timestep)
    below = np.where(theta_dot < SETTLE_THRESH)[0]
    fall_steps = below[0] + 1 if len(below) > 0 else DROP_STEPS
    fall_time = fall_steps * model.opt.timestep

    samples = arr[HOLD_STEPS::30]
    sample_str = " ".join(f"{v:.1f}" for v in samples[:15])

    if is_sweep:
        pv = {0: cur_damping[0], 1: cur_armature[0], 2: cur_fricloss[0], 3: cur_kp}[param_idx]
        print(f"  {pv:10.4f}  {fall_time:12.3f}  {theta_start:10.1f}  {theta_final:10.1f}  {delta_deg:9.1f}")
    else:
        final_g = data.qfrc_bias[dof_adr]
        print(f"  theta_start = {theta_start:.1f}°  theta_final = {theta_final:.1f}°  delta = {delta_deg:.1f}°")
        if drop_trace_g is not None:
            print(f"  gravity_torque(release) = {drop_trace_g[0]:.3f} {drop_trace_g[1]:.3f} {drop_trace_g[2]:.3f} {drop_trace_g[3]:.3f} N·m")
        print(f"  gravity_torque(final)   = {final_g[0]:.3f} {final_g[1]:.3f} {final_g[2]:.3f} {final_g[3]:.3f} N·m")
        print(f"  fall_time = {fall_time:.3f}s ({fall_steps} steps)")
        print(f"  theta trace (30-step): [{sample_str}]")
        _export_csv()


def tick_drop() -> None:
    global auto_step, auto_mode
    _drop_tick()
    if auto_step >= HOLD_STEPS + DROP_STEPS:
        _drop_done(is_sweep=False)
        auto_mode = None


def tick_sweep() -> None:
    global auto_step, auto_mode, sweep_idx
    _drop_tick()
    if auto_step >= HOLD_STEPS + DROP_STEPS:
        _drop_done(is_sweep=True)
        sweep_idx += 1
        if sweep_idx >= len(sweep_values):
            print(f"  [Sweep Done]")
            auto_mode = None
            reset_all()
        else:
            set_parameter_to(sweep_values[sweep_idx], scale_all=False)
            mujoco.mj_resetData(model, data)
            _reset_unwrap()
            auto_step = 0
            drop_trace.clear()


def key_callback(keycode):
    global param_idx, base_fixed, auto_mode
    if auto_mode is not None:
        return
    if keycode == 49:   param_idx = 0; print_header()
    elif keycode == 50: param_idx = 1; print_header()
    elif keycode == 51: param_idx = 2; print_header()
    elif keycode == 52: param_idx = 3; print_header()
    elif keycode == 265:
        set_parameter_to(2.0, scale_all=True); print_header()
    elif keycode == 264:
        set_parameter_to(0.5, scale_all=True); print_header()
    elif keycode == 82:
        reset_all(); print_header()
    elif keycode == 83:
        base_fixed = not base_fixed
        print(f"\n[Switch] base_fixed = {base_fixed}")
    elif keycode == 32:
        start_drop_test()
    elif keycode == 65:
        start_param_sweep()
    elif keycode == 81:
        print("\nExit"); sys.exit(0)


model.qpos0[2] = BASE_HEIGHT
model.qpos0[3:7] = [1, 0, 0, 0]
mujoco.mj_resetData(model, data)
sync()

print_header()
print(f"\n   [Space]单次下摆  [A]参数扫描  [↑/↓]调节  [1-4]选参  [S]固定  [R]复位  [Q]退出")
print(f"   流程: VMC PD φ₀→{np.degrees(PHI0_EXTREME):.0f}°(θ={np.degrees(np.pi/2-PHI0_EXTREME):.0f}°) L₀→{L0_TARGET:.2f}m ({HOLD_STEPS*model.opt.timestep:.1f}s) → 零力下落({DROP_STEPS*model.opt.timestep:.1f}s)\n")

with mujoco.viewer.launch_passive(model, data,
                                   key_callback=key_callback) as viewer:
    viewer.sync()
    while viewer.is_running():
        lock_base()
        if auto_mode == "drop":
            tick_drop()
        elif auto_mode == "sweep":
            tick_sweep()
        mujoco.mj_step(model, data)
        viewer.sync()