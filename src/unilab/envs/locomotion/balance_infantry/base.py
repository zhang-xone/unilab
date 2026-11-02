from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    PdControlConfig,
)

# === 常量
# 注意: chassis.xml 的 actuator 顺序为 [LF, RF, LR, RR, LW, RW]
NUM_BALANCE_ACTIONS = 6
LEG_JOINT_INDICES = (0, 1, 2, 3)   # LF, RF, LR, RR
WHEEL_JOINT_INDICES = (4, 5)         # LW, RW
NUM_LEG_ACTIONS = len(LEG_JOINT_INDICES)
NUM_WHEEL_ACTIONS = len(WHEEL_JOINT_INDICES)

WHEEL_RADIUS = 0.06
WHEELBASE = 0.380
WHEEL_ERR_VEL_THRESHOLD = 0.3
WHEEL_ERR_DECAY_STEPS = 4

# action=0 对应的 target joint_pos（机器人自然站立时的关节角度）
# 顺序: [左前腿, 右前腿, 左后腿, 右后腿, 左轮子, 右轮子]
DEFAULT_ANGLES: np.ndarray = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
JOINT_ZERO_OFFSET = DEFAULT_ANGLES.copy()

# 各关节的物理限位（来自 balance_infantry.xml 的 joint range 属性），顺序同 DEFAULT_ANGLES
JOINT_LIMITS: np.ndarray = np.array(
    [[-2.27, 2.40],     # left_front_joint
     [-2.27, 2.40],     # right_front_joint
     [-2.62, 1.57],     # left_rear_joint
     [-2.62, 1.57],     # right_rear_joint
     [-np.inf, np.inf], # left_wheel_joint
     [-np.inf, np.inf]],# right_wheel_joint
    dtype=np.float64,
)

# 被驱动关节的 sensor name（与 chassis.xml actuator 顺序一致）
JOINT_SENSOR_PREFIXES: tuple[str, ...] = (
    "left_front_joint",
    "right_front_joint",
    "left_rear_joint",
    "right_rear_joint",
    "left_wheel_joint",
    "right_wheel_joint",
)


def stack_joint_sensors(backend, suffix: str, *, dtype: np.dtype | type) -> np.ndarray:
    """按 JOINT_SENSOR_PREFIXES 批量读取 joint_pos / joint_vel sensor"""
    names = tuple(f"{prefix}_{suffix}" for prefix in JOINT_SENSOR_PREFIXES)
    values = backend.get_sensor_data_batch(names)
    return np.asarray(values.reshape(values.shape[0], -1), dtype=dtype)


# === 配置类
@dataclass
class NoiseConfig(BaseNoiseConfig):
    scale_vmc_phi0: float = 0.008
    scale_vmc_L0: float = 0.003
    # θ̇ / L̇₀ 的噪声: 刻意不启用。
    # 29 维 actor 观测不含这两列 (critic 用 noisy=False)，加噪只会在
    # _vmc_obs_columns 里算完丢弃。要启用必须把这两列加进 actor 观测
    # (33 维)，那会同时改变 checkpoint / deploy_obs / 量化导出契约。
    scale_vmc_phi0_dot: float = 0.3
    scale_vmc_L0_dot: float = 0.2


@dataclass
class VMCConfig:
    """VMC (Virtual Model Control) 参数配置 —— 五连杆腿机构

    五连杆实际杆长 (m):
        L1: 前上连杆 (连电机 φ₁, LF/RF)
        L2: 上连杆 (B-C)
        L3: 下连杆 (C-D)
        L4: 后下连杆 (连电机 φ₄, LR/RR)
        L5: 足端偏移

    虚拟空间 PD 增益:
        kp_theta: 腿角度 P 增益
        kd_theta: 腿角度 D 增益
        kp_l0: 腿长 P 增益
        kd_l0: 腿长 D 增益

    前馈:
        feedforward_force: 重力补偿前馈力 (N)

    动作缩放:
        action_scale_theta: action → θ_ref 缩放 (±50°=±0.87rad)
        action_scale_l0: action → L0_ref 缩放
        action_scale_wheel: action → wheel_ref 缩放
        default_L0: action=0 时的默认腿长 (m)
    """
    enabled: bool = True
    L1: float = 0.215
    L2: float = 0.254
    L3: float = 0.254
    L4: float = 0.215
    L5: float = 0.0
    kp_theta: float = 50.0
    kd_theta: float = 3.0
    kp_l0: float = 500.0
    kd_l0: float = 50.0
    feedforward_force: float = 90.0
    ff_anneal_target: float = 0.0
    action_scale_theta: float = 0.7
    action_scale_l0: float = 0.15
    action_scale_wheel: float = 10.0
    default_L0: float = 0.28
    theta_clip: float = 0.5
    L0_clip: tuple[float, float] = (0.15, 0.45)


@dataclass
class WheelTorqueSpeedConfig:
    """轮子执行器扭矩-转速（功率）包络。

    真机轮电机受功率限制，可用力矩随转速下降；恒力矩上限会让策略在 sim 里
    学到“高速还能出满力矩”的行为。模型（默认解析式）：

        τ_max(ω) = min(stall_torque, max_power / max(|ω|, eps))
        |ω| >= max_speed 时 τ_max = 0

    也可以用 ``torque_speed_curve``（[(speed_rad_s, torque_nm), ...]）直接给
    M3508 这类实测曲线，此时忽略 stall/power/max_speed 的解析式。

    默认 ``enabled=False``，保持旧行为；打开后只 clip 轮子力矩，不影响腿/奖励。
    """

    enabled: bool = False
    stall_torque: float = 4.0
    max_power: float = 100.0
    max_speed: float = 60.0
    torque_speed_curve: list[tuple[float, float]] = field(default_factory=list)
    max_wheel_speed: float = 0.0
    """>0 时把策略的轮速目标夹紧到 ±max_wheel_speed (rad/s)。"""

    # 华南虎 M3508+C620 实测曲线（冷路径加载）。CSV 支持两种格式：
    # 1) 华南虎格式：Motor_shaft_torque_Nm_ideal, Motor_shaft_speed_rpm
    # 2) 简单两列：speed_rad_s, torque_nm
    # 设置后优先于 torque_speed_curve；为空时用 inline 曲线/解析式。
    curve_path: str = ""
    curve_gear_ratio: float = 268.0 / 17.0
    curve_gearbox_efficiency: float = 1.0


class WheelTorqueEnvelope:
    """按扭矩-转速包络裁剪轮子力矩（热路径只做插值/算术，不读文件）。"""

    def __init__(self, cfg: WheelTorqueSpeedConfig):
        self._cfg = cfg
        self._enabled = bool(cfg.enabled)
        if cfg.curve_path:
            curve = self._load_curve_csv(cfg)
        elif cfg.torque_speed_curve:
            curve = np.asarray(cfg.torque_speed_curve, dtype=np.float64)
        else:
            curve = None
        if curve is not None and curve.ndim == 2 and curve.shape[1] == 2 and len(curve) >= 2:
            order = np.argsort(curve[:, 0])
            speed = np.ascontiguousarray(curve[order, 0])
            torque = np.maximum(np.ascontiguousarray(curve[order, 1]), 0.0)
            if cfg.stall_torque > 0.0:
                torque = np.minimum(torque, float(cfg.stall_torque))
            self._curve_speed = speed
            self._curve_torque = torque
        else:
            self._curve_speed = None
            self._curve_torque = None

    @staticmethod
    def _load_curve_csv(cfg: WheelTorqueSpeedConfig) -> np.ndarray:
        """冷路径加载华南虎 M3508 曲线或简单两列曲线。"""
        path = Path(cfg.curve_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"wheel torque-speed curve not found: {path}")

        rows: list[tuple[float, float]] = []
        with path.open(newline="") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"wheel torque-speed curve is empty: {path}")
            scut = any("Motor_shaft_torque" in str(h) for h in header) and any(
                "Motor_shaft_speed" in str(h) for h in header
            )
            if scut:
                headers = [str(h) for h in header]

                def _find(token: str) -> int | None:
                    return next((i for i, h in enumerate(headers) if token in h), None)

                motor_torque_idx = _find("Motor_shaft_torque")
                motor_speed_idx = _find("Motor_shaft_speed")
                output_torque_idx = _find("Original_output_torque")
                output_speed_idx = _find("Original_output_speed")
                gear_ratio_idx = _find("Gear_ratio_used")
                use_output = output_torque_idx is not None and output_speed_idx is not None
                for row in reader:
                    try:
                        if use_output:
                            # 优先用 CSV 自带的输出轴实测列，避免 gear_ratio 配置不一致
                            output_torque = float(row[output_torque_idx])
                            output_rpm = float(row[output_speed_idx])
                            output_speed = output_rpm * 2.0 * np.pi / 60.0
                        else:
                            motor_torque = float(row[motor_torque_idx])
                            motor_rpm = float(row[motor_speed_idx])
                            if gear_ratio_idx is not None and row[gear_ratio_idx]:
                                gear_ratio = float(row[gear_ratio_idx])
                            else:
                                gear_ratio = cfg.curve_gear_ratio
                            if not np.isfinite(gear_ratio) or gear_ratio <= 0.0:
                                continue
                            output_torque = motor_torque * gear_ratio * cfg.curve_gearbox_efficiency
                            output_speed = motor_rpm / gear_ratio * 2.0 * np.pi / 60.0
                    except (IndexError, ValueError):
                        continue
                    rows.append((output_speed, output_torque))
            else:
                for row in [header, *reader]:
                    if len(row) < 2:
                        continue
                    try:
                        speed = float(row[0])
                        torque = float(row[1])
                    except ValueError:
                        continue
                    rows.append((speed, torque))

        valid = [
            (float(speed), float(torque))
            for speed, torque in rows
            if np.isfinite(speed + torque) and speed >= 0.0 and torque >= 0.0
        ]
        if len(valid) < 2:
            raise ValueError(f"wheel torque-speed curve has too few valid points: {path}")
        valid.sort(key=lambda item: item[0])
        speed = np.asarray([item[0] for item in valid], dtype=np.float64)
        torque = np.asarray([item[1] for item in valid], dtype=np.float64)
        # np.interp 要求 x 严格递增；重复速度保留第一个点
        unique_speed, unique_idx = np.unique(speed, return_index=True)
        torque = torque[unique_idx]
        if cfg.stall_torque > 0.0:
            torque = np.minimum(torque, float(cfg.stall_torque))
        return np.column_stack([unique_speed, torque])

    @property
    def enabled(self) -> bool:
        return self._enabled

    def limit(self, torque: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """τ_max(ω) 的对称裁剪。torque / omega 形状相同（通常 (N, 2)）。"""
        if not self._enabled:
            return torque
        abs_w = np.abs(np.asarray(omega, dtype=np.float64))
        if self._curve_speed is not None and self._curve_torque is not None:
            limit = np.interp(
                abs_w,
                self._curve_speed,
                self._curve_torque,
                left=float(self._curve_torque[0]),
                right=0.0,
            )
        else:
            limit = np.full_like(abs_w, float(self._cfg.stall_torque))
            if self._cfg.max_power > 0.0:
                limit = np.minimum(limit, self._cfg.max_power / np.maximum(abs_w, 1e-6))
            if self._cfg.max_speed > 0.0:
                limit = np.where(abs_w >= self._cfg.max_speed, 0.0, limit)
        limit = np.maximum(limit, 0.0)
        return np.clip(torque, -limit, limit)

    def clamp_target(self, target: np.ndarray) -> np.ndarray:
        """把轮速目标夹到 ±max_wheel_speed（速度目标侧的限制）。"""
        if not self._enabled or self._cfg.max_wheel_speed <= 0.0:
            return target
        return np.clip(target, -self._cfg.max_wheel_speed, self._cfg.max_wheel_speed)


@dataclass
class ControlConfig(PdControlConfig):
    """PD 控制参数（每关节独立配置）。wheel_kd 长度 2：[LW, RW]。"""
    wheel_kd: tuple = (0.1, 0.1)
    leg_kp: tuple = (40., 40., 40., 40.)
    leg_kd: tuple = (2., 2., 2., 2.)
    clip_actions: float = 1.0
    action_scale_joint: float = 0.5
    action_scale_wheel: float = 10.0
    action_clip_initial: float = 0.3
    action_smoothing: float = 0.8
    action_smoothing_range: tuple = (0.2, 1.0)
    action_delay_range: tuple = (0, 0)  # (min, max) substeps, FIFO 随机延迟
    obs_delay_range: tuple = (0, 0)     # (min, max) substeps, 观测延迟（默认跟随 action_delay_range）
    wheel_torque_speed: WheelTorqueSpeedConfig = field(default_factory=WheelTorqueSpeedConfig)
    vmc: VMCConfig = field(default_factory=VMCConfig)


@dataclass
class Asset:
    base_name = "base_link"
    ground = "floor"


@dataclass
class BalanceInfantryBaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    obs_manual_norm: bool = True
    obs_vel: bool = False


# === VMC (Virtual Model Control) 五连杆辅助函数 ====================


def forward_kinematics_fivebar(
    phi1: np.ndarray,
    phi4: np.ndarray,
    L1: float,
    L2: float,
    L3: float,
    L4: float,
    L5: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """五连杆正运动学: 电机角度 (φ₁, φ₄) → L₀, φ₀ + 中间变量

    几何:
        B = (L₁·cos(φ₁), L₁·sin(φ₁))
        D = (L₅ + L₄·cos(φ₄), L₄·sin(φ₄))
        解四边形 B-C-D 求 φ₂ (BC 连杆角度)
        C = B + L₂·(cos(φ₂), sin(φ₂))
        L₀ = |C - (L₅/2, 0)|,  φ₀ = atan2(y_C, x_C - L₅/2)

    Args:
        phi1: 前电机角度, shape (num_envs, 2) [左腿, 右腿]
        phi4: 后电机角度, shape (num_envs, 2) [左腿, 右腿]
        L1..L5: 五连杆杆长

    Returns:
        L0:     虚拟腿长, shape (num_envs, 2)
        phi0:   虚拟腿角度 (从水平方向), shape (num_envs, 2)
        phi2:   连杆 BC 角度, shape (num_envs, 2)
        phi3:   连杆 DC 角度 (方向 D→C), shape (num_envs, 2)
        sigma1: sin(φ₃ - φ₂), shape (num_envs, 2)
    """
    x_B = L1 * np.cos(phi1)
    y_B = L1 * np.sin(phi1)
    x_D = L5 + L4 * np.cos(phi4)
    y_D = L4 * np.sin(phi4)

    dx = x_D - x_B
    dy = y_D - y_B
    BD_sq = dx * dx + dy * dy

    A0 = 2.0 * L2 * dx
    b0 = 2.0 * L2 * dy
    C0 = L2 * L2 + BD_sq - L3 * L3

    disc = np.maximum(A0 * A0 + b0 * b0 - C0 * C0, 0.0)
    temp = b0 + np.sqrt(disc)
    phi2 = 2.0 * np.arctan2(temp, A0 + C0)

    x_C = x_B + L2 * np.cos(phi2)
    y_C = y_B + L2 * np.sin(phi2)
    x_C_shifted = x_C - L5 / 2.0

    L0 = np.sqrt(x_C_shifted * x_C_shifted + y_C * y_C)
    phi0 = np.arctan2(y_C, x_C_shifted)

    phi3 = np.arctan2(y_C - y_D, x_C - x_D)

    sigma1 = np.sin(phi3 - phi2)

    return L0, phi0, phi2, phi3, sigma1


def vmc_mapping_fivebar(
    F: np.ndarray,
    T: np.ndarray,
    phi1: np.ndarray,
    phi4: np.ndarray,
    phi0: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    sigma1: np.ndarray,
    L0: np.ndarray,
    L1: float,
    L4: float,
) -> tuple[np.ndarray, np.ndarray]:
    """五连杆雅可比映射: [F, T] → [τ₁, τ₄]

    雅可比 Jᵀ 定义 (转置布局, 对标 C 代码):
        Jᵀ = [[dL₀/dφ₁, dφ₀/dφ₁],
              [dL₀/dφ₄, dφ₀/dφ₄]]

    σ₁ = sin(φ₃ - φ₂),  σ₂ = sin(φ₃ - φ₄),  σ₃ = sin(φ₁ - φ₂)
    J00 = L₁·sin(φ₀-φ₃)·σ₃ / σ₁              (dL₀/dφ₁)
    J01 = L₁·cos(φ₀-φ₃)·σ₃ / L₀ / σ₁        (dφ₀/dφ₁)
    J10 = L₄·sin(φ₀-φ₂)·σ₂ / σ₁              (dL₀/dφ₄)
    J11 = L₄·cos(φ₀-φ₂)·σ₂ / L₀ / σ₁        (dφ₀/dφ₄)

    虚功原理: τ₁·δφ₁ + τ₄·δφ₄ = F·δL₀ + T·δφ₀
        τ = Jᵀ · [F, T]ᵀ
        → τ₁ = J00·F + J01·T = (dL₀/dφ₁)·F + (dφ₀/dφ₁)·T
        → τ₄ = J10·F + J11·T = (dL₀/dφ₄)·F + (dφ₀/dφ₄)·T

    Args:
        F: 虚拟力 (沿腿方向), shape (num_envs, 2)
        T: 虚拟力矩, shape (num_envs, 2)
        phi1, phi4: 电机角度, shape (num_envs, 2)
        phi0, phi2, phi3: 中间角度, shape (num_envs, 2)
        sigma1: sin(φ₃-φ₂), shape (num_envs, 2)
        L0: 虚拟腿长, shape (num_envs, 2)
        L1, L4: 杆长

    Returns:
        tau1: φ₁ 电机力矩, shape (num_envs, 2)
        tau4: φ₄ 电机力矩, shape (num_envs, 2)
    """
    sigma2 = np.sin(phi3 - phi4)
    sigma3 = np.sin(phi1 - phi2)

    eps = 1e-8
    inv_sigma1 = 1.0 / (sigma1 + eps)
    inv_L0 = 1.0 / (L0 + eps)

    J00 = L1 * np.sin(phi0 - phi3) * sigma3 * inv_sigma1            # dL₀/dφ₁
    J01 = L1 * np.cos(phi0 - phi3) * sigma3 * inv_L0 * inv_sigma1  # dφ₀/dφ₁
    J10 = L4 * np.sin(phi0 - phi2) * sigma2 * inv_sigma1            # dL₀/dφ₄
    J11 = L4 * np.cos(phi0 - phi2) * sigma2 * inv_L0 * inv_sigma1  # dφ₀/dφ₄

    tau1 = J00 * F + J01 * T
    tau4 = J10 * F + J11 * T

    tau1 = np.nan_to_num(tau1, nan=0.0, posinf=40.0, neginf=-40.0)
    tau4 = np.nan_to_num(tau4, nan=0.0, posinf=40.0, neginf=-40.0)

    return tau1, tau4


def vmc_feedforward_torques_no_ramp(
    feedforward_force: np.ndarray,
    theta: np.ndarray,
    phi1: np.ndarray,
    phi4: np.ndarray,
    phi0: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    sigma1: np.ndarray,
    L0: np.ndarray,
    L1: float,
    L4: float,
    num_envs: int,
    dtype: Any,
) -> np.ndarray:
    """无力期专用腿力矩：F = ff / cos(θ)（**不减淡**），Tp = 0，轮子列为 0。

    用来模拟真实机器人断电后"内部机械力把腿自然蹬出"的行为，所以这里不乘
    ``_compute_vmc_torques`` 里那个按 ``|θ|`` 淡出的 ramp：腿折得越狠，前馈
    反而越要顶住，而不是淡到 0。

    数值安全：``vmc_mapping_fivebar`` 内部对 ``σ₁`` 有 1e-8 保护、并把 ``±inf``
    夹到 ±40，因此深折叠时只会顶到力矩上限（相当于机械限位），不会产生
    inf/nan。

    Returns:
        vmc_torques: shape (num_envs, NUM_BALANCE_ACTIONS)，只有 4 个腿关节非零。
    """
    F = feedforward_force.reshape(-1, 1) / np.cos(np.clip(theta, -1.5, 1.5))
    tau1, tau4 = vmc_mapping_fivebar(
        F, np.zeros_like(F), phi1, phi4, phi0, phi2, phi3, sigma1, L0, L1, L4
    )

    vmc_torques = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=dtype)
    vmc_torques[:, 0] = tau4[:, 0]   # τ₄ → LF (左前)
    vmc_torques[:, 1] = tau4[:, 1]   # τ₄ → RF (右前)
    vmc_torques[:, 2] = tau1[:, 0]   # τ₁ → LR (左后)
    vmc_torques[:, 3] = tau1[:, 1]   # τ₁ → RR (右后)
    return vmc_torques


def compute_vmc_torques_fivebar(
    phi1: np.ndarray,
    phi4: np.ndarray,
    phi0: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    sigma1: np.ndarray,
    L0: np.ndarray,
    L0_dot: np.ndarray,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    theta_ref: np.ndarray,
    L0_ref: np.ndarray,
    L1: float,
    L4: float,
    kp_theta: np.ndarray,
    kd_theta: np.ndarray,
    kp_l0: np.ndarray,
    kd_l0: np.ndarray,
    feedforward_force: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """五连杆完整 VMC: 虚拟空间 PD → 雅可比映射 → 关节力矩

    Args:
        phi1, phi4: 电机角度, shape (num_envs, 2)
        phi0: 虚拟腿角度, shape (num_envs, 2)
        phi2, phi3: 中间角度, shape (num_envs, 2)
        sigma1: sin(φ₃-φ₂), shape (num_envs, 2)
        L0: 虚拟腿长, shape (num_envs, 2)
        L0_dot: 虚拟腿长速度, shape (num_envs, 2)
        theta: 当前摆角 θ, shape (num_envs, 2)
        theta_dot: 摆角速度, shape (num_envs, 2)
        theta_ref: 目标摆角 θ (相对竖直), shape (num_envs, 2)
        L0_ref: 目标虚拟腿长, shape (num_envs, 2)
        L1, L4: 杆长
        kp_theta, kd_theta: 腿角度 PD 增益
        kp_l0, kd_l0: 腿长 PD 增益
        feedforward_force: 重力补偿前馈力, shape (num_envs, 2)

    Returns:
        tau1: φ₁ (后大腿) 电机力矩, shape (num_envs, 2)
        tau4: φ₄ (前大腿) 电机力矩, shape (num_envs, 2)
    """
    T = kp_theta * (theta - theta_ref) + kd_theta * theta_dot
    F = kp_l0 * (L0_ref - L0) - kd_l0 * L0_dot + feedforward_force

    return vmc_mapping_fivebar(F, T, phi1, phi4, phi0, phi2, phi3, sigma1, L0, L1, L4)


# === PD 控制计算
def compute_balance_motor_ctrl(
    policy_ctrl: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    leg_kp: np.ndarray,
    leg_kd: np.ndarray,
    wheel_kd: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """策略 targets → MuJoCo torque（Python 层 PD 控制）

    腿关节: PD 位置控制
        torque = kp * (target_pos - joint_pos) - kd * joint_vel
    轮子关节: D 速度控制
        torque = wheel_kd * (target_vel - joint_vel)
    """
    out.fill(0.0)

    leg_indices = np.asarray(LEG_JOINT_INDICES, dtype=np.intp)
    for local_idx, global_idx in enumerate(leg_indices):
        out[:, global_idx] = leg_kp[:, local_idx] * (policy_ctrl[:, global_idx] - joint_pos[:, global_idx]) \
                             - leg_kd[:, local_idx] * joint_vel[:, global_idx]

    wheel_indices = np.asarray(WHEEL_JOINT_INDICES, dtype=np.intp)
    for idx in wheel_indices:
        out[:, idx] = wheel_kd[:, idx - NUM_LEG_ACTIONS] * (policy_ctrl[:, idx] - joint_vel[:, idx])

    np.clip(out, ctrl_lower, ctrl_upper, out=out)
    return out


# === 环境基类
class BalanceInfantryBaseEnv(LocomotionBaseEnv):
    _cfg: BalanceInfantryBaseCfg

    def _init_action_space(self) -> None:
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(NUM_BALANCE_ACTIONS,),
            dtype=np.float32,
        )

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.default_angles = np.asarray(DEFAULT_ANGLES, dtype=self.default_angles.dtype)

    def _get_curriculum_scale(self) -> float:
        provider = getattr(getattr(self, '_dr_manager', None), '_provider', None)
        if provider is not None:
            return provider._get_curriculum_scale(self)
        return 1.0

    def _noisy_attitude(self, gravity_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """带噪声的姿态观测 (pitch, roll)。

        噪声作用在重力向量上，再由同一个重力向量解算 pitch/roll，语义与
        go1/go2 的 ``scale_gravity`` 一致；不要直接给角度加噪，否则会破坏
        pitch/roll 与重力方向的对应关系。
        训练侧 critic 仍然使用无噪姿态（privileged）。
        """
        noisy_gravity = self._obs_noise(gravity_body, self._cfg.noise_config.scale_gravity)
        pitch = np.arcsin(np.clip(noisy_gravity[:, 0], -1.0, 1.0)).reshape(-1, 1)
        roll = np.arcsin(np.clip(-noisy_gravity[:, 1], -1.0, 1.0)).reshape(-1, 1)
        return pitch, roll

    def _obs_noise(self, data: np.ndarray, scale: float) -> np.ndarray:
        level = float(self._cfg.noise_config.level)
        if level <= 0.0:
            return data
        curriculum_scale = self._get_curriculum_scale()
        effective_level = level * curriculum_scale
        if effective_level <= 0.0:
            return data
        noise = np.random.uniform(-1.0, 1.0, data.shape).astype(data.dtype) * effective_level * scale
        return data + noise

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        raise NotImplementedError("子类应覆盖 apply_action 返回 targets")

    def get_dof_pos(self) -> np.ndarray:
        return stack_joint_sensors(self._backend, "pos", dtype=self.default_angles.dtype)

    def get_dof_pos_biased(self) -> np.ndarray:
        return self.get_dof_pos() - np.asarray(JOINT_ZERO_OFFSET, dtype=self.default_angles.dtype)

    def get_dof_vel(self) -> np.ndarray:
        return stack_joint_sensors(self._backend, "vel", dtype=self.default_angles.dtype)