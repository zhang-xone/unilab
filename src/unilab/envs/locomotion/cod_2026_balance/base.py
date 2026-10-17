from __future__ import annotations

from dataclasses import dataclass, field
import gymnasium as gym
import numpy as np

from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    PdControlConfig,
)

# === 常量（按 go2w 风格分区：腿 / 轮子）
NUM_COD2026_ACTIONS = 6
LEG_JOINT_INDICES = (0, 1, 3, 4)   # R_front, R_rear, L_front, L_rear
WHEEL_JOINT_INDICES = (2, 5)         # R_wheel, L_wheel
NUM_LEG_ACTIONS = len(LEG_JOINT_INDICES)
NUM_WHEEL_ACTIONS = len(WHEEL_JOINT_INDICES)

# action=0 对应的 target joint_pos（机器人自然站立时的关节角度）
# 顺序: [右前腿, 右后腿, 右轮子, 左前腿, 左后腿, 左轮子]
DEFAULT_ANGLES: np.ndarray = np.array(
    [0.239130, -0.519644, 0.0,
     0.162861, -0.497974, 0.0],
    dtype=np.float64,
)
JOINT_ZERO_OFFSET = DEFAULT_ANGLES.copy()

# 各关节的物理限位（来自 XML range 属性），顺序同 DEFAULT_ANGLES
JOINT_LIMITS: np.ndarray = np.array(
    [[-1.57, 2.40],     # Right_front_joint
     [-1.57, 1.57],     # Right_rear_joint
     [-np.inf, np.inf], # Right_Wheel_joint
     [-1.57, 2.40],     # Left_front_joint
     [-1.57, 1.57],     # Left_rear_joint
     [-np.inf, np.inf]],# Left_Wheel_joint
    dtype=np.float64,
)

# 被驱动关节的 sensor name（与 XML 中 actuator 顺序一致）
JOINT_SENSOR_PREFIXES: tuple[str, ...] = (
    "Right_front_joint",
    "Right_rear_joint",
    "Right_Wheel_joint",
    "Left_front_joint",
    "Left_rear_joint",
    "Left_Wheel_joint",
)


def stack_joint_sensors(backend, suffix: str, *, dtype: np.dtype | type) -> np.ndarray:
    """按 JOINT_SENSOR_PREFIXES 批量读取 joint_pos / joint_vel sensor"""
    names = tuple(f"{prefix}_{suffix}" for prefix in JOINT_SENSOR_PREFIXES)
    values = backend.get_sensor_data_batch(names)
    return np.asarray(values.reshape(values.shape[0], -1), dtype=dtype)


# === 配置类（对齐 go2w：独立的 leg/wheel 增益）
@dataclass
class NoiseConfig(BaseNoiseConfig):
    pass


@dataclass
class ControlConfig(PdControlConfig):
    """PD 控制参数（对齐 go2w 风格：每关节独立配置）

    leg_kp/leg_kd 长度 4（对应 LEG_JOINT_INDICES = (R_front, R_rear, L_front, L_rear)）。
    wheel_kd 长度 2：[R_wheel_damping, L_wheel_damping]。"""
    # leg: 每关节独立 PD 位置控制（长度 4：[R_front, R_rear, L_front, L_rear]）
    leg_kp: tuple = (40, 40, 40, 40)
    leg_kd: tuple = (3, 3, 3, 3)
    action_scale: float = 1  # action → target_angle 缩放
    # wheel: 每轮子独立 PD 速度控制（长度 2：[R_wheel, L_wheel]）
    wheel_kd: tuple = (1.6, 1.6)  # 轮子阻尼增益（小，防止前倾 actual_vel 大时 PD 饱和）
    wheel_action_scale: float = 5.0  # action∈[-1,1] → target_vel ∈ [-5,5] rad/s（静止时 torque≈±7.5）

    clip_actions: float = 1.0       # 对齐 action_space 的 [-1, 1]
    action_clip_initial: float = 0.3  # curriculum 初期 action clip 范围，线性增长到 1.0
 


@dataclass
class Asset:
    base_name = "base"
    ground = "floor"


@dataclass
class Cod2026BaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig) # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig) # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02


# === PD 控制计算（对齐 go2w/compute_go2w_motor_ctrl 热路径风格）
def compute_cod2026_motor_ctrl(
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

    与 go2w 架构完全一致：
      - 腿关节: PD 位置控制
          torque = kp * (target_pos - joint_pos) - kd * joint_vel
      - 轮子关节: D 速度控制
          torque = wheel_kd * (target_vel - joint_vel)
      - 最后按 actuator 上下界 clip torque
    """
    out.fill(0.0)

    # --- leg PD 位置控制（in-place）---
    # leg_kp/leg_kd 为 4 维 [R_front, R_rear, L_front, L_rear]，用 enumerate 取局部索引
    leg_indices = np.asarray(LEG_JOINT_INDICES, dtype=np.intp)
    for local_idx, global_idx in enumerate(leg_indices):
        out[:, global_idx] = leg_kp[:, local_idx] * (policy_ctrl[:, global_idx] - joint_pos[:, global_idx]) \
                             - leg_kd[:, local_idx] * joint_vel[:, global_idx]

    # --- wheel D 速度控制（in-place）---
    # wheel_kd 为 2 维 [R_wheel, L_wheel]，用 idx - NUM_LEG_ACTIONS 做局部索引映射
    wheel_indices = np.asarray(WHEEL_JOINT_INDICES, dtype=np.intp)
    for idx in wheel_indices:
        out[:, idx] = wheel_kd[:, idx - NUM_LEG_ACTIONS] * (policy_ctrl[:, idx] - joint_vel[:, idx])

    # --- torque clip（按 actuator 硬件上下界）---
    np.clip(out, ctrl_lower, ctrl_upper, out=out)
    return out


# === 环境基类
class Cod2026BaseEnv(LocomotionBaseEnv):
    _cfg: Cod2026BaseCfg

    def _init_action_space(self) -> None:
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(NUM_COD2026_ACTIONS,),
            dtype=np.float32,
        )

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.default_angles = np.asarray(DEFAULT_ANGLES, dtype=self.default_angles.dtype)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """策略 action → 关节 targets（leg: target_pos, wheel: target_vel）

        子类 _pre_step_motor_control 会把 targets 经 PD 转为 torque。
        对齐 go2w/joystick.py apply_action。
        """
        raise NotImplementedError("子类应覆盖 apply_action 返回 targets")

    def get_dof_pos(self) -> np.ndarray:
        return stack_joint_sensors(self._backend, "pos", dtype=self.default_angles.dtype)

    def get_dof_pos_biased(self) -> np.ndarray:
        """偏置后的 joint_pos → 策略观测用
        dof_pos_biased = dof_pos - JOINT_ZERO_OFFSET
        → 0 值对应 "自然站立" 姿态（action=0 时 PD 控制要到达的目标）
        """
        return self.get_dof_pos() - np.asarray(JOINT_ZERO_OFFSET, dtype=self.default_angles.dtype)

    def get_dof_vel(self) -> np.ndarray:
        return stack_joint_sensors(self._backend, "vel", dtype=self.default_angles.dtype)