from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_wrap_to_pi, np_yaw_from_quat


@dataclass
class Commands:
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [
            [-0.6, -0.4, -0.8],  # [vx_min, vy_min, vyaw_min]
            [1.0, 0.4, 0.8],  # [vx_max, vy_max, vyaw_max]
        ]
    )
    initial_vel_range: list[float] = field(default_factory=lambda: [-0.2, 0.2])
    initial_yaw_range: float = 1.0
    cmd_curriculum_enabled: bool = True
    jump_curriculum_enabled: bool = True
    auto_jump_threshold: float = 0.0   # 测距触发自动跳跃阈值 (m)，>0 启用
    auto_jump_hold_time: float = 0.4   # 自动跳跃触发后 cmd_jump 保持时长 (s)
    auto_jump_cooldown: float = 1.0    # 两次自动跳跃触发的最小间隔 (s)
    cmd_curriculum_sustain_threshold: int = 4
    cmd_curriculum_sustain_threshold_bad: int = 4
    cmd_curriculum_eval_steps: int = 2000
    """速度课程评估窗口（env steps）；窗口内只统计 tracking，不按 episode 数触发。"""
    cmd_curriculum_min_cmd_speed: float = 0.0
    """只让 episode 平均 |cmd_vx| 超过该阈值的 episode 参与升/降速判定（0 = 不筛选）。

    置 0 时原地自转（vx=0）和站立 episode 也参与判定：它们天然 tracking≈1，
    会把速度范围一路推高，与策略真实可达速度脱钩。>0 时这类 episode 不参与
    升/降速，只把连续计数清零。
    """
    height_limit: list[float] = field(default_factory=lambda: [0.28, 0.32])
    jumph_range: list[float] = field(default_factory=lambda: [0.0, 0.3])
    resampling_time: float = 0.0
    heading_command: bool = False
    heading_range: list[float] = field(default_factory=lambda: [-3.14, 3.14])
    heading_control_stiffness: float = 0.5
    rel_standing_envs: float = 0.0


def sample_velocity_commands(
    rng: np.random.Generator, num_samples: int, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    return np.asarray(
        rng.uniform(low=low, high=high, size=(num_samples, 3)), dtype=get_global_dtype()
    )


def zero_small_xy_commands(commands: np.ndarray, *, threshold: float = 0.2) -> None:
    """Zero ``commands[:, :2]`` in-place wherever its norm is below ``threshold``."""
    moving = np.linalg.norm(commands[:, :2], axis=1) > threshold
    commands[:, :2] *= moving[:, None]


def sample_heading_commands(env: Any, num_samples: int) -> np.ndarray:
    """Uniformly sample heading targets from ``env.cfg.commands.heading_range``."""
    heading_range = np.asarray(env.cfg.commands.heading_range, dtype=get_global_dtype())
    if heading_range.shape != (2,):
        raise ValueError(f"commands.heading_range must have shape (2,), got {heading_range.shape}")
    low, high = float(np.min(heading_range)), float(np.max(heading_range))
    return np.asarray(np.random.uniform(low, high, size=(num_samples,)), dtype=get_global_dtype())


def apply_heading_yaw_feedback(
    commands: np.ndarray,
    base_quat: np.ndarray,
    heading_commands: np.ndarray,
    *,
    stiffness: float,
    clip: float = 2.0,
) -> None:
    """In-place P-control on heading error → ``commands[:, 2]`` (yaw rate)."""
    heading = np_yaw_from_quat(base_quat)
    commands[:, 2] = np.clip(stiffness * np_wrap_to_pi(heading_commands - heading), -clip, clip)