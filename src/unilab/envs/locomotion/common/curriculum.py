from __future__ import annotations

from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype


# 命令课程按 env reset 时的 episode tracking 逐个更新。
# 下面的评估窗口常量仅为旧配置/日志字段保留，不参与命令范围更新。
_CURRICULUM_EVAL_EPISODES_AT_REF_ENVS = 25
_CURRICULUM_EVAL_REF_ENVS = 2048


class CurriculumManager:

    def __init__(self, cfg: Any, num_envs: int):
        dtype = get_global_dtype()
        self._cfg = cfg
        self._num_envs = num_envs
        self._np_dtype = dtype

        # ── 统一 curriculum scale ──
        self._scale = 0.0
        self._scale_step_counter = 0

        # ── 指令 curriculum ──
        vel_limit = self._cfg.commands.vel_limit
        self._cmd_vel_curriculum_max = float(vel_limit[1][0])
        self._cmd_yaw_curriculum_max = float(vel_limit[1][2])
        if getattr(self._cfg.commands, "cmd_curriculum_enabled", True):
            init_range = self._cfg.commands.initial_vel_range
            init_yaw = float(getattr(self._cfg.commands, "initial_yaw_range", 1.0))
            vel_bounds = np.array([[float(init_range[0]), float(init_range[1])]], dtype=dtype)
            yaw_bounds = np.array([-init_yaw, init_yaw], dtype=dtype)
        else:
            vel_bounds = np.array([
                [float(vel_limit[0][0]), float(vel_limit[1][0])],
            ], dtype=dtype)
            yaw_bounds = np.array([float(vel_limit[0][2]), float(vel_limit[1][2])], dtype=dtype)
        self._cmd_vel_range_env = np.tile(vel_bounds, (num_envs, 1))
        self._cmd_yaw_range_env = np.tile(yaw_bounds, (num_envs, 1))
        self._cmd_consecutive_good = np.zeros(num_envs, dtype=np.int32)
        self._cmd_consecutive_bad = np.zeros(num_envs, dtype=np.int32)
        self._cmd_promote_count = 0
        self._cmd_demote_count = 0
        self._init_yaw_range = float(getattr(self._cfg.commands, "initial_yaw_range", 1.0))

        self._curriculum_good_accum = 0.0
        self._curriculum_step_accum = 0.0
        self._curriculum_episode_count = 0
        self._curriculum_eval_episode_interval = max(
            _CURRICULUM_EVAL_EPISODES_AT_REF_ENVS,
            int(round(
                _CURRICULUM_EVAL_EPISODES_AT_REF_ENVS * num_envs / _CURRICULUM_EVAL_REF_ENVS
            )),
        )
        self._curriculum_consecutive_good = 0
        self._curriculum_consecutive_bad = 0
        self._curriculum_sustain_threshold = max(
            1, int(getattr(self._cfg.commands, "cmd_curriculum_sustain_threshold", 4))
        )
        self._curriculum_sustain_threshold_bad = max(
            1,
            int(
                getattr(
                    self._cfg.commands,
                    "cmd_curriculum_sustain_threshold_bad",
                    self._curriculum_sustain_threshold,
                )
            ),
        )
        self._curriculum_last_eval_step = 0
        self._curriculum_eval_step_interval = max(
            1, int(getattr(self._cfg.commands, "cmd_curriculum_eval_steps", 2000))
        )

        # ── 跳跃 curriculum ──
        self._jump_curriculum_enabled = False
        self._jump_curriculum_consecutive_good = 0
        self._curriculum_height_good_accum = 0.0
        self._curriculum_height_episode_count = 0

        # ── 跟踪 sigma curriculum ──
        rc = self._cfg.reward_config
        self._tracking_sigma_lin_base = float(rc.tracking_sigma_lin)
        self._tracking_sigma_ang_base = float(rc.tracking_sigma_ang)
        self._tracking_sigma_yaw_base = float(rc.tracking_sigma_yaw)
        self._tracking_sigma_lin = self._tracking_sigma_lin_base
        self._tracking_sigma_ang = self._tracking_sigma_ang_base
        self._tracking_sigma_yaw = self._tracking_sigma_yaw_base

    # =================================================================
    # 属性
    # =================================================================

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def cmd_vel_range(self) -> np.ndarray:
        """Backward-compatible aggregate view; actual sampling uses per-env bounds."""
        return np.mean(self._cmd_vel_range_env, axis=0, keepdims=True)

    @property
    def cmd_yaw_range(self) -> np.ndarray:
        """Backward-compatible aggregate view; actual sampling uses per-env bounds."""
        return np.mean(self._cmd_yaw_range_env, axis=0)

    @property
    def cmd_vel_range_env(self) -> np.ndarray:
        return self._cmd_vel_range_env

    @property
    def cmd_yaw_range_env(self) -> np.ndarray:
        return self._cmd_yaw_range_env

    def pop_cmd_curriculum_stats(self) -> tuple[int, int]:
        """Return and reset promote/demote event counts since the last call."""
        stats = (self._cmd_promote_count, self._cmd_demote_count)
        self._cmd_promote_count = 0
        self._cmd_demote_count = 0
        return stats

    @property
    def jump_curriculum_enabled(self) -> bool:
        return self._jump_curriculum_enabled

    @property
    def tracking_sigma_lin(self) -> float:
        return self._tracking_sigma_lin

    @property
    def tracking_sigma_ang(self) -> float:
        return self._tracking_sigma_ang

    @property
    def tracking_sigma_yaw(self) -> float:
        return self._tracking_sigma_yaw

    @property
    def effective_powerless_duration(self) -> float:
        dr = getattr(self._cfg, "domain_rand", None)
        if dr is None or not getattr(dr, "curriculum_enabled", False):
            return self._cfg.reward_config.powerless_duration
        warmup = int(getattr(dr, "curriculum_warmup_steps", 0))
        full = int(getattr(dr, "curriculum_full_steps", 1))
        if full <= warmup:
            return self._cfg.reward_config.powerless_duration
        progress = (self._scale_step_counter - warmup) / (full - warmup)
        if progress < self._cfg.reward_config.powerless_curriculum_threshold:
            return 0.0
        return self._cfg.reward_config.powerless_duration

    # =================================================================
    # 统一更新入口
    # =================================================================

    def step(self, step_counter: int) -> None:
        self._update_scale(step_counter)
        self._try_eval(step_counter)
        self._update_tracking_sigmas()

    def record_step_tracking(self, tracking_values: np.ndarray) -> None:
        if not getattr(self._cfg.commands, "cmd_curriculum_enabled", True):
            return
        self._curriculum_good_accum += float(np.sum(tracking_values))
        self._curriculum_step_accum += float(len(tracking_values))

    def on_reset(
        self,
        env_ids: np.ndarray,
        episode_vel_sum_fwd: np.ndarray,
        episode_cmd_vel_sum_fwd: np.ndarray,
        episode_vel_sum_bwd: np.ndarray,
        episode_cmd_vel_sum_bwd: np.ndarray,
        episode_steps: np.ndarray,
        episode_tilt_sum: np.ndarray,
        episode_swing_err_sum: np.ndarray,
        episode_height_err_sum: np.ndarray,
        episode_tracking_reward_sum: np.ndarray,
    ) -> None:
        self._update_cmd_curriculum(
            env_ids,
            episode_vel_sum_fwd,
            episode_cmd_vel_sum_fwd,
            episode_vel_sum_bwd,
            episode_cmd_vel_sum_bwd,
            episode_steps,
            episode_tilt_sum,
            episode_swing_err_sum,
            episode_height_err_sum,
            episode_tracking_reward_sum,
        )

    # =================================================================
    # 内部方法
    # =================================================================

    def _update_scale(self, step_counter: int) -> None:
        self._scale_step_counter = step_counter
        dr = getattr(self._cfg, "domain_rand", None)
        if dr is None or not getattr(dr, "curriculum_enabled", False):
            self._scale = 1.0
            return
        warmup = int(getattr(dr, "curriculum_warmup_steps", 0))
        full = int(getattr(dr, "curriculum_full_steps", 1))
        initial = float(getattr(dr, "curriculum_initial_scale", 0.0))
        shape = str(getattr(dr, "curriculum_shape", "linear"))
        if step_counter <= warmup:
            self._scale = float(initial)
            return
        if step_counter >= full:
            self._scale = 1.0
            return
        progress = float(step_counter - warmup) / float(max(1, full - warmup))
        if shape == "smooth":
            t = float(progress)
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            normalized = 3.0 * t * t - 2.0 * t * t * t
        elif shape == "quadratic":
            t = float(progress)
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            normalized = t * t
        else:
            normalized = float(progress)
            normalized = 0.0 if normalized < 0.0 else 1.0 if normalized > 1.0 else normalized
        self._scale = float(initial + (1.0 - initial) * normalized)

    def _update_cmd_curriculum(
        self,
        env_ids: np.ndarray,
        episode_vel_sum_fwd: np.ndarray,
        episode_cmd_vel_sum_fwd: np.ndarray,
        episode_vel_sum_bwd: np.ndarray,
        episode_cmd_vel_sum_bwd: np.ndarray,
        episode_steps: np.ndarray,
        episode_tilt_sum: np.ndarray,
        episode_swing_err_sum: np.ndarray,
        episode_height_err_sum: np.ndarray,
        episode_tracking_reward_sum: np.ndarray,
    ) -> None:
        if len(env_ids) == 0:
            return

        env_ids = np.asarray(env_ids, dtype=np.int64)
        eps_steps = episode_steps[env_ids].astype(self._np_dtype)
        eps_steps = np.maximum(eps_steps, 1.0)

        # episode 内平均 |cmd_vx|：只累计 |cmd|>0.05 的步（env 侧口径），
        # 用于把“没有速度指令”的 episode（原地自转 / 站立）排除在课程判定之外。
        episode_cmd_speed_mean = (
            episode_cmd_vel_sum_fwd[env_ids] + episode_cmd_vel_sum_bwd[env_ids]
        ) / eps_steps

        if getattr(self._cfg.commands, "cmd_curriculum_enabled", True):
            self._update_cmd_ranges_per_env(
                env_ids,
                episode_tracking_reward_sum,
                eps_steps,
                episode_cmd_speed_mean,
            )

        self._curriculum_episode_count += len(env_ids)

        height_err_mean = episode_height_err_sum[env_ids] / eps_steps
        height_th = float(getattr(self._cfg.reward_config, "jump_curriculum_height_threshold", 0.03))
        height_ok = height_err_mean < height_th
        self._curriculum_height_good_accum += float(np.sum(height_ok))
        self._curriculum_height_episode_count += len(env_ids)

    def _update_cmd_ranges_per_env(
        self,
        env_ids: np.ndarray,
        episode_tracking_reward_sum: np.ndarray,
        eps_steps: np.ndarray,
        episode_cmd_speed_mean: np.ndarray,
    ) -> None:
        """Promote/demote each resetting env independently.

        The episode tracking sum is already normalized by the tracking reward
        scale (0~1), so the 0.7/0.3 thresholds are reward fractions and must not
        be multiplied by the scale again.

        Only episodes that actually asked for a speed command participate
        (``commands.cmd_curriculum_min_cmd_speed``, 0 = every episode
        participates). Without this gate, rotate-only (vx=0) and standing
        episodes always score tracking≈1, so they promote the range on every
        reset and the curriculum decouples from the achievable speed: the
        range ratchets to the cap while the tracked speed never improves.
        Non-participating episodes leave the range untouched (their good/bad
        flags are False, which also resets the per-env streak counters).
        """
        mean_tracking = episode_tracking_reward_sum[env_ids] / eps_steps
        min_cmd_speed = float(
            getattr(self._cfg.commands, "cmd_curriculum_min_cmd_speed", 0.0) or 0.0
        )
        if min_cmd_speed > 0.0:
            participates = episode_cmd_speed_mean > min_cmd_speed
        else:
            participates = np.ones(env_ids.shape[0], dtype=bool)

        good = participates & (mean_tracking > 0.7)
        bad = participates & (mean_tracking < 0.4)
        # 仅供 debug 打印：记录本次 episode 的判定中间量（不影响任何逻辑）
        self._last_cmd_diag = {
            "mean_tracking": mean_tracking.copy(),
            "participates": participates.copy(),
            "cmd_speed_mean": episode_cmd_speed_mean.copy(),
        }

        self._cmd_consecutive_good[env_ids] = np.where(
            good, self._cmd_consecutive_good[env_ids] + 1, 0
        )
        self._cmd_consecutive_bad[env_ids] = np.where(
            bad, self._cmd_consecutive_bad[env_ids] + 1, 0
        )

        promote = env_ids[
            good & (self._cmd_consecutive_good[env_ids] >= self._curriculum_sustain_threshold)
        ]
        demote = env_ids[
            bad & (self._cmd_consecutive_bad[env_ids] >= self._curriculum_sustain_threshold_bad)
        ]

        if len(promote) > 0:
            self._cmd_consecutive_good[promote] = 0
            self._cmd_apply_promotion(promote)
        if len(demote) > 0:
            self._cmd_consecutive_bad[demote] = 0
            self._cmd_apply_demotion(demote)

    def _cmd_apply_promotion(self, env_ids: np.ndarray) -> None:
        high = self._cmd_vel_range_env[env_ids, 1]
        step = np.maximum(0.03, high * 0.2)
        self._cmd_vel_range_env[env_ids, 0] = np.clip(
            self._cmd_vel_range_env[env_ids, 0] - step,
            -self._cmd_vel_curriculum_max,
            -1.0,
        )
        self._cmd_vel_range_env[env_ids, 1] = np.clip(
            self._cmd_vel_range_env[env_ids, 1] + step,
            1.0,
            self._cmd_vel_curriculum_max,
        )
        yaw_step = 0.6
        self._cmd_yaw_range_env[env_ids, 0] = np.clip(
            self._cmd_yaw_range_env[env_ids, 0] - yaw_step,
            -self._cmd_yaw_curriculum_max,
            -self._init_yaw_range,
        )
        self._cmd_yaw_range_env[env_ids, 1] = np.clip(
            self._cmd_yaw_range_env[env_ids, 1] + yaw_step,
            self._init_yaw_range,
            self._cmd_yaw_curriculum_max,
        )
        self._cmd_promote_count += len(env_ids)

    def _cmd_apply_demotion(self, env_ids: np.ndarray) -> None:
        high = self._cmd_vel_range_env[env_ids, 1]
        step = np.maximum(0.03, high * 0.2)
        self._cmd_vel_range_env[env_ids, 0] = np.clip(
            self._cmd_vel_range_env[env_ids, 0] + step,
            -self._cmd_vel_curriculum_max,
            -1.0,
        )
        self._cmd_vel_range_env[env_ids, 1] = np.clip(
            self._cmd_vel_range_env[env_ids, 1] - step,
            1.0,
            self._cmd_vel_curriculum_max,
        )
        yaw_step = 0.6
        self._cmd_yaw_range_env[env_ids, 0] = np.clip(
            self._cmd_yaw_range_env[env_ids, 0] + yaw_step,
            -self._cmd_yaw_curriculum_max,
            -self._init_yaw_range,
        )
        self._cmd_yaw_range_env[env_ids, 1] = np.clip(
            self._cmd_yaw_range_env[env_ids, 1] - yaw_step,
            self._init_yaw_range,
            self._cmd_yaw_curriculum_max,
        )
        self._cmd_demote_count += len(env_ids)

    def _try_eval(self, step_counter: int) -> None:
        # Command ranges are updated per env on episode reset in on_reset().
        # Keep this hook only for the jump curriculum.
        _ = step_counter
        self._try_eval_jump_curriculum()

    def _try_eval_jump_curriculum(self) -> None:
        if not getattr(self._cfg.commands, "jump_curriculum_enabled", True):
            return
        if self._jump_curriculum_enabled:
            return
        jump_height_count = self._curriculum_height_episode_count
        if jump_height_count > 0:
            height_good_ratio = self._curriculum_height_good_accum / jump_height_count
            if height_good_ratio > 0.7:
                self._jump_curriculum_consecutive_good += 1
            else:
                self._jump_curriculum_consecutive_good = 0
            if self._jump_curriculum_consecutive_good >= self._curriculum_sustain_threshold:
                self._jump_curriculum_enabled = True
        self._curriculum_height_good_accum = 0.0
        self._curriculum_height_episode_count = 0

    def _update_tracking_sigmas(self) -> None:
        self._tracking_sigma_lin = self._tracking_sigma_lin_base
        self._tracking_sigma_ang = self._tracking_sigma_ang_base
        self._tracking_sigma_yaw = self._tracking_sigma_yaw_base