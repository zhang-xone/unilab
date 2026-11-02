"""Spawn-origin managers for locomotion envs.

``BaseSpawnManager`` is a no-op default: every env spawns at the world origin
(plus the existing per-env xy jitter from the dr_provider). Used whenever the
env has no procedural terrain — flat scenes don't need spatial separation

``TerrainSpawnManager`` overrides this for terrain scenes: it indexes
``terrain_origins[level, type_col]`` so each env spawns on a specific cell, and
optionally promotes/demotes ``level`` per-env on episode end. With
``enabled=True`` levels start at 0; with ``enabled=False`` levels are uniformly
distributed and never change — but spawn still uses cell-aware xyz so robots
land on the correct surface height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BaseSpawnManager:
    """Default no-op spawn manager: returns zeros, records nothing."""

    def __init__(self, z_offset: float = 0.0) -> None:
        # 在 keyframe 高度统一抬高的 z（m）。0 表示完全按 keyframe。
        self.z_offset = float(z_offset)

    def origins_for(self, env_ids: np.ndarray) -> np.ndarray:
        return np.zeros((env_ids.shape[0], 3), dtype=np.float64)

    def apply_spawn(
        self,
        env_ids: np.ndarray,
        qpos_xyz: np.ndarray,
        *,
        yaw: np.ndarray | None = None,
    ) -> np.ndarray:
        del yaw
        out = np.asarray(qpos_xyz, dtype=np.float64) + self.origins_for(env_ids)
        if self.z_offset:
            out = out.copy()
            out[:, 2] += self.z_offset
        return out

    def record_episode_start(self, env_ids: np.ndarray, qpos_xyz: np.ndarray) -> None:
        del env_ids, qpos_xyz

    def update_on_done(
        self,
        done_indices: np.ndarray,
        current_xyz: np.ndarray,
        *,
        episode_tracking_reward_sum: np.ndarray | None = None,
        episode_steps: np.ndarray | None = None,
    ) -> dict[str, float]:
        del done_indices, current_xyz, episode_tracking_reward_sum, episode_steps
        return {}

    def set_iteration(self, iteration: int) -> None:
        """No-op: terrain type curriculum is only active in :class:`TerrainSpawnManager`."""
        del iteration


@dataclass
class TerrainCurriculumCfg:
    enabled: bool = False
    """If True, levels start at 0 and evolve via promote/demote."""
    init_level: int = 0
    """Initial terrain level. 恢复训练时设为上次 mean_level，避免从 0 重新爬。"""
    randomize_init_level: bool = False
    """If True, each env gets a random level in [0, init_level]."""
    promote_frac: float = 0.5
    """Walked distance > promote_frac * cell_size promotes one level."""
    demote_frac: float = 0.25
    """Walked distance < demote_frac * cell_size demotes one level."""
    cycle_top_frac: float = 0.5
    """When level overflows the top row, resample uniformly in
    ``[num_rows * cycle_top_frac, num_rows - 1]``."""
    spawn_height_margin: float = 0.05
    """Extra z added on top of the sampled terrain surface height."""
    seed: int | None = None
    curriculum_mode: str = "distance"
    """Promote/demote strategy: "distance" | "tracking" | "hybrid".
    
    - ``distance``: promote/demote based on walked distance (promote_frac / demote_frac).
    - ``tracking``: promote/demote based on velocity tracking reward.
    - ``hybrid``: promote by distance, demote by tracking reward (Fudan style).
    """
    tracking_reward_threshold: float = 0.7
    """Mean tracking reward > this threshold promotes one level."""
    tracking_reward_demote_threshold: float = 0.3
    """Mean tracking reward < this threshold demotes one level."""
    type_curriculum_enabled: bool = False
    """If True, terrain types advance in registration order every ``type_curriculum_switch_interval``
    iterations. After all types are exhausted, ``type_cols`` is randomized per env."""
    type_curriculum_switch_interval: int = 200
    """Number of training iterations between terrain-type switches."""
    type_curriculum_iteration_offset: int = 0
    """Offset subtracted from training iteration before computing type index.
    Set to ``-current_iter`` on resume to restart type sequence from 0,
    or leave at 0 to continue from the current iteration's type."""


class TerrainSpawnManager(BaseSpawnManager):
    def __init__(
        self,
        num_envs: int,
        terrain_origins: np.ndarray,
        cell_size: float,
        cfg: TerrainCurriculumCfg,
        terrain_surface_sampler: object | None = None,
        spawn_height_points: np.ndarray | None = None,
        terrain_type_names: list[str] | None = None,
    ) -> None:
        if terrain_origins.ndim != 3 or terrain_origins.shape[2] != 3:
            raise ValueError(
                f"terrain_origins must have shape (num_rows, num_cols, 3); "
                f"got {terrain_origins.shape}"
            )
        num_rows, num_cols, _ = terrain_origins.shape
        if cfg.enabled and num_rows < 2:
            raise ValueError(
                f"Curriculum requires terrain_generator.num_rows >= 2; got {num_rows}."
            )

        self._terrain_origins = terrain_origins.astype(np.float64, copy=False)
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._cell_size = float(cell_size)
        self._cfg = cfg
        self._terrain_surface_sampler = terrain_surface_sampler
        if terrain_type_names is not None:
            self._terrain_type_names = list(terrain_type_names)
        else:
            self._terrain_type_names = [str(i) for i in range(num_cols)]
        if spawn_height_points is None:
            self._spawn_height_points = np.zeros((1, 3), dtype=np.float64)
        else:
            points = np.asarray(spawn_height_points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    f"spawn_height_points must have shape (num_points, 3), got {points.shape}"
                )
            self._spawn_height_points = points
        self._rng = np.random.default_rng(cfg.seed)

        self.type_cols = self._rng.integers(0, num_cols, size=num_envs).astype(np.int32)
        if cfg.enabled:
            init_lvl = max(0, min(int(cfg.init_level), num_rows - 1))
            if cfg.randomize_init_level:
                self.levels = self._rng.integers(0, init_lvl + 1, size=num_envs).astype(np.int32)
            else:
                self.levels = np.full(num_envs, init_lvl, dtype=np.int32)
        else:
            self.levels = self._rng.integers(0, num_rows, size=num_envs).astype(np.int32)

        self._episode_start_xyz = np.zeros((num_envs, 3), dtype=np.float64)
        self._has_started = np.zeros(num_envs, dtype=bool)

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def origins_for(self, env_ids: np.ndarray) -> np.ndarray:
        rows = self.levels[env_ids]
        cols = self.type_cols[env_ids]
        out = self._terrain_origins[rows, cols].copy()
        out[:, 2] += self._cfg.spawn_height_margin
        return out

    def get_terrain_type_for(self, env_ids: np.ndarray) -> list[str]:
        cols = self.type_cols[env_ids]
        return [self._terrain_type_names[c] for c in cols]

    def get_terrain_difficulty_for(self, env_ids: np.ndarray) -> np.ndarray:
        rows = self.levels[env_ids]
        return rows.astype(np.float64) / max(self._num_rows - 1, 1)

    def apply_spawn(
        self,
        env_ids: np.ndarray,
        qpos_xyz: np.ndarray,
        *,
        yaw: np.ndarray | None = None,
    ) -> np.ndarray:
        rows = self.levels[env_ids]
        cols = self.type_cols[env_ids]
        origins = self._terrain_origins[rows, cols]
        out = np.asarray(qpos_xyz, dtype=np.float64).copy()
        base_height = out[:, 2].copy()
        out[:, 0:2] += origins[:, 0:2]
        if self._terrain_surface_sampler is None:
            out[:, 2] += origins[:, 2] + self._cfg.spawn_height_margin
            return out

        required_base_z = self._sample_spawn_required_base_height(
            out[:, 0:2],
            base_height=base_height,
            yaw=yaw,
        )
        out[:, 2] = required_base_z + self._cfg.spawn_height_margin
        return out

    def _sample_spawn_required_base_height(
        self,
        base_xy: np.ndarray,
        *,
        base_height: np.ndarray,
        yaw: np.ndarray | None,
    ) -> np.ndarray:
        sampler = self._terrain_surface_sampler
        sample_height = getattr(sampler, "sample_height", None)
        if not callable(sample_height):
            raise TypeError("terrain_surface_sampler must expose sample_height(xy)")

        points = self._spawn_height_points
        local_xy = points[:, :2]
        local_z = points[:, 2]
        base_surface = np.asarray(sample_height(base_xy), dtype=np.float64)
        required_base_z = base_surface + np.asarray(base_height, dtype=np.float64)
        if yaw is None:
            rotated_xy = np.broadcast_to(local_xy, (base_xy.shape[0],) + local_xy.shape)
        else:
            yaw_arr = np.asarray(yaw, dtype=np.float64).reshape(-1)
            if yaw_arr.shape != (base_xy.shape[0],):
                raise ValueError(f"yaw must have shape ({base_xy.shape[0]},), got {yaw_arr.shape}")
            cos_yaw = np.cos(yaw_arr)
            sin_yaw = np.sin(yaw_arr)
            rotated_xy = np.empty((base_xy.shape[0], local_xy.shape[0], 2), dtype=np.float64)
            rotated_xy[:, :, 0] = (
                cos_yaw[:, None] * local_xy[None, :, 0] - sin_yaw[:, None] * local_xy[None, :, 1]
            )
            rotated_xy[:, :, 1] = (
                sin_yaw[:, None] * local_xy[None, :, 0] + cos_yaw[:, None] * local_xy[None, :, 1]
            )

        sample_xy = base_xy[:, None, :] + rotated_xy
        sampled = np.asarray(sample_height(sample_xy.reshape(-1, 2)), dtype=np.float64).reshape(
            base_xy.shape[0], points.shape[0]
        )
        required_support_z = np.max(sampled - local_z[None, :], axis=1)
        return np.maximum(required_base_z, required_support_z)

    def record_episode_start(self, env_ids: np.ndarray, qpos_xyz: np.ndarray) -> None:
        self._episode_start_xyz[env_ids] = qpos_xyz
        self._has_started[env_ids] = True

    def update_on_done(
        self,
        done_indices: np.ndarray,
        current_xyz: np.ndarray,
        *,
        episode_tracking_reward_sum: np.ndarray | None = None,
        episode_steps: np.ndarray | None = None,
    ) -> dict[str, float]:
        active_mask = self._has_started[done_indices]
        active = done_indices[active_mask]
        num_skipped = int((~active_mask).sum())

        if active.size == 0:
            return {
                "mean_level": float(self.levels.mean()),
                "max_level": float(self.levels.max()),
                "mean_walked": 0.0,
                "num_promoted": 0,
                "num_demoted": 0,
                "num_skipped": num_skipped,
            }

        starts = self._episode_start_xyz[active, :2]
        ends = current_xyz[active_mask, :2]
        walked = np.linalg.norm(ends - starts, axis=1)

        num_promoted = 0
        num_demoted = 0
        if self._cfg.enabled:
            if self._cfg.curriculum_mode == "hybrid":
                promote_threshold = self._cfg.promote_frac * self._cell_size
                promote_mask = walked > promote_threshold

                if episode_tracking_reward_sum is not None and episode_steps is not None:
                    eps_steps = episode_steps[done_indices][active_mask].astype(np.float64)
                    eps_steps = np.maximum(eps_steps, 1.0)
                    tracking_reward_mean = episode_tracking_reward_sum[done_indices][active_mask] / eps_steps
                    demote_mask = (tracking_reward_mean < self._cfg.tracking_reward_demote_threshold) & (~promote_mask)
                else:
                    demote_mask = np.zeros(active.size, dtype=bool)
            elif self._cfg.curriculum_mode == "tracking" and episode_tracking_reward_sum is not None and episode_steps is not None:
                eps_steps = episode_steps[done_indices][active_mask].astype(np.float64)
                eps_steps = np.maximum(eps_steps, 1.0)
                tracking_reward_mean = episode_tracking_reward_sum[done_indices][active_mask] / eps_steps

                promote_mask = tracking_reward_mean > self._cfg.tracking_reward_threshold
                demote_mask = tracking_reward_mean < self._cfg.tracking_reward_demote_threshold
            else:
                promote_threshold = self._cfg.promote_frac * self._cell_size
                demote_threshold = self._cfg.demote_frac * self._cell_size
                promote_mask = walked > promote_threshold
                demote_mask = walked < demote_threshold

            promote_ids = active[promote_mask]
            demote_ids = active[demote_mask]
            num_promoted = int(promote_ids.size)
            num_demoted = int(demote_ids.size)

            self.levels[promote_ids] += 1
            self.levels[demote_ids] -= 1

            overflow_mask = self.levels[promote_ids] >= self._num_rows
            if overflow_mask.any():
                lo = int(self._num_rows * self._cfg.cycle_top_frac)
                lo = min(max(lo, 0), self._num_rows - 1)
                overflow_ids = promote_ids[overflow_mask]
                self.levels[overflow_ids] = self._rng.integers(
                    lo, self._num_rows, size=overflow_ids.size
                ).astype(np.int32)

            np.clip(self.levels, 0, self._num_rows - 1, out=self.levels)

        return {
            "mean_level": float(self.levels.mean()),
            "max_level": float(self.levels.max()),
            "mean_walked": float(walked.mean()),
            "num_promoted": num_promoted,
            "num_demoted": num_demoted,
            "num_skipped": num_skipped,
        }

    def set_iteration(self, iteration: int) -> None:
        if not self._cfg.type_curriculum_enabled:
            return
        interval = max(1, self._cfg.type_curriculum_switch_interval)
        effective_iter = max(0, iteration + self._cfg.type_curriculum_iteration_offset)
        type_idx = effective_iter // interval
        if type_idx >= self._num_cols:
            self.type_cols = self._rng.integers(0, self._num_cols, size=len(self.type_cols)).astype(np.int32)
        else:
            self.type_cols[:] = type_idx