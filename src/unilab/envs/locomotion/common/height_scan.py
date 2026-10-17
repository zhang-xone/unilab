"""Shared height-scan and terrain-bound helpers for rough locomotion envs.

These functions and the ``HeightScanConfig`` dataclass are consumed by Go2,
Go2W, Go1, and G1 rough environments — anywhere the policy / critic ingests
a forward-looking height grid sampled from a procedural heightfield.

Each helper mirrors the original implementation that lived inline in
``go2/rough.py`` so that behavior is bit-for-bit identical; the only change
is the host: env classes import these instead of redefining them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype

DEFAULT_SCAN_POINTS_X: tuple[float, ...] = (
    -0.8,
    -0.7,
    -0.6,
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
)
DEFAULT_SCAN_POINTS_Y: tuple[float, ...] = (
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
)


@dataclass
class HeightScanConfig:
    enabled: bool = True
    hfield_name: str = "terrain_hfield"
    geom_name: str = "floor"
    measured_points_x: list[float] = field(default_factory=lambda: list(DEFAULT_SCAN_POINTS_X))
    measured_points_y: list[float] = field(default_factory=lambda: list(DEFAULT_SCAN_POINTS_Y))
    vertical_offset: float = 0.5
    scale: float = 5.0


def height_scan_offsets(points_x: Sequence[float], points_y: Sequence[float]) -> np.ndarray:
    """Build a contiguous (P, 2) array of (x, y) sampling offsets in body frame."""
    x_grid, y_grid = np.meshgrid(
        np.asarray(points_x, dtype=np.float64),
        np.asarray(points_y, dtype=np.float64),
        indexing="ij",
    )
    offsets = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)
    return np.ascontiguousarray(offsets, dtype=np.float64)


def configured_height_scan_dim(scan_cfg: HeightScanConfig) -> int:
    return len(scan_cfg.measured_points_x) * len(scan_cfg.measured_points_y)


def init_height_scan_sensor(env: Any, scan_cfg: HeightScanConfig, base_body_name: str) -> None:
    """Wire a yaw-aligned heightfield scanner onto ``env``.

    Sets the following attributes on ``env``:
        _height_scan_dim, _height_scan_hfield_geom_id, _height_scan_frame_body_id,
        _height_scan_offsets, _height_scan_sensor.
    """
    env._height_scan_dim = configured_height_scan_dim(scan_cfg)
    if env._height_scan_dim <= 0:
        raise ValueError("terrain_scan measured points must be non-empty")

    env._height_scan_hfield_geom_id = None
    env._height_scan_frame_body_id = None
    env._height_scan_offsets = None
    env._height_scan_sensor = None
    if not scan_cfg.enabled:
        return

    env._height_scan_hfield_geom_id = env._backend.get_geom_id(scan_cfg.geom_name)
    env._height_scan_frame_body_id = env._backend.get_body_id(base_body_name)
    env._height_scan_offsets = height_scan_offsets(
        scan_cfg.measured_points_x,
        scan_cfg.measured_points_y,
    )
    env._height_scan_sensor = env._backend.create_hfield_scanner(
        hfield_geom_id=env._height_scan_hfield_geom_id,
        offsets=env._height_scan_offsets,
        frame_body_id=env._height_scan_frame_body_id,
        alignment="yaw",
        output="height",
    )


def raw_height_scan_obs(env: Any, num_obs: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (raw_heights (N, P), base_pos (N, 3)) or (None, None) if sensor not ready."""
    if (
        env._height_scan_hfield_geom_id is None
        or env._height_scan_frame_body_id is None
        or env._height_scan_offsets is None
        or env._height_scan_sensor is None
    ):
        return None, None

    base_pos = np.asarray(env._backend.get_base_pos(), dtype=get_global_dtype())
    if base_pos.shape[0] != num_obs:
        return None, None

    try:
        raw_heights = env._height_scan_sensor.scan()
    except (ValueError, RuntimeError):
        return None, None
    if raw_heights.shape != (num_obs, env._height_scan_dim):
        return None, None
    return np.asarray(raw_heights, dtype=get_global_dtype()), base_pos


def height_scan_obs(env: Any, scan_cfg: HeightScanConfig, num_obs: int) -> np.ndarray:
    """Clipped, scaled height observation matching the Go2 rough format."""
    raw_heights, base_pos = raw_height_scan_obs(env, num_obs)
    if raw_heights is None or base_pos is None:
        return np.zeros((num_obs, env._height_scan_dim), dtype=get_global_dtype())
    heights = np.clip(base_pos[:, 2:3] - scan_cfg.vertical_offset - raw_heights, -1.0, 1.0)
    return np.asarray(heights * scan_cfg.scale, dtype=get_global_dtype())


def base_height_from_scan(env: Any, num_obs: int | None = None) -> np.ndarray:
    """Estimate base-relative height by averaging the heightfield samples below the body."""
    if num_obs is None:
        num_obs = int(np.asarray(env._backend.get_base_pos()).shape[0])
    raw_heights, base_pos = raw_height_scan_obs(env, num_obs)
    if raw_heights is None or base_pos is None:
        base_pos = np.asarray(env._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())
    return np.asarray(np.mean(base_pos[:, 2:3] - raw_heights, axis=1), dtype=get_global_dtype())


def combined_height_scan_obs(
    env: Any,
    scan_cfg: HeightScanConfig,
    num_obs: int,
    surface_sampler: Any | None = None,
) -> np.ndarray:
    """Height scan that uses a Combined/Hfield surface sampler.

    Unlike ``height_scan_obs`` which reads the MuJoCo hfield sensor, this
    queries the terrain surface sampler directly — it handles box geoms
    (e.g. box-based pyramid stairs) in addition to hfield cells.

    Args:
        env:      MuJoCo environment instance.
        scan_cfg: Height scan configuration (points, clip, scale).
        num_obs:  Number of parallel environments.
        surface_sampler:  A HeightfieldSurfaceSampler or CombinedSurfaceSampler.
                          If None, falls back to ``env._terrain_surface_sampler``.
    Returns:
        A (num_obs, P) float array of clipped & scaled scan heights,
        same format as ``height_scan_obs``.
    """
    if surface_sampler is None:
        surface_sampler = getattr(env, "_terrain_surface_sampler", None)
    if surface_sampler is None:
        return height_scan_obs(env, scan_cfg, num_obs)

    base_pos = np.asarray(env._backend.get_base_pos(), dtype=get_global_dtype())
    base_quat = np.asarray(env._backend.get_base_quat(), dtype=get_global_dtype())

    if base_pos.shape[0] != num_obs or base_quat.shape[0] != num_obs:
        return np.zeros((num_obs, env._height_scan_dim), dtype=get_global_dtype())

    offsets = height_scan_offsets(
        scan_cfg.measured_points_x,
        scan_cfg.measured_points_y,
    )
    P = offsets.shape[0]
    dim = env._height_scan_dim
    if P != dim:
        return np.zeros((num_obs, dim), dtype=get_global_dtype())

    qw = base_quat[:, 0]
    qx = base_quat[:, 1]
    qy = base_quat[:, 2]
    qz = base_quat[:, 3]
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    sin_yaw = 2.0 * (qw * qz + qx * qy)

    ox = offsets[:, 0]  # (P,)
    oy = offsets[:, 1]  # (P,)
    cos_yaw_2d = cos_yaw[:, None]  # (num_obs, 1)
    sin_yaw_2d = sin_yaw[:, None]  # (num_obs, 1)
    world_x = base_pos[:, 0:1] + cos_yaw_2d * ox[None, :] - sin_yaw_2d * oy[None, :]
    world_y = base_pos[:, 1:2] + sin_yaw_2d * ox[None, :] + cos_yaw_2d * oy[None, :]
    world_xy_flat = np.column_stack([world_x.ravel(), world_y.ravel()])
    heights_flat = surface_sampler.sample_height(world_xy_flat)
    raw_heights = heights_flat.reshape(num_obs, P)

    heights = np.clip(
        base_pos[:, 2:3] - scan_cfg.vertical_offset - raw_heights,
        -1.0, 1.0,
    )
    return np.asarray(heights * scan_cfg.scale, dtype=get_global_dtype())


def combined_base_height_from_scan(env: Any, scan_cfg: HeightScanConfig | None = None) -> np.ndarray:
    """Estimate base-relative height using the surface sampler.

    Same as ``base_height_from_scan`` but queries the terrain surface sampler
    (supports box geoms) instead of the MuJoCo hfield sensor.
    """
    num_obs = int(np.asarray(env._backend.get_base_pos()).shape[0])
    if scan_cfg is None:
        scan_cfg = getattr(env._cfg, "terrain_scan", None)
    base_pos = np.asarray(env._backend.get_base_pos(), dtype=get_global_dtype())
    if base_pos.shape[0] != num_obs:
        return np.zeros((num_obs,), dtype=get_global_dtype())

    sampler = getattr(env, "_terrain_surface_sampler", None)
    if sampler is None:
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())

    heights = sampler.sample_height(base_pos[:, :2])
    return np.asarray(base_pos[:, 2] - heights, dtype=get_global_dtype())


def terrain_num_cols(terrain_cfg: Any) -> int:
    if terrain_cfg.curriculum:
        return len(terrain_cfg.sub_terrains)
    return int(terrain_cfg.num_cols)


def terrain_out_of_bounds(env: Any, terrain_cfg: Any, distance_buffer: float) -> np.ndarray:
    """Boolean mask: True when the body's (x, y) is outside the terrain footprint."""
    if terrain_cfg is None:
        return np.zeros((env._num_envs,), dtype=bool)

    size_x, size_y = terrain_cfg.size
    num_cols = terrain_num_cols(terrain_cfg)
    map_width = terrain_cfg.num_rows * float(size_x) + 2.0 * float(terrain_cfg.border_width)
    map_height = num_cols * float(size_y) + 2.0 * float(terrain_cfg.border_width)
    base_pos = np.asarray(env._backend.get_base_pos(), dtype=get_global_dtype())
    if base_pos.shape[0] != env._num_envs:
        return np.zeros((env._num_envs,), dtype=bool)
    x_out = np.abs(base_pos[:, 0]) > 0.5 * map_width - distance_buffer
    y_out = np.abs(base_pos[:, 1]) > 0.5 * map_height - distance_buffer
    return np.asarray(x_out | y_out, dtype=bool)


# ---------------------------------------------------------------------------
# General ray-cast surface sampler
# ---------------------------------------------------------------------------
class MjRaySurfaceSampler:
    """General surface sampler using ``mujoco.mj_ray`` ray casting.

    Casts vertical rays downward from each query (x, y) against **all** static
    geoms (hfield, mesh, box, plane, cylinder, ...).  Compatible with
    ``combined_base_height_from_scan`` and ``combined_height_scan_obs``.

    A spatial grid cache (default 2 cm resolution) avoids re-casting rays
    for nearby points that land on the same grid cell.
    """

    def __init__(
        self,
        model: Any,
        *,
        grid_resolution: float = 0.02,
        ray_height: float = 50.0,
    ) -> None:
        import mujoco as _mj

        if not isinstance(model, _mj.MjModel):
            raise TypeError(f"model must be a mujoco.MjModel, got {type(model)}")
        self._model = model
        self._data = _mj.MjData(model)
        _mj.mj_forward(model, self._data)
        self._ray_start = np.array([[0.0], [0.0], [float(ray_height)]], dtype=np.float64)
        self._ray_dir = np.array([[0.0], [0.0], [-1.0]], dtype=np.float64)
        self._geomid = np.array([[-1]], dtype=np.int32)
        self._grid_res = float(grid_resolution)
        self._cache: dict[tuple[int, int], float] = {}

    def sample_height(self, xy: np.ndarray) -> np.ndarray:
        import mujoco as _mj

        points = np.asarray(xy, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 2:
            raise ValueError(f"xy must have shape (..., 2), got {points.shape}")
        flat = points.reshape(-1, 2)
        h = np.zeros(flat.shape[0], dtype=np.float64)
        inv_res = 1.0 / self._grid_res

        for i in range(flat.shape[0]):
            fx, fy = flat[i, 0], flat[i, 1]
            key = (int(round(fx * inv_res)), int(round(fy * inv_res)))
            cached = self._cache.get(key)
            if cached is not None:
                h[i] = cached
                continue
            self._ray_start[0, 0] = fx
            self._ray_start[1, 0] = fy
            dist = _mj.mj_ray(
                self._model, self._data,
                self._ray_start, self._ray_dir,
                None, 1, -1,
                self._geomid,
            )
            height = float(self._ray_start[2, 0] - dist) if dist >= 0 else 0.0
            self._cache[key] = height
            h[i] = height

        return h.reshape(points.shape[:-1])


def create_mjray_surface_sampler(env: Any, grid_resolution: float = 0.02) -> MjRaySurfaceSampler | None:
    """Convenience factory that builds a ``MjRaySurfaceSampler`` from an env's backend model.

    Returns ``None`` if the backend model is not accessible.
    """
    try:
        model = env._backend._model
        return MjRaySurfaceSampler(model, grid_resolution=grid_resolution)
    except Exception:
        return None