"""Terrains composed of heightfields.

This module provides terrain generation functionality using heightfields,
adapted from the IsaacLab terrain generation system.

References:
    IsaacLab mesh terrain implementation:
    https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/height_field/hf_terrains.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from unilab.terrains.terrain_generator import (
    BoxGeomSpec,
    CylinderGeomSpec,
    SubTerrainCfg,
    TerrainHeightField,
    TerrainOutput,
)
from unilab.terrains.utils import (
    bilinear_resample_grid,
    find_flat_patches_from_heightfield,
)

_MIN_BASE_THICKNESS = 0.01


def _build_heightfield(
    noise: np.ndarray,
    *,
    size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
    base_thickness_ratio: float,
    z_offset_fn: Callable[[float], float] = lambda max_h: 0.0,
) -> TerrainHeightField:
    elevation_min = int(np.min(noise))
    elevation_max = int(np.max(noise))
    elevation_range = elevation_max - elevation_min if elevation_max != elevation_min else 1
    max_physical_height = elevation_range * vertical_scale
    base_thickness = max(max_physical_height * base_thickness_ratio, _MIN_BASE_THICKNESS)
    z_offset = z_offset_fn(max_physical_height)
    return TerrainHeightField(
        noise=noise.copy(),
        size=size,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        max_physical_height=max_physical_height,
        base_thickness=base_thickness,
        z_offset=z_offset,
    )


def _compute_flat_patches(
    noise: np.ndarray,
    vertical_scale: float,
    horizontal_scale: float,
    z_offset: float,
    flat_patch_sampling: dict | None,
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    """Compute flat patches for a heightfield terrain if configured."""
    if flat_patch_sampling is None:
        return None
    physical_heights = (noise.astype(np.float64) - noise.min()) * vertical_scale
    flat_patches: dict[str, np.ndarray] = {}
    for name, patch_cfg in flat_patch_sampling.items():
        flat_patches[name] = find_flat_patches_from_heightfield(
            heights=physical_heights,
            horizontal_scale=horizontal_scale,
            z_offset=z_offset,
            cfg=patch_cfg,
            rng=rng,
        )
    return flat_patches


def _make_terrain_output(
    noise: np.ndarray,
    *,
    size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
    base_thickness_ratio: float,
    origin: np.ndarray,
    flat_patch_sampling: dict | None,
    rng: np.random.Generator,
    z_offset_fn: Callable[[float], float] = lambda max_h: 0.0,
    box_geoms: list[BoxGeomSpec] | None = None,
    cylinder_geoms: list[CylinderGeomSpec] | None = None,
) -> TerrainOutput:
    terrain_hfield = _build_heightfield(
        noise,
        size=size,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        base_thickness_ratio=base_thickness_ratio,
        z_offset_fn=z_offset_fn,
    )

    flat_patches = _compute_flat_patches(
        noise,
        vertical_scale,
        horizontal_scale,
        terrain_hfield.z_offset,
        flat_patch_sampling,
        rng,
    )
    return TerrainOutput(
        origin=origin,
        heightfield=terrain_hfield,
        flat_patches=flat_patches,
        box_geoms=box_geoms,
        cylinder_geoms=cylinder_geoms,
    )


@dataclass(kw_only=True)
class HfPyramidSlopedTerrainCfg(SubTerrainCfg):
    slope_range: tuple[float, float]
    """Range of slope gradients (rise / run), interpolated by difficulty."""
    platform_width: float = 1.0
    """Side length of the flat square platform at the terrain center, in meters."""
    inverted: bool = False
    """If True, the pyramid is inverted so the platform is at the bottom."""
    border_width: float = 0.0
    """Width of the flat border around the terrain edges, in meters. Must be >=
    horizontal_scale if non-zero."""
    horizontal_scale: float = 0.1
    """Heightfield grid resolution along x and y, in meters per cell."""
    vertical_scale: float = 0.005
    """Heightfield height resolution, in meters per integer unit of the noise array."""
    base_thickness_ratio: float = 1.0
    """Ratio of the heightfield base thickness to its maximum surface height."""

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        if self.inverted:
            slope = -self.slope_range[0] - difficulty * (self.slope_range[1] - self.slope_range[0])
        else:
            slope = self.slope_range[0] + difficulty * (self.slope_range[1] - self.slope_range[0])

        if self.border_width > 0 and self.border_width < self.horizontal_scale:
            raise ValueError(
                f"Border width ({self.border_width}) must be >= horizontal scale "
                f"({self.horizontal_scale})"
            )

        border_pixels = int(self.border_width / self.horizontal_scale)
        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)

        inner_width_pixels = width_pixels - 2 * border_pixels
        inner_length_pixels = length_pixels - 2 * border_pixels

        noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        if border_pixels > 0:
            height_max = int(
                slope * (inner_width_pixels * self.horizontal_scale) / 2 / self.vertical_scale
            )

            center_x = int(inner_width_pixels / 2)
            center_y = int(inner_length_pixels / 2)

            x = np.arange(0, inner_width_pixels)
            y = np.arange(0, inner_length_pixels)
            xx, yy = np.meshgrid(x, y, sparse=True)

            xx = (center_x - np.abs(center_x - xx)) / center_x
            yy = (center_y - np.abs(center_y - yy)) / center_y

            xx = xx.reshape(inner_width_pixels, 1)
            yy = yy.reshape(1, inner_length_pixels)

            hf_raw = height_max * xx * yy

            platform_width = int(self.platform_width / self.horizontal_scale / 2)
            x_pf = inner_width_pixels // 2 - platform_width
            y_pf = inner_length_pixels // 2 - platform_width
            z_pf = hf_raw[x_pf, y_pf] if x_pf >= 0 and y_pf >= 0 else 0
            hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

            noise[
                border_pixels : -border_pixels if border_pixels else width_pixels,
                border_pixels : -border_pixels if border_pixels else length_pixels,
            ] = np.rint(hf_raw).astype(np.int16)
        else:
            height_max = int(slope * self.size[0] / 2 / self.vertical_scale)

            center_x = int(width_pixels / 2)
            center_y = int(length_pixels / 2)

            x = np.arange(0, width_pixels)
            y = np.arange(0, length_pixels)
            xx, yy = np.meshgrid(x, y, sparse=True)

            xx = (center_x - np.abs(center_x - xx)) / center_x
            yy = (center_y - np.abs(center_y - yy)) / center_y

            xx = xx.reshape(width_pixels, 1)
            yy = yy.reshape(1, length_pixels)

            hf_raw = height_max * xx * yy

            platform_width = int(self.platform_width / self.horizontal_scale / 2)
            x_pf = width_pixels // 2 - platform_width
            y_pf = length_pixels // 2 - platform_width
            z_pf = hf_raw[x_pf, y_pf]
            hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

            noise = np.rint(hf_raw).astype(np.int16)

        z_offset_fn = (lambda max_h: -max_h) if self.inverted else (lambda max_h: 0.0)
        terrain_hfield = _build_heightfield(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            z_offset_fn=z_offset_fn,
        )

        spawn_height = (
            terrain_hfield.z_offset if self.inverted else terrain_hfield.max_physical_height
        )
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
            z_offset_fn=z_offset_fn,
        )


@dataclass(kw_only=True)
class HfRandomUniformTerrainCfg(SubTerrainCfg):
    noise_range: tuple[float, float]
    """Min and max height noise, in meters."""
    noise_step: float = 0.005
    """Height quantization step, in meters. Sampled heights are multiples of this
    value within noise_range."""
    downsampled_scale: float | None = None
    """Spacing between randomly sampled height points before interpolation, in
    meters. If None, uses horizontal_scale. Must be >= horizontal_scale."""
    horizontal_scale: float = 0.1
    """Heightfield grid resolution along x and y, in meters per cell."""
    vertical_scale: float = 0.005
    """Heightfield height resolution, in meters per integer unit of the noise array."""
    base_thickness_ratio: float = 1.0
    """Ratio of the heightfield base thickness to its maximum surface height."""
    border_width: float = 0.0
    """Width of the flat border around the terrain edges, in meters. Must be >=
    horizontal_scale if non-zero."""

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        del difficulty  # Unused.

        if self.border_width > 0 and self.border_width < self.horizontal_scale:
            raise ValueError(
                f"Border width ({self.border_width}) must be >= horizontal scale "
                f"({self.horizontal_scale})"
            )

        if self.downsampled_scale is None:
            downsampled_scale = self.horizontal_scale
        elif self.downsampled_scale < self.horizontal_scale:
            raise ValueError(
                f"Downsampled scale must be >= horizontal scale: "
                f"{self.downsampled_scale} < {self.horizontal_scale}"
            )
        else:
            downsampled_scale = self.downsampled_scale

        border_pixels = int(self.border_width / self.horizontal_scale)
        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)

        noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        if border_pixels > 0:
            inner_width_pixels = width_pixels - 2 * border_pixels
            inner_length_pixels = length_pixels - 2 * border_pixels
            inner_size = (
                inner_width_pixels * self.horizontal_scale,
                inner_length_pixels * self.horizontal_scale,
            )

            width_downsampled = int(inner_size[0] / downsampled_scale)
            length_downsampled = int(inner_size[1] / downsampled_scale)

            height_min = int(self.noise_range[0] / self.vertical_scale)
            height_max = int(self.noise_range[1] / self.vertical_scale)
            height_step = int(self.noise_step / self.vertical_scale)

            height_range = np.arange(height_min, height_max + height_step, height_step)
            height_field_downsampled = rng.choice(
                height_range, size=(width_downsampled, length_downsampled)
            )

            x = np.linspace(0, inner_size[0], width_downsampled)
            y = np.linspace(0, inner_size[1], length_downsampled)

            x_upsampled = np.linspace(0, inner_size[0], inner_width_pixels)
            y_upsampled = np.linspace(0, inner_size[1], inner_length_pixels)
            z_upsampled = bilinear_resample_grid(
                x, y, height_field_downsampled, x_upsampled, y_upsampled
            )

            noise[
                border_pixels : -border_pixels if border_pixels else width_pixels,
                border_pixels : -border_pixels if border_pixels else length_pixels,
            ] = np.rint(z_upsampled).astype(np.int16)
        else:
            width_downsampled = int(self.size[0] / downsampled_scale)
            length_downsampled = int(self.size[1] / downsampled_scale)
            height_min = int(self.noise_range[0] / self.vertical_scale)
            height_max = int(self.noise_range[1] / self.vertical_scale)
            height_step = int(self.noise_step / self.vertical_scale)

            height_range = np.arange(height_min, height_max + height_step, height_step)
            height_field_downsampled = rng.choice(
                height_range, size=(width_downsampled, length_downsampled)
            )

            x = np.linspace(0, self.size[0], width_downsampled)
            y = np.linspace(0, self.size[1], length_downsampled)

            x_upsampled = np.linspace(0, self.size[0], width_pixels)
            y_upsampled = np.linspace(0, self.size[1], length_pixels)
            z_upsampled = bilinear_resample_grid(
                x, y, height_field_downsampled, x_upsampled, y_upsampled
            )
            noise = np.rint(z_upsampled).astype(np.int16)

        spawn_height = (self.noise_range[0] + self.noise_range[1]) / 2
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
        )


@dataclass(kw_only=True)
class HfWaveTerrainCfg(SubTerrainCfg):
    amplitude_range: tuple[float, float]
    """Min and max wave amplitude, in meters. Interpolated by difficulty."""
    num_waves: int = 1
    """Number of complete wave cycles along the terrain length."""
    horizontal_scale: float = 0.1
    """Heightfield grid resolution along x and y, in meters per cell."""
    vertical_scale: float = 0.005
    """Heightfield height resolution, in meters per integer unit of the noise array."""
    base_thickness_ratio: float = 0.25
    """Ratio of the heightfield base thickness to its maximum surface height."""
    border_width: float = 0.0
    """Width of the flat border around the terrain edges, in meters. Must be >=
    horizontal_scale if non-zero."""

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        if self.num_waves <= 0:
            raise ValueError(f"Number of waves must be positive. Got: {self.num_waves}")

        if self.border_width > 0 and self.border_width < self.horizontal_scale:
            raise ValueError(
                f"Border width ({self.border_width}) must be >= horizontal scale "
                f"({self.horizontal_scale})"
            )

        amplitude = self.amplitude_range[0] + difficulty * (
            self.amplitude_range[1] - self.amplitude_range[0]
        )

        border_pixels = int(self.border_width / self.horizontal_scale)
        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)

        noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        if border_pixels > 0:
            inner_width_pixels = width_pixels - 2 * border_pixels
            inner_length_pixels = length_pixels - 2 * border_pixels

            amplitude_pixels = int(0.5 * amplitude / self.vertical_scale)
            wave_length = inner_length_pixels / self.num_waves
            wave_number = 2 * np.pi / wave_length

            x = np.arange(0, inner_width_pixels)
            y = np.arange(0, inner_length_pixels)
            xx, yy = np.meshgrid(x, y, sparse=True)
            xx = xx.reshape(inner_width_pixels, 1)
            yy = yy.reshape(1, inner_length_pixels)

            hf_raw = amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))

            noise[
                border_pixels : -border_pixels if border_pixels else width_pixels,
                border_pixels : -border_pixels if border_pixels else length_pixels,
            ] = np.rint(hf_raw).astype(np.int16)
        else:
            amplitude_pixels = int(0.5 * amplitude / self.vertical_scale)
            wave_length = length_pixels / self.num_waves
            wave_number = 2 * np.pi / wave_length

            x = np.arange(0, width_pixels)
            y = np.arange(0, length_pixels)
            xx, yy = np.meshgrid(x, y, sparse=True)
            xx = xx.reshape(width_pixels, 1)
            yy = yy.reshape(1, length_pixels)

            hf_raw = amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))
            noise = np.rint(hf_raw).astype(np.int16)

        spawn_height = 0.0
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
            z_offset_fn=lambda max_h: -max_h / 2,
        )


@dataclass(kw_only=True)
class HfPyramidStairsTerrainCfg(SubTerrainCfg):
    """A pyramid stairs terrain encoded as a heightfield.

    Concentric square rings from the outside in form a staircase climbing toward
    a central platform. With ``holes=True`` the four diagonal corners of each
    ring are carved out to a deep pit; agents falling into the pit reach a
    terminating depth instead of an infinite void.
    """

    step_height_range: tuple[float, float]
    """Min and max step height, in meters. Interpolated by difficulty."""
    step_width: float
    """Depth (run) of each step, in meters. Must be a multiple of horizontal_scale."""
    platform_width: float = 1.0
    """Side length of the flat square platform at the top of the staircase, in meters."""
    border_width: float = 0.0
    """Width of the flat outer border around the staircase, in meters."""
    holes: bool = False
    """If True, carve deep pits at the diagonal corners of each step ring."""
    pit_depth: float = 5.0
    """Depth of holes-mode pits below the lowest stair, in meters."""
    horizontal_scale: float = 0.05
    """Heightfield grid resolution. Overwritten by TerrainGenerator."""
    vertical_scale: float = 0.005
    """Heightfield height resolution. Overwritten by TerrainGenerator."""
    base_thickness_ratio: float = 1.0
    """Ratio of the heightfield base thickness to its surface height."""

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )

        W = int(round(self.size[0] / self.horizontal_scale))
        L = int(round(self.size[1] / self.horizontal_scale))
        step_px = int(round(self.step_width / self.horizontal_scale))
        plat_px = int(round(self.platform_width / self.horizontal_scale))
        border_px = int(round(self.border_width / self.horizontal_scale))
        step_units = int(round(step_height / self.vertical_scale))

        noise = np.zeros((W, L), dtype=np.int16)

        inner = min(W, L) - 2 * border_px
        n_steps = max(0, (inner - plat_px) // (2 * step_px)) if step_px > 0 else 0

        for k in range(n_steps):
            lo_x = border_px + k * step_px
            hi_x = W - border_px - k * step_px
            lo_y = border_px + k * step_px
            hi_y = L - border_px - k * step_px
            noise[lo_x:hi_x, lo_y:hi_y] = (k + 1) * step_units

        plat_lo_x = border_px + n_steps * step_px
        plat_hi_x = W - border_px - n_steps * step_px
        plat_lo_y = border_px + n_steps * step_px
        plat_hi_y = L - border_px - n_steps * step_px
        noise[plat_lo_x:plat_hi_x, plat_lo_y:plat_hi_y] = (n_steps + 1) * step_units

        if self.holes:
            pit_units = -int(round(self.pit_depth / self.vertical_scale))
            if pit_units < np.iinfo(np.int16).min:
                raise ValueError(
                    f"pit_depth={self.pit_depth} m at vertical_scale="
                    f"{self.vertical_scale} m overflows int16 range."
                )
            for k in range(n_steps):
                lo_x = border_px + k * step_px
                hi_x = W - border_px - k * step_px
                lo_y = border_px + k * step_px
                hi_y = L - border_px - k * step_px
                # Four outer corner squares of this ring.
                noise[lo_x : lo_x + step_px, lo_y : lo_y + step_px] = pit_units
                noise[lo_x : lo_x + step_px, hi_y - step_px : hi_y] = pit_units
                noise[hi_x - step_px : hi_x, lo_y : lo_y + step_px] = pit_units
                noise[hi_x - step_px : hi_x, hi_y - step_px : hi_y] = pit_units

        spawn_z = (n_steps + 1) * step_units * self.vertical_scale
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_z])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
        )


@dataclass(kw_only=True)
class HfInvertedPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg):
    """A pit-style pyramid stairs terrain encoded as a heightfield.

    Inverts :class:`HfPyramidStairsTerrainCfg`: outer ring sits at world z=0,
    rings descend toward a central platform at the bottom. With ``holes=True``
    the diagonal corners are even deeper than the platform.
    """

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )

        W = int(round(self.size[0] / self.horizontal_scale))
        L = int(round(self.size[1] / self.horizontal_scale))
        step_px = int(round(self.step_width / self.horizontal_scale))
        plat_px = int(round(self.platform_width / self.horizontal_scale))
        border_px = int(round(self.border_width / self.horizontal_scale))
        step_units = int(round(step_height / self.vertical_scale))

        noise = np.zeros((W, L), dtype=np.int16)

        inner = min(W, L) - 2 * border_px
        n_steps = max(0, (inner - plat_px) // (2 * step_px)) if step_px > 0 else 0

        # Rings descend (negative values) from outer to inner.
        for k in range(n_steps):
            lo_x = border_px + k * step_px
            hi_x = W - border_px - k * step_px
            lo_y = border_px + k * step_px
            hi_y = L - border_px - k * step_px
            noise[lo_x:hi_x, lo_y:hi_y] = -(k + 1) * step_units

        plat_lo_x = border_px + n_steps * step_px
        plat_hi_x = W - border_px - n_steps * step_px
        plat_lo_y = border_px + n_steps * step_px
        plat_hi_y = L - border_px - n_steps * step_px
        noise[plat_lo_x:plat_hi_x, plat_lo_y:plat_hi_y] = -(n_steps + 1) * step_units

        if self.holes:
            min_existing = int(noise.min())
            pit_units = min_existing - int(round(self.pit_depth / self.vertical_scale))
            if pit_units < np.iinfo(np.int16).min:
                raise ValueError(
                    f"pit_depth={self.pit_depth} m below platform at vertical_scale="
                    f"{self.vertical_scale} m overflows int16 range."
                )
            for k in range(n_steps):
                lo_x = border_px + k * step_px
                hi_x = W - border_px - k * step_px
                lo_y = border_px + k * step_px
                hi_y = L - border_px - k * step_px
                noise[lo_x : lo_x + step_px, lo_y : lo_y + step_px] = pit_units
                noise[lo_x : lo_x + step_px, hi_y - step_px : hi_y] = pit_units
                noise[hi_x - step_px : hi_x, lo_y : lo_y + step_px] = pit_units
                noise[hi_x - step_px : hi_x, hi_y - step_px : hi_y] = pit_units

        # Place data such that the original "0" layer (outer ring top) sits at
        # world z=0 — matches the existing HfPyramidSloped(inverted) convention.
        spawn_z = -(n_steps + 1) * step_units * self.vertical_scale
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_z])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
            z_offset_fn=lambda max_h: -max_h,
        )


@dataclass(kw_only=True)
class BoxPyramidStairsTerrainCfg(SubTerrainCfg):
    """Pyramid stairs terrain using MuJoCo box primitives.

    Hfield is flat (z=0) — only box geoms carry the geometry.  A combined
    surface sampler (`CombinedSurfaceSampler`) resolves heights by taking
    max(hfield, box) per sample point, so spawning and height queries stay
    accurate without hfield stair-edge blur in collision.

    Optional random obstacles can be scattered on the stair surfaces to
    train robustness for stair descent with unexpected bumps/debris.
    """

    step_height_range: tuple[float, float]
    """Min and max step height, in meters. Interpolated by difficulty."""
    step_width: float
    """Depth (run) of each step, in meters."""
    platform_width: float = 1.0
    """Side length of the flat square platform at the top, in meters."""
    border_width: float = 0.0
    """Width of the flat outer border around the staircase, in meters."""
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 1.0
    obstacle_per_step: int = 0
    """Number of random obstacles per visible step ring. 0 = disabled."""
    obstacle_half_width: float = 0.05
    """Fixed half-width (travel direction), in meters. Robot must step over this."""
    obstacle_half_length_range: tuple[float, float] = (0.10, 0.30)
    """Min and max half-length (along step edge), in meters. Blocks more of the step."""
    obstacle_height_range: tuple[float, float] = (0.02, 0.06)
    """Min and max height of obstacle boxes (z), in meters."""
    obstacle_margin: float = 0.12
    """Minimum distance from step edges to avoid placing obstacles on boundaries."""

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )

        cell_w, cell_l = self.size
        W = int(round(cell_w / self.horizontal_scale))
        L = int(round(cell_l / self.horizontal_scale))
        step_px = int(round(self.step_width / self.horizontal_scale))
        plat_px = int(round(self.platform_width / self.horizontal_scale))
        border_px = int(round(self.border_width / self.horizontal_scale))

        noise = np.zeros((W, L), dtype=np.int16)

        inner = min(W, L) - 2 * border_px
        n_steps = max(0, (inner - plat_px) // (2 * step_px)) if step_px > 0 else 0

        half_h = step_height / 2
        step_width_m = self.step_width
        border_w_m = self.border_width
        box_geoms: list[BoxGeomSpec] = []
        for k in range(n_steps):
            lo = border_w_m + k * step_width_m
            size_xy = (cell_w - lo * 2) / 2
            box_geoms.append(BoxGeomSpec(
                size=(size_xy, size_xy, half_h),
                pos=(cell_w / 2, cell_l / 2, (k + 1) * step_height - half_h),
            ))
        plat_lo = border_w_m + n_steps * step_width_m
        plat_size_xy = (cell_w - plat_lo * 2) / 2
        box_geoms.append(BoxGeomSpec(
            size=(plat_size_xy, plat_size_xy, half_h),
            pos=(cell_w / 2, cell_l / 2, (n_steps + 1) * step_height - half_h),
        ))

        # Random obstacles on stair surfaces — per-step allocation
        if self.obstacle_per_step > 0 and n_steps > 0:
            for k in range(n_steps):
                lo = border_w_m + k * step_width_m
                hi = border_w_m + (k + 1) * step_width_m
                z_surface = (k + 1) * step_height
                for _ in range(self.obstacle_per_step):
                    obs_half_len = rng.uniform(*self.obstacle_half_length_range)
                    obs_height = rng.uniform(*self.obstacle_height_range)
                    obs_half_z = obs_height / 2.0
                    edge = rng.integers(0, 4)
                    if edge == 0:  # bottom strip
                        ox = rng.uniform(lo + self.obstacle_margin,
                                         cell_w - lo - self.obstacle_margin)
                        oy = rng.uniform(lo + self.obstacle_margin,
                                         hi - self.obstacle_margin)
                        size_xy = (obs_half_len, self.obstacle_half_width)
                    elif edge == 1:  # top strip
                        ox = rng.uniform(lo + self.obstacle_margin,
                                         cell_w - lo - self.obstacle_margin)
                        oy = rng.uniform(cell_l - hi + self.obstacle_margin,
                                         cell_l - lo - self.obstacle_margin)
                        size_xy = (obs_half_len, self.obstacle_half_width)
                    elif edge == 2:  # left strip
                        ox = rng.uniform(lo + self.obstacle_margin,
                                         hi - self.obstacle_margin)
                        oy = rng.uniform(hi + self.obstacle_margin,
                                         cell_l - hi - self.obstacle_margin)
                        size_xy = (self.obstacle_half_width, obs_half_len)
                    else:  # right strip
                        ox = rng.uniform(cell_w - hi + self.obstacle_margin,
                                         cell_w - lo - self.obstacle_margin)
                        oy = rng.uniform(hi + self.obstacle_margin,
                                         cell_l - hi - self.obstacle_margin)
                        size_xy = (self.obstacle_half_width, obs_half_len)
                    box_geoms.append(BoxGeomSpec(
                        size=(*size_xy, obs_half_z),
                        pos=(ox, oy, z_surface + obs_half_z),
                    ))

        spawn_z = (n_steps + 1) * step_height
        origin = np.array([cell_w / 2, cell_l / 2, spawn_z])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
            box_geoms=box_geoms,
        )


@dataclass(kw_only=True)
class HfFlatTerrainCfg(SubTerrainCfg):
    """A flat heightfield terrain (all-zero noise array)."""

    horizontal_scale: float = 0.05
    """Heightfield grid resolution. Overwritten by TerrainGenerator."""
    vertical_scale: float = 0.005
    """Heightfield height resolution. Overwritten by TerrainGenerator."""
    base_thickness_ratio: float = 0.0
    """Ratio of the heightfield base thickness to its surface height. The
    helper enforces a minimum thickness so a literal zero is fine here."""

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        del difficulty  # Unused.

        width_pixels = int(round(self.size[0] / self.horizontal_scale))
        length_pixels = int(round(self.size[1] / self.horizontal_scale))
        noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
        )


@dataclass(kw_only=True)
class HfPlateauTerrainCfg(SubTerrainCfg):
    """中央高台地形：平坦的高台 + 四周地面，训练机器人从高台边缘下落保持稳定。

    高台是矩形平面，边缘为陡峭落差（非斜坡），四周为平坦地面。
    """

    platform_height_range: tuple[float, float] = (0.05, 0.45)
    """高台高度范围 (米)，由 difficulty 插值。"""
    platform_width: float = 3.0
    """高台平台边长 (米)。"""
    border_width: float = 0.5
    """高台外平坦地面宽度 (米)。"""
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 1.0

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        platform_height = (
            self.platform_height_range[0]
            + difficulty * (self.platform_height_range[1] - self.platform_height_range[0])
        )

        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)
        border_pixels = int(self.border_width / self.horizontal_scale)
        platform_pixels = int(self.platform_width / self.horizontal_scale)

        noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        height_value = int(round(platform_height / self.vertical_scale))

        # 计算高台在网格中的位置 (居中)
        x_start = (width_pixels - platform_pixels) // 2
        x_end = x_start + platform_pixels
        y_start = (length_pixels - platform_pixels) // 2
        y_end = y_start + platform_pixels

        # 确保高台不超出边界
        x_start = max(x_start, border_pixels)
        x_end = min(x_end, width_pixels - border_pixels)
        y_start = max(y_start, border_pixels)
        y_end = min(y_end, length_pixels - border_pixels)

        noise[x_start:x_end, y_start:y_end] = height_value

        terrain_hfield = _build_heightfield(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            z_offset_fn=lambda max_h: 0.0,
        )

        origin = np.array([
            self.size[0] / 2,
            self.size[1] / 2,
            terrain_hfield.max_physical_height,
        ])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
            z_offset_fn=lambda max_h: 0.0,
        )


@dataclass(kw_only=True)
class BoxPlateauTerrainCfg(SubTerrainCfg):
    """中央高台地形，使用 MuJoCo box 基元。

    Hfield 全平 (z=0)，只有 box geom 承载高台几何体。
    相比 HfPlateauTerrainCfg，box 边缘精确不模糊，避免从高台边缘
    踏空时脚穿透斜坡的问题。
    """

    platform_height_range: tuple[float, float] = (0.05, 0.45)
    """高台高度范围 (米)，由 difficulty 插值。"""
    platform_width: float = 3.0
    """高台平台边长 (米)。"""
    border_width: float = 0.5
    """高台外平坦地面宽度 (米)。"""
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 1.0

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        platform_height = (
            self.platform_height_range[0]
            + difficulty * (self.platform_height_range[1] - self.platform_height_range[0])
        )

        cell_w, cell_l = self.size
        W = int(round(cell_w / self.horizontal_scale))
        L = int(round(cell_l / self.horizontal_scale))

        noise = np.zeros((W, L), dtype=np.int16)

        half_h = platform_height / 2
        plat_size_xy = self.platform_width / 2
        box_geoms: list[BoxGeomSpec] = [
            BoxGeomSpec(
                size=(plat_size_xy, plat_size_xy, half_h),
                pos=(cell_w / 2, cell_l / 2, platform_height - half_h),
            ),
        ]

        spawn_z = platform_height
        origin = np.array([cell_w / 2, cell_l / 2, spawn_z])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
            box_geoms=box_geoms,
            z_offset_fn=lambda max_h: 0.0,
        )


@dataclass(kw_only=True)
class BoxInvertedPyramidStairsTerrainCfg(SubTerrainCfg):
    """倒金字塔台阶地形，使用 MuJoCo box 门框（每层4根方条围成正方形框）。

    Hfield 全平沉到坑底，只有 box 门框承载下沉台阶几何体。
    从外框到内框逐级下陷，中心平台最深（hf 面）。

    与 HfInvertedPyramidStairsTerrainCfg 的区别：box 边缘精确不模糊，
    避免高度场锯齿导致的接触约束不稳定。

    """

    step_height_range: tuple[float, float]
    """Min and max step height, in meters. Interpolated by difficulty."""
    step_width: float
    """Depth (run) of each step, also the thickness of each frame bar."""
    platform_width: float = 1.0
    """Side length of the flat square hole at the bottom, in meters."""
    border_width: float = 0.0
    """Width of the flat outer border around the staircase, in meters."""
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 1.0

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )

        cell_w, cell_l = self.size
        W = int(round(cell_w / self.horizontal_scale))
        L = int(round(cell_l / self.horizontal_scale))
        step_px = int(round(self.step_width / self.horizontal_scale))
        plat_px = int(round(self.platform_width / self.horizontal_scale))
        border_px = int(round(self.border_width / self.horizontal_scale))

        noise = np.zeros((W, L), dtype=np.int16)
        half_h = step_height / 2
        step_width_m = self.step_width
        border_w_m = self.border_width

        inner = min(W, L) - 2 * border_px
        n_steps = max(0, (inner - plat_px) // (2 * step_px) - 1) if step_px > 0 else 0
        box_geoms: list[BoxGeomSpec] = []

        for k in range(n_steps + 1):
            z_center = -(k + 0.5) * step_height
            s_center = border_w_m + (k + 0.5) * step_width_m
            bar_half_w = step_width_m / 2

            if k == 0:
                inner_edge = border_w_m + step_width_m
                short_half = inner_edge / 2
                short_center = inner_edge / 2
                long_extent = cell_w / 2
            else:
                short_half = bar_half_w
                short_center = s_center
                long_extent = max((cell_w - 2 * s_center) / 2 + bar_half_w, bar_half_w)

            box_geoms.extend([
                BoxGeomSpec(
                    size=(long_extent, short_half, half_h),
                    pos=(cell_w / 2, short_center, z_center),
                ),
                BoxGeomSpec(
                    size=(long_extent, short_half, half_h),
                    pos=(cell_w / 2, cell_l - short_center, z_center),
                ),
                BoxGeomSpec(
                    size=(short_half, max(long_extent - 2 * short_half, 0.0), half_h),
                    pos=(short_center, cell_l / 2, z_center),
                ),
                BoxGeomSpec(
                    size=(short_half, max(long_extent - 2 * short_half, 0.0), half_h),
                    pos=(cell_w - short_center, cell_l / 2, z_center),
                ),
            ])

        pit_bottom = -(n_steps + 1) * step_height
        spawn_z = pit_bottom
        origin = np.array([cell_w / 2, cell_l / 2, spawn_z])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
            box_geoms=box_geoms,
            z_offset_fn=lambda max_h: pit_bottom,
        )


@dataclass(kw_only=True)
class BoxWaveTerrainCfg(SubTerrainCfg):
    """Wave terrain built with truncated-cone (frustum) cylinder geoms on a flat hfield.

    Truncated cones lie horizontally along the Y axis (walking direction).
    Diameter and spacing both scale with difficulty: small/dense at easy
    levels, large/sparse at hard levels.
    """

    amplitude_range: tuple[float, float] = (0.0, 0.15)
    """Min and max amplitude of the cosine envelope, in meters. Interpolated by difficulty."""
    num_waves: int = 20
    """Number of complete wave cycles along the terrain length."""
    cone_height: float = 0.08
    """Height (Z half-length * 2) of each truncated cone, in meters."""
    bottom_diameter_min: float = 0.06
    """Bottom diameter at difficulty=0, in meters."""
    bottom_diameter_max: float = 0.16
    """Bottom diameter at difficulty=1, in meters."""
    spacing_min: float | None = None
    """Ridge spacing at difficulty=0, in meters. Defaults to bottom_diameter_min."""
    spacing_max: float | None = None
    """Ridge spacing at difficulty=1, in meters. Defaults to bottom_diameter_max."""
    border_width: float = 0.25
    """Width of the flat border around the terrain edges, in meters."""
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 0.25

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:

        cell_w, cell_l = self.size
        W = int(round(cell_w / self.horizontal_scale))
        L = int(round(cell_l / self.horizontal_scale))
        noise = np.zeros((W, L), dtype=np.int16)

        inner_l = cell_l - 2.0 * self.border_width
        inner_w = cell_w - 2.0 * self.border_width
        half_len_y = cell_l / 2.0

        bd = self.bottom_diameter_min + difficulty * (self.bottom_diameter_max - self.bottom_diameter_min)
        td = bd / 2.0
        br = bd / 2.0
        tr = td / 2.0

        sp_min = self.spacing_min if self.spacing_min is not None else self.bottom_diameter_min
        sp_max = self.spacing_max if self.spacing_max is not None else self.bottom_diameter_max
        spacing = sp_min + difficulty * (sp_max - sp_min)

        quat_y = (0.7071068, 0.7071068, 0.0, 0.0)

        cylinder_geoms: list[CylinderGeomSpec] = []
        x0 = self.border_width
        r_bound = cell_w - self.border_width
        while x0 < r_bound:
            cx = x0 + br
            if cx + br > r_bound:
                break
            cylinder_geoms.append(CylinderGeomSpec(
                size=(tr, br, half_len_y),
                pos=(cx, cell_l / 2.0, 0.0),
                quat=quat_y,
            ))
            x0 += spacing

        origin = np.array([cell_w / 2, cell_l / 2, 0.0])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
            cylinder_geoms=cylinder_geoms,
        )