from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_BORDER_BASE_THICKNESS = 0.05


@dataclass
class FlatPatchSamplingCfg:
    """Configuration for sampling flat patches on a heightfield surface."""

    num_patches: int = 10
    """Number of flat patches to sample per sub-terrain."""
    patch_radius: float = 0.5
    """Radius of the circular footprint used to test flatness, in meters."""
    max_height_diff: float = 0.05
    """Maximum allowed height variation within the patch footprint, in meters."""
    x_range: tuple[float, float] = (-1e6, 1e6)
    """Allowed range of x coordinates for sampled patches, in meters."""
    y_range: tuple[float, float] = (-1e6, 1e6)
    """Allowed range of y coordinates for sampled patches, in meters."""
    z_range: tuple[float, float] = (-1e6, 1e6)
    """Allowed range of z coordinates (world height) for sampled patches, in meters."""
    grid_resolution: float | None = None
    """Resolution of the grid used for flat-patch detection, in meters. When
    ``None`` (default), the terrain's own ``horizontal_scale`` is used. Set to a
    smaller value (e.g. 0.025) for finer boundary precision at the cost of a
    larger intermediate grid."""


@dataclass
class TerrainHeightField:
    """Backend-agnostic heightfield data for one sub-terrain patch."""

    noise: np.ndarray
    """Quantized height units before normalization."""
    size: tuple[float, float]
    """Patch size as ``(x, y)`` in meters."""
    horizontal_scale: float
    vertical_scale: float
    elevation_min: int
    elevation_max: int
    max_physical_height: float
    base_thickness: float
    z_offset: float

    def physical_heights_xy(self) -> np.ndarray:
        """Return world-space surface heights as an ``(x, y)`` matrix."""
        return (
            self.noise.astype(np.float64) - self.elevation_min
        ) * self.vertical_scale + self.z_offset

    def normalized_elevation(self) -> np.ndarray:
        """Return normalized hfield values in ``[0, 1]``."""
        elevation_range = self.elevation_max - self.elevation_min
        if elevation_range <= 0:
            return np.zeros_like(self.noise, dtype=np.float64)
        return (self.noise.astype(np.float64) - self.elevation_min) / elevation_range


@dataclass
class BoxGeomSpec:
    """Specification for a box geom to inject into the MuJoCo scene."""

    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    name: str | None = None
    material_rgba: tuple[float, float, float, float] | None = None
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)


@dataclass
class CylinderGeomSpec:
    """Specification for a cylinder/frustum geom to inject into the MuJoCo scene.

    size: (radius_top, radius_bottom, half_length) — when radii differ, MuJoCo
    renders it as a truncated cone (frustum).
    quat: (w, x, y, z) — orientation quaternion. Defaults to identity (upright).
    """

    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    name: str | None = None
    material_rgba: tuple[float, float, float, float] | None = None
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class TerrainOutput:
    origin: np.ndarray
    """Spawn origin position (x, y, z) in the sub-terrain's local frame."""
    heightfield: TerrainHeightField
    """Backend-agnostic heightfield data."""
    flat_patches: dict[str, np.ndarray] | None = None
    """Named sets of flat patch positions, each an (N, 3) array. None if not configured."""
    box_geoms: list[BoxGeomSpec] | None = None
    """Optional box geoms for mesh/box-based terrain. World-relative positions."""
    cylinder_geoms: list[CylinderGeomSpec] | None = None
    """Optional cylinder/frustum geoms. World-relative positions."""


@dataclass
class GeneratedTerrain:
    """Merged terrain heightfield ready to be exported as a single PNG asset."""

    heights_yx: np.ndarray
    """World-space surface heights in image convention: rows=y, cols=x."""
    horizontal_scale: float
    z_min: float
    z_max: float
    base_thickness: float
    terrain_origins: np.ndarray
    box_geoms: list[BoxGeomSpec] = field(default_factory=list)
    """Box geoms collected from sub-terrains, positioned in world frame."""
    cylinder_geoms: list[CylinderGeomSpec] = field(default_factory=list)
    """Cylinder/frustum geoms collected from sub-terrains, positioned in world frame."""
    cell_size: tuple[float, float] | None = None
    """Sub-terrain cell size (width, length) in meters. Set when grid layout is used."""
    num_rows: int = 0
    """Number of sub-terrain rows."""
    num_cols: int = 0
    """Number of sub-terrain columns."""

    @property
    def size(self) -> tuple[float, float]:
        rows_y, cols_x = self.heights_yx.shape
        return (cols_x * self.horizontal_scale, rows_y * self.horizontal_scale)

    @property
    def height_extent(self) -> float:
        return max(self.z_max - self.z_min, self.horizontal_scale * 0.02)

    @property
    def hfield_size(self) -> tuple[float, float, float, float]:
        size_x, size_y = self.size
        return (size_x / 2, size_y / 2, self.height_extent, self.base_thickness)

    @property
    def geom_pos(self) -> tuple[float, float, float]:
        return (0.0, 0.0, self.z_min)

    def to_uint16(self) -> np.ndarray:
        span = self.z_max - self.z_min
        if span <= 0.0:
            return np.zeros_like(self.heights_yx, dtype=np.uint16)
        normalized = (self.heights_yx - self.z_min) / span
        return np.rint(np.clip(normalized, 0.0, 1.0) * np.iinfo(np.uint16).max).astype(np.uint16)

    def surface_sampler(self) -> HeightfieldSurfaceSampler | CombinedSurfaceSampler:
        hfield = HeightfieldSurfaceSampler(
            heights_uint16=self.to_uint16(),
            horizontal_scale=float(self.horizontal_scale),
            z_min=float(self.z_min),
            height_extent=float(self.height_extent),
        )
        if self.box_geoms or self.cylinder_geoms:
            return CombinedSurfaceSampler(hfield, self.box_geoms, self.cylinder_geoms,
                                         cell_size=self.cell_size,
                                         num_rows=self.num_rows,
                                         num_cols=self.num_cols)
        return hfield

    def write_png(self, path: Path) -> None:
        """Write the merged hfield as a 16-bit grayscale PNG."""
        import imageio.v3 as iio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(path, self.to_uint16())

    def hfield_size_xml(self) -> str:
        return " ".join(f"{value:.9g}" for value in self.hfield_size)

    def geom_pos_xml(self) -> str:
        return " ".join(f"{value:.9g}" for value in self.geom_pos)


@dataclass
class HeightfieldSurfaceSampler:
    """Compact world-space sampler for a generated hfield surface."""

    heights_uint16: np.ndarray
    horizontal_scale: float
    z_min: float
    height_extent: float

    @property
    def size(self) -> tuple[float, float]:
        rows_y, cols_x = self.heights_uint16.shape
        return (cols_x * self.horizontal_scale, rows_y * self.horizontal_scale)

    def sample_height(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 2:
            raise ValueError(f"xy must have shape (..., 2), got {points.shape}")

        flat = points.reshape(-1, 2)
        size_x, size_y = self.size
        rows_y, cols_x = self.heights_uint16.shape
        cols = np.rint((flat[:, 0] + size_x * 0.5) / self.horizontal_scale).astype(np.intp)
        rows = np.rint((-flat[:, 1] + size_y * 0.5) / self.horizontal_scale).astype(np.intp)
        cols = np.clip(cols, 0, cols_x - 1)
        rows = np.clip(rows, 0, rows_y - 1)

        raw = self.heights_uint16[rows, cols].astype(np.float64)
        normalized = raw / float(np.iinfo(np.uint16).max)
        heights = self.z_min + normalized * self.height_extent
        return heights.reshape(points.shape[:-1])


class CombinedSurfaceSampler:
    """Surface sampler that queries both the hfield and a list of box/cylinder geoms.

    For each sample point the returned height is ``max(hfield_height, top of
    any box/cylinder geom that covers the point)``.  This lets box-based terrain
    (e.g. pyramid stairs) coexist with hfield-based terrain in the same grid —
    the boxes provide sharp collision edges while the hfield feeds the height
    scanner in cells without boxes.

    When *cell_size*, *num_rows* and *num_cols* are provided, cylinders are
    pre-partitioned by their cell and sample queries only check the cell that
    contains each point, dramatically reducing the number of collision checks.
    """

    def __init__(
        self,
        hfield_sampler: HeightfieldSurfaceSampler,
        box_geoms: list[BoxGeomSpec],
        cylinder_geoms: list[CylinderGeomSpec] | None = None,
        cell_size: tuple[float, float] | None = None,
        num_rows: int = 0,
        num_cols: int = 0,
    ) -> None:
        self._hfield = hfield_sampler
        self._boxes = box_geoms
        self._cylinders = cylinder_geoms or []

        self._cyl_by_cell: list[list[CylinderGeomSpec]] | None = None
        self._grid_ofs = (0.0, 0.0)
        self._cell_w = 0.0
        self._cell_l = 0.0
        self._grid_rows = 0
        self._grid_cols = 0
        if cell_size is not None and num_rows > 0 and num_cols > 0:
            cw, cl = cell_size
            gox = -num_rows * cw / 2.0
            goy = -num_cols * cl / 2.0
            cyl_by_cell: list[list[CylinderGeomSpec]] = [[] for _ in range(num_rows * num_cols)]
            for c in self._cylinders:
                row = max(0, min(int((c.pos[0] - gox) / cw), num_rows - 1))
                col = max(0, min(int((c.pos[1] - goy) / cl), num_cols - 1))
                cyl_by_cell[row * num_cols + col].append(c)
            self._cyl_by_cell = cyl_by_cell
            self._grid_ofs = (gox, goy)
            self._cell_w = cw
            self._cell_l = cl
            self._grid_rows = num_rows
            self._grid_cols = num_cols

    def sample_height(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 2:
            raise ValueError(f"xy must have shape (..., 2), got {points.shape}")
        flat = points.reshape(-1, 2)
        h = self._hfield.sample_height(flat)
        for b in self._boxes:
            inside = (np.abs(flat[:, 0] - b.pos[0]) <= b.size[0]) & (
                np.abs(flat[:, 1] - b.pos[1]) <= b.size[1]
            )
            h = np.maximum(h, np.where(inside, b.pos[2] + b.size[2], h))
        if self._cyl_by_cell is not None:
            h = self._sample_cylinders_celled(flat, h)
        else:
            for c in self._cylinders:
                half_len = c.size[2]
                top_r, bottom_r = c.size[0], c.size[1]
                max_r = max(top_r, bottom_r)
                if c.quat == (1.0, 0.0, 0.0, 0.0):
                    inside = (np.abs(flat[:, 0] - c.pos[0]) <= max_r) & (
                        np.abs(flat[:, 1] - c.pos[1]) <= max_r
                    )
                    h = np.maximum(h, np.where(inside, c.pos[2] + half_len, h))
                else:
                    inside = (np.abs(flat[:, 0] - c.pos[0]) <= max_r) & (
                        np.abs(flat[:, 1] - c.pos[1]) <= half_len
                    )
                    t = np.clip((flat[inside, 1] - (c.pos[1] - half_len)) / (2 * half_len), 0.0, 1.0)
                    r_at_y = top_r + t * (bottom_r - top_r)
                    h[inside] = np.maximum(h[inside], c.pos[2] + r_at_y)
        return h.reshape(points.shape[:-1])

    def _sample_cylinders_celled(self, flat: np.ndarray, h: np.ndarray) -> np.ndarray:
        gox, goy = self._grid_ofs
        nr, nc = self._grid_rows, self._grid_cols
        rows = np.clip(((flat[:, 0] - gox) / self._cell_w).astype(np.int32), 0, nr - 1)
        cols = np.clip(((flat[:, 1] - goy) / self._cell_l).astype(np.int32), 0, nc - 1)
        cyl_by_cell = self._cyl_by_cell  # type: ignore[assignment]
        for ri in range(nr):
            for ci in range(nc):
                clist = cyl_by_cell[ri * nc + ci]
                if not clist:
                    continue
                mask = (rows == ri) & (cols == ci)
                if not np.any(mask):
                    continue
                f_cell = flat[mask]
                h_cell = h[mask]
                for c in clist:
                    half_len = c.size[2]
                    top_r, bottom_r = c.size[0], c.size[1]
                    max_r = max(top_r, bottom_r)
                    if c.quat == (1.0, 0.0, 0.0, 0.0):
                        inside = (np.abs(f_cell[:, 0] - c.pos[0]) <= max_r) & (
                            np.abs(f_cell[:, 1] - c.pos[1]) <= max_r
                        )
                        h_cell = np.maximum(h_cell, np.where(inside, c.pos[2] + half_len, h_cell))
                    else:
                        inside = (np.abs(f_cell[:, 0] - c.pos[0]) <= max_r) & (
                            np.abs(f_cell[:, 1] - c.pos[1]) <= half_len
                        )
                        t = np.clip(
                            (f_cell[inside, 1] - (c.pos[1] - half_len)) / (2 * half_len), 0.0, 1.0,
                        )
                        r_at_y = top_r + t * (bottom_r - top_r)
                        h_cell[inside] = np.maximum(h_cell[inside], c.pos[2] + r_at_y)
                h[mask] = h_cell
        return h


@dataclass
class SubTerrainCfg(abc.ABC):
    proportion: float = 1.0
    """Robot spawning weight for this terrain type.

    In curriculum mode, controls how many robots are spawned on this terrain's
    column relative to other terrain types. Each terrain type always gets
    exactly one column; proportion only affects spawning distribution.

    In random mode, controls the sampling probability for each patch.
    """
    size: tuple[float, float] = (10.0, 10.0)
    """Width and length of the terrain patch, in meters."""
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    """Named flat-patch sampling configurations, or None to disable."""

    @abc.abstractmethod
    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        """Generate backend-agnostic terrain data.

        Returns:
            TerrainOutput containing spawn origin and heightfield data.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class TerrainGeneratorCfg:
    seed: int | None = None
    """Random seed for terrain generation. None uses a random seed."""
    curriculum: bool = False
    """Controls terrain allocation mode:

    - curriculum=True: Each terrain type gets exactly ONE column. The generator uses
        ``len(sub_terrains)`` columns regardless of ``num_cols``. Difficulty increases
        along rows. The ``proportion`` field controls how many robots are spawned per
        column, not column count.

    - curriculum=False: Every patch is randomly sampled from all terrain types.
        Proportions control sampling probability. Use this for random variety.
    """
    size: tuple[float, float]
    """Width and length of each sub-terrain patch, in meters. Both components
    must be integer multiples of ``horizontal_scale``."""
    horizontal_scale: float = 0.05
    """Heightfield grid resolution along x and y, in meters per cell. Shared by
    every sub-terrain (overwritten in :class:`TerrainGenerator` ``__init__``).
    All length-like sub-terrain parameters (step_width, platform_width,
    border_width, etc.) must be integer multiples of this value."""
    vertical_scale: float = 0.005
    """Heightfield height resolution, in meters per integer unit of the noise
    array. Shared by every sub-terrain (overwritten in
    :class:`TerrainGenerator` ``__init__``)."""
    border_width: float = 0.0
    """Width of the flat border around the entire terrain grid, in meters. Must
    be an integer multiple of ``horizontal_scale`` if non-zero. The border is a
    flat hfield slab whose top surface is flush with the inner-terrain floor at
    z=0; it is NOT a wall."""
    num_rows: int = 1
    """Number of sub-terrain rows in the grid. Represents difficulty levels in
    curriculum mode. Note: Environments are randomly assigned to rows, so multiple
    envs can share the same patch."""
    num_cols: int = 1
    """Number of sub-terrain columns in the grid.

    In curriculum mode the generator ignores this value and uses one column per terrain
    type (``len(sub_terrains)``). In random mode it is used as-is."""
    sub_terrains: dict[str, SubTerrainCfg] = field(default_factory=dict)
    """Named sub-terrain configurations to populate the grid."""
    difficulty_range: tuple[float, float] = (0.0, 1.0)
    """Min and max difficulty values used when generating sub-terrains."""
    add_lights: bool = False
    """If True, adds a directional light above the terrain grid."""

    selected_terrain_types: Optional[list[str]] = None
    """Optional list of terrain type names to keep; if set, only these types are used
    and their proportions are renormalized to sum to 1."""


class TerrainGenerator:
    """Generates procedural terrain grids with configurable difficulty.

    Creates a grid of terrain patches where each patch can be a different
    terrain type. Supports two modes:

    - **Random mode** (curriculum=False): Every patch independently samples a
        terrain type weighted by proportions. Results in random variety across
        all patches.

    - **Curriculum mode** (curriculum=True): Each terrain type gets exactly one column
        (the generator uses ``len(sub_terrains)`` columns regardless of ``num_cols``).
        Difficulty increases along rows. The ``proportion`` field controls robot spawning
        distribution, not column count.

    Terrain types are weighted by proportion and their geometry is generated
    based on a difficulty value in the configured range. The grid is centered
    at the world origin. A border can be added around the entire grid along with
    optional overhead lighting.
    """

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu") -> None:
        if _selected := cfg.selected_terrain_types:
            total_prop = sum(cfg.sub_terrains[t].proportion for t in _selected)
            cfg.sub_terrains = {t: cfg.sub_terrains[t] for t in _selected}
            norm = 1.0 / total_prop if total_prop > 0 else 1.0
            for sub in cfg.sub_terrains.values():
                object.__setattr__(sub, "proportion", sub.proportion * norm)
        if len(cfg.sub_terrains) == 0:
            raise ValueError("At least one sub_terrain must be specified.")

        self.cfg = cfg
        self.device = device

        # In curriculum mode, one column per terrain type.
        if self.cfg.curriculum:
            self._num_cols = len(self.cfg.sub_terrains)
        else:
            self._num_cols = self.cfg.num_cols

        for sub_cfg in self.cfg.sub_terrains.values():
            sub_cfg.size = self.cfg.size

        self._propagate_resolution()
        self._validate_resolution()

        if self.cfg.seed is not None:
            seed = self.cfg.seed
        else:
            seed = np.random.randint(0, 10000)
        self.np_rng = np.random.default_rng(seed)

        self.terrain_origins = np.zeros((self.cfg.num_rows, self._num_cols, 3))

        # Pre-allocate flat patch storage by scanning all sub-terrain configs.
        self.flat_patches: dict[str, np.ndarray] = {}
        self.flat_patch_radii: dict[str, float] = {}
        patch_names: dict[str, int] = {}
        for sub_cfg in self.cfg.sub_terrains.values():
            if sub_cfg.flat_patch_sampling is not None:
                for name, patch_cfg in sub_cfg.flat_patch_sampling.items():
                    if name in patch_names:
                        patch_names[name] = max(patch_names[name], patch_cfg.num_patches)
                    else:
                        patch_names[name] = patch_cfg.num_patches
                    self.flat_patch_radii[name] = max(
                        self.flat_patch_radii.get(name, 0.0), patch_cfg.patch_radius
                    )
        for name, max_num_patches in patch_names.items():
            self.flat_patches[name] = np.zeros(
                (self.cfg.num_rows, self._num_cols, max_num_patches, 3)
            )

    def generate(self) -> GeneratedTerrain:
        """Generate the full terrain as one backend-agnostic merged hfield."""
        tile_x_px = int(round(self.cfg.size[0] / self.cfg.horizontal_scale))
        tile_y_px = int(round(self.cfg.size[1] / self.cfg.horizontal_scale))
        border_px = int(round(self.cfg.border_width / self.cfg.horizontal_scale))
        rows_y = self._num_cols * tile_y_px + 2 * border_px
        cols_x = self.cfg.num_rows * tile_x_px + 2 * border_px
        heights_yx = np.zeros((rows_y, cols_x), dtype=np.float64)

        max_base_thickness = _BORDER_BASE_THICKNESS if self.cfg.border_width > 0.0 else 0.0
        self.terrain_origins.fill(0.0)
        all_box_geoms: list[BoxGeomSpec] = []
        all_cylinder_geoms: list[CylinderGeomSpec] = []

        def place_output(output: TerrainOutput, sub_row: int, sub_col: int) -> None:
            nonlocal max_base_thickness
            patch_heights_xy = output.heightfield.physical_heights_xy()
            if patch_heights_xy.shape != (tile_x_px, tile_y_px):
                raise ValueError(
                    "Sub-terrain heightfield shape does not match TerrainGeneratorCfg.size: "
                    f"{patch_heights_xy.shape} != {(tile_x_px, tile_y_px)}"
                )
            x0 = border_px + sub_row * tile_x_px
            y0 = border_px + (self._num_cols - 1 - sub_col) * tile_y_px
            heights_yx[y0 : y0 + tile_y_px, x0 : x0 + tile_x_px] = patch_heights_xy.T

            max_base_thickness = max(
                max_base_thickness, output.heightfield.base_thickness,
            )

            world_position = self._get_sub_terrain_position(sub_row, sub_col)
            spawn_origin = output.origin + world_position
            self.terrain_origins[sub_row, sub_col] = spawn_origin
            arr = self.flat_patches
            if output.flat_patches:
                for name, patches in output.flat_patches.items():
                    if name not in arr:
                        continue
                    patches = np.asarray(patches, dtype=np.float64)
                    arr[name][sub_row, sub_col, : len(patches)] = patches + world_position
                    arr[name][sub_row, sub_col, len(patches) :] = spawn_origin
            if output.box_geoms:
                for bg in output.box_geoms:
                    wx, wy, wz = bg.pos
                    bg.pos = (wx + world_position[0], wy + world_position[1], wz)
                all_box_geoms.extend(output.box_geoms)
            if output.cylinder_geoms:
                for cg in output.cylinder_geoms:
                    wx, wy, wz = cg.pos
                    cg.pos = (wx + world_position[0], wy + world_position[1], wz)
                all_cylinder_geoms.extend(output.cylinder_geoms)

        if self.cfg.curriculum:
            sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
            for sub_col in range(self._num_cols):
                for sub_row in range(self.cfg.num_rows):
                    lower, upper = self.cfg.difficulty_range
                    difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                    difficulty = lower + (upper - lower) * difficulty
                    output = sub_terrains_cfgs[sub_col].function(difficulty, self.np_rng)
                    place_output(output, sub_row, sub_col)
        else:
            proportions = np.array(
                [sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()]
            )
            proportions /= np.sum(proportions)
            sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
            for index in range(self.cfg.num_rows * self._num_cols):
                sub_row, sub_col = np.unravel_index(index, (self.cfg.num_rows, self._num_cols))
                sub_row = int(sub_row)
                sub_col = int(sub_col)
                sub_index = self.np_rng.choice(len(proportions), p=proportions)
                difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
                output = sub_terrains_cfgs[sub_index].function(difficulty, self.np_rng)
                place_output(output, sub_row, sub_col)

        z_min = float(np.min(heights_yx))
        z_max = float(np.max(heights_yx))
        return GeneratedTerrain(
            heights_yx=heights_yx,
            horizontal_scale=self.cfg.horizontal_scale,
            z_min=z_min,
            z_max=z_max,
            base_thickness=max(max_base_thickness, 1e-3),
            terrain_origins=self.terrain_origins.copy(),
            box_geoms=all_box_geoms,
            cylinder_geoms=all_cylinder_geoms,
            cell_size=self.cfg.size,
            num_rows=self.cfg.num_rows,
            num_cols=self._num_cols,
        )

    def write_png(self, path: Path) -> GeneratedTerrain:
        terrain = self.generate()
        terrain.write_png(path)
        return terrain

    def _get_sub_terrain_position(self, row: int, col: int) -> np.ndarray:
        """Get the world position for a sub-terrain at the given grid indices.

        This returns the position of the sub-terrain's corner (not center).
        The entire grid is centered at the world origin.
        """
        # Calculate position relative to grid corner.
        rel_x = row * self.cfg.size[0]
        rel_y = col * self.cfg.size[1]

        # Offset to center the entire grid at world origin.
        grid_offset_x = -self.cfg.num_rows * self.cfg.size[0] * 0.5
        grid_offset_y = -self._num_cols * self.cfg.size[1] * 0.5

        return np.array([grid_offset_x + rel_x, grid_offset_y + rel_y, 0.0])

    def _propagate_resolution(self) -> None:
        """Force every sub-terrain config to share the generator's resolution."""
        hs = self.cfg.horizontal_scale
        vs = self.cfg.vertical_scale
        for sub_cfg in self.cfg.sub_terrains.values():
            if hasattr(sub_cfg, "horizontal_scale"):
                setattr(sub_cfg, "horizontal_scale", hs)
            if hasattr(sub_cfg, "vertical_scale"):
                setattr(sub_cfg, "vertical_scale", vs)

    def _validate_resolution(self) -> None:
        """Check that all length-like config values divide evenly by horizontal_scale."""
        hs = self.cfg.horizontal_scale
        if hs <= 0:
            raise ValueError(f"horizontal_scale must be positive, got {hs}.")
        if self.cfg.vertical_scale <= 0:
            raise ValueError(f"vertical_scale must be positive, got {self.cfg.vertical_scale}.")

        def _is_multiple(value: float) -> bool:
            return abs(round(value / hs) * hs - value) <= 1e-9

        for axis, sz in zip(("size[0]", "size[1]"), self.cfg.size):
            if not _is_multiple(sz):
                raise ValueError(
                    f"TerrainGeneratorCfg.{axis}={sz} must be an integer multiple of "
                    f"horizontal_scale={hs}."
                )

        if self.cfg.border_width > 0 and not _is_multiple(self.cfg.border_width):
            raise ValueError(
                f"TerrainGeneratorCfg.border_width={self.cfg.border_width} must be "
                f"an integer multiple of horizontal_scale={hs}."
            )

        for name, sub_cfg in self.cfg.sub_terrains.items():
            for fld in ("step_width", "platform_width", "border_width"):
                if not hasattr(sub_cfg, fld):
                    continue
                value = getattr(sub_cfg, fld)
                if value is None or value == 0:
                    continue
                if not _is_multiple(value):
                    raise ValueError(
                        f"Sub-terrain '{name}' field '{fld}'={value} must be an "
                        f"integer multiple of horizontal_scale={hs}."
                    )