"""Procedural terrain generation.

Ported from mjlab (https://github.com/mjlab/mjlab).
The terrain generator builds a grid of difficulty-graded sub-terrains and writes
a merged heightfield PNG at cold path. Backend materializers consume that output
to build backend-specific scene models.
"""

from unilab.terrains.config import (
    ALL_TERRAIN_PRESETS,
    ROUGH_TERRAINS_CFG,
    STAIRS_TERRAINS_CFG,
    box_plateau,
    box_pyramid_stairs,
    box_pyramid_stairs_inv,
    box_wave,
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    plateau,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
    terrain_preset,
    wave_terrain,
)
from unilab.terrains.heightfield_terrains import (
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPlateauTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxWaveTerrainCfg,
    HfFlatTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPlateauTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
)
from unilab.terrains.terrain_generator import (
    BoxGeomSpec,
    CombinedSurfaceSampler,
    FlatPatchSamplingCfg,
    GeneratedTerrain,
    SubTerrainCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
    TerrainHeightField,
    TerrainOutput,
)

__all__ = [
    "ALL_TERRAIN_PRESETS",
    "BoxGeomSpec",
    "BoxInvertedPyramidStairsTerrainCfg",
    "BoxPlateauTerrainCfg",
    "BoxPyramidStairsTerrainCfg",
    "BoxWaveTerrainCfg",
    "CombinedSurfaceSampler",
    "FlatPatchSamplingCfg",
    "GeneratedTerrain",
    "HfFlatTerrainCfg",
    "HfInvertedPyramidStairsTerrainCfg",
    "HfPlateauTerrainCfg",
    "HfPyramidSlopedTerrainCfg",
    "HfPyramidStairsTerrainCfg",
    "HfRandomUniformTerrainCfg",
    "HfWaveTerrainCfg",
    "ROUGH_TERRAINS_CFG",
    "STAIRS_TERRAINS_CFG",
    "SubTerrainCfg",
    "TerrainGenerator",
    "TerrainGeneratorCfg",
    "TerrainHeightField",
    "TerrainOutput",
    "box_plateau",
    "box_pyramid_stairs",
    "box_pyramid_stairs_inv",
    "box_wave",
    "flat",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "plateau",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "random_rough",
    "terrain_preset",
    "wave_terrain",
]