from __future__ import annotations

from dataclasses import dataclass, field

from unilab.terrains.terrain_generator import TerrainGeneratorCfg


@dataclass
class TerrainSceneCfg:
    """Backend-agnostic terrain slot declaration for a scene."""

    generator: TerrainGeneratorCfg | None = None
    hfield_name: str = "terrain_hfield"
    geom_name: str | None = None


@dataclass
class SceneCfg:
    """Scene source and optional cold-path composition configuration."""

    model_file: str
    fragment_files: list[str] = field(default_factory=list)
    terrain: TerrainSceneCfg | None = None
    # 渲染专用场景(可选)。物理用 model_file，看视频用这个，可大幅降低 draw call。
    # 不设置时沿用 model_file，行为与之前一致。
    visual_model_file: str | None = None
