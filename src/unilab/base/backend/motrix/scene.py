from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from unilab.terrains.terrain_generator import TerrainGeneratorCfg

if TYPE_CHECKING:
    from motrixsim import SceneModel
    from motrixsim.msd import Link, World


def _resolve_scene_fragment_path(fragment_file: str, model_file: Path) -> Path:
    path = Path(fragment_file)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return (model_file.parent / path).resolve()


# Motrix's default equality-connect spring (k=1e8, d=20000) crudely mirrors
# MuJoCo's `solref=\"0.0001 1\"`, but Motrix's LTL constraint solver cannot
# handle such extreme stiffness for closed-chain linkages.  We therefore
# read the original MuJoCo XML `solref` / `solimp` attributes and convert
# them to Motrix spring parameters *with a safety cap* on the stiffness.
#
# MuJoCo   solref(time_constant, damping_ratio)  →  Motrix spring(k, d)
#   k ≈ 1 / time_constant²
#   d ≈ 2 · damping_ratio / time_constant
#
# The effective stiffness also depends on the constrained bodies' inertia;
# using unit effective mass gives a conservative upper bound.
_MAX_EQ_CONNECT_K: float = 1e4
_EQ_CONNECT_K_FALLBACK: float = 1e4
_EQ_CONNECT_D_FALLBACK: float = 500.0
_EQ_CONNECT_MIN_DAMPING_RATIO: float = 1.0


def _parse_connect_sol_params(xml_path: str | Path) -> dict:
    """Parse <equality><connect> solref / solimp from a MuJoCo XML file.

    Returns {name: (k, d, (d0, dmax, width, middle, power))}.
    """
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError:
        return {}
    params: dict = {}
    default_solref = (0.01, 1.0)
    default_solimp = (0.99, 0.99, 0.001, 0.5, 2.0)
    for eq_el in tree.iter("equality"):
        for el in eq_el:
            if el.tag != "connect":
                continue
            name = el.get("name")
            if not name:
                continue
            try:
                sr = [float(v) for v in (el.get("solref", "") or "0.01").split()[:2]]
                si = [float(v) for v in (el.get("solimp", "") or "0.99 0.99 0.001 0.5 2").split()[:5]]
            except ValueError:
                sr, si = list(default_solref), list(default_solimp)
            if len(sr) < 2:
                sr.append(default_solref[1])
            if len(si) < 5:
                si = list(default_solimp)
            tau, xi = sr[0], sr[1]
            k = 1.0 / (tau * tau)
            d = 2.0 * xi / tau
            params[name] = (k, d, tuple(si[:5]))
    return params


def _soften_equality_constraint_springs(world, xml_path=None) -> None:
    ecs = getattr(world, "equality_constraint_set", None)
    if ecs is None:
        return
    connect_params = _parse_connect_sol_params(xml_path) if xml_path else {}
    for ec in ecs.connects:
        spring = getattr(ec, "spring", None)
        if spring is None:
            continue
        ss = getattr(spring, "spring", None)
        if ss is None:
            continue
        name = getattr(ec, "name", None)
        if name and name in connect_params:
            k_raw, d_raw, imp_tuple = connect_params[name]
        else:
            k_raw, d_raw = _EQ_CONNECT_K_FALLBACK, _EQ_CONNECT_D_FALLBACK
            imp_tuple = None
        k = min(k_raw, _MAX_EQ_CONNECT_K)
        xi = d_raw / (2.0 * k_raw ** 0.5) if k_raw > 0 else 1.0
        xi = max(xi, _EQ_CONNECT_MIN_DAMPING_RATIO)
        d = 2.0 * xi * k ** 0.5
        ss.k = k
        ss.d = d
        if imp_tuple is not None:
            imp = getattr(spring, "impedance", None)
            if imp is not None:
                imp.d0, imp.dmax, imp.width, imp.middle, imp.power = imp_tuple


def _extract_keyframes(fragment_file: Path) -> list[ET.Element]:
    """Return ``<keyframe>`` child elements declared inside ``fragment_file``."""
    root = ET.parse(fragment_file).getroot()
    return list(root.findall("keyframe"))


def _materialize_robot_with_fragment_keyframes(
    robot_path: Path, fragment_paths: Sequence[Path]
) -> Path:
    """Inject fragment ``<keyframe>`` blocks into a temporary copy of ``robot_path``.

    motrix's ``msd.from_file`` validates ``<keyframe>`` qpos against the loaded
    model. fragment XMLs only carry sensors/contacts (no body), so a fragment
    with its own keyframe fails to parse on its own. Mujoco backend already
    merges fragments into the scene XML before parsing; this helper does the
    equivalent for the keyframe block so motrix can load a robot model that
    owns the keyframe declared in a sibling fragment.

    Returns the original ``robot_path`` when no fragment has a keyframe.
    """
    fragment_keyframes: list[ET.Element] = []
    for fragment_path in fragment_paths:
        fragment_keyframes.extend(_extract_keyframes(fragment_path))
    if not fragment_keyframes:
        return robot_path

    tree = ET.parse(robot_path)
    root = tree.getroot()
    existing = root.find("keyframe")
    if existing is None:
        existing = ET.SubElement(root, "keyframe")
    for keyframe in fragment_keyframes:
        existing.extend(list(keyframe))

    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{robot_path.name}",
        dir=str(robot_path.parent),
        mode="w",
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name)
    return Path(tmp.name)


def _materialize_fragment_without_keyframes(fragment_file: Path) -> Path:
    """Strip ``<keyframe>`` from a fragment XML; return original if no change."""
    tree = ET.parse(fragment_file)
    root = tree.getroot()
    keyframes = root.findall("keyframe")
    if not keyframes:
        return fragment_file
    for keyframe in keyframes:
        root.remove(keyframe)
    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{fragment_file.name}",
        dir=str(fragment_file.parent),
        mode="w",
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name)
    return Path(tmp.name)


def _cleanup_temp_xml(path: Path, original: Path) -> None:
    if path == original:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _attach_motrix_scene_fragment(world: World, fragment_file: Path) -> None:
    import motrixsim.msd as msd

    sanitized = _materialize_fragment_without_keyframes(fragment_file)
    try:
        fragment = msd.from_file(str(sanitized))
    finally:
        _cleanup_temp_xml(sanitized, fragment_file)
    world.attach(fragment)


def _iter_motrix_links(link: Link):
    yield link
    for child in link.children:
        yield from _iter_motrix_links(child)


def _motrix_world_link_names(world: World) -> list[str]:
    names: list[str] = []
    for body in world.hierarchy.bodies:
        for link in _iter_motrix_links(body.link):
            if link.name:
                names.append(link.name)
    return names


def add_motrix_tracking_frame_sensors(world: World, *, base_name: str) -> None:
    """Add Motrix-native frame sensors matching the legacy tracking sensor contract."""
    import motrixsim.msd as msd

    link_names = _motrix_world_link_names(world)
    if base_name not in link_names:
        raise ValueError(f"Base link '{base_name}' not found in Motrix scene")

    existing = {sensor.name for sensor in world.sensors.frame if sensor.name}
    sensor_specs = (
        ("track_pos_b", msd.FrameSensorType.FramePos),
        ("track_quat_b", msd.FrameSensorType.FrameQuat),
        ("track_linvel_b", msd.FrameSensorType.FrameLinVel),
        ("track_angvel_b", msd.FrameSensorType.FrameAngVel),
    )
    ref_frame = msd.FrameSensorRef.object(msd.ObjectType.link(base_name))
    for link_name in link_names:
        object_type = msd.ObjectType.link(link_name)
        for prefix, sensor_type in sensor_specs:
            sensor_name = f"{prefix}_{link_name}"
            if sensor_name in existing:
                continue
            sensor = msd.FrameSensor()
            sensor.name = sensor_name
            sensor.sensor_type = sensor_type
            sensor.object_type = object_type
            sensor.ref_frame = ref_frame
            world.sensors.frame.append(sensor)


def materialize_motrix_scene(
    *,
    model_file: str,
    fragment_files: Sequence[str] = (),
    add_body_sensors: bool = False,
    base_name: str = "base",
) -> SceneModel:
    """Build a MotrixSim model through MSD scene composition."""
    import motrixsim.msd as msd

    model_path = Path(model_file).resolve()
    fragment_paths = [
        _resolve_scene_fragment_path(fragment_file, model_path) for fragment_file in fragment_files
    ]
    robot_path = _materialize_robot_with_fragment_keyframes(model_path, fragment_paths)
    try:
        world = msd.from_file(str(robot_path))
        _soften_equality_constraint_springs(world, xml_path=robot_path)
        for fragment_path in fragment_paths:
            _attach_motrix_scene_fragment(world, fragment_path)
        if add_body_sensors:
            add_motrix_tracking_frame_sensors(world, base_name=base_name)
        return msd.build(world)
    finally:
        _cleanup_temp_xml(robot_path, model_path)


@overload
def materialize_motrix_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: Literal[False] = False,
) -> tuple[SceneModel, np.ndarray]: ...


@overload
def materialize_motrix_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: Literal[True],
) -> tuple[SceneModel, np.ndarray, object]: ...


def materialize_motrix_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: bool = False,
) -> tuple[SceneModel, np.ndarray] | tuple[SceneModel, np.ndarray, object]:
    """Build a MotrixSim model with generated hfield terrain and attached robot."""
    import motrixsim.msd as msd

    from unilab.terrains import TerrainGenerator

    robot_path = Path(model_file).resolve()
    generated = TerrainGenerator(terrain_cfg).generate()

    world = msd.World()
    world.name = "unilab materialized hfield scene"

    hfield = msd.HFieldSource()
    hfield.nrow = int(generated.heights_yx.shape[0])
    hfield.ncol = int(generated.heights_yx.shape[1])
    # MotrixSim's hfield source uses MuJoCo-style X/Y half extents.
    hfield.size = [float(generated.hfield_size[0]), float(generated.hfield_size[1])]
    hfield.height_scale = float(generated.height_extent)
    # MotrixSim buffers use compiled hfield row order: row 0 is the -Y side.
    hfield_data = np.ascontiguousarray(np.flipud(generated.heights_yx).astype(np.float32))
    hfield.source_type = msd.HFieldSourceType.buffer(
        hfield_data.reshape(-1),
        f"{hfield_name}_buffer",
    )
    world.assets.hfields[hfield_name] = hfield

    terrain_geom = msd.Geometry()
    terrain_geom.name = geom_name
    terrain_geom.shape = msd.ShapeType.HField
    terrain_geom.hfield = hfield_name
    terrain_geom.position = np.asarray(generated.geom_pos, dtype=np.float32)
    terrain_geom.collision_mask = msd.CollisionMask.collide_with_all()
    terrain_geom.margin = 0.0
    terrain_geom.physics_material.friction = [1.0, 0.005, 0.0001]
    world.hierarchy.geoms.append(terrain_geom)

    for i, bg in enumerate(generated.box_geoms):
        box_geom = msd.Geometry()
        box_geom.name = bg.name if bg.name else f"box_geom_{i}"
        box_geom.shape = msd.ShapeType.Box
        box_geom.size = [float(bg.size[0]), float(bg.size[1]), float(bg.size[2])]
        box_geom.position = np.asarray(bg.pos, dtype=np.float32)
        box_geom.collision_mask = msd.CollisionMask.collide_with_all()
        friction = bg.friction if bg.friction else (1.0, 0.005, 0.0001)
        box_geom.physics_material.friction = [float(friction[0]), float(friction[1]), float(friction[2])]
        world.hierarchy.geoms.append(box_geom)

    for i, cg in enumerate(generated.cylinder_geoms):
        cyl_geom = msd.Geometry()
        cyl_geom.name = cg.name if cg.name else f"cyl_geom_{i}"
        cyl_geom.shape = msd.ShapeType.Cylinder
        cyl_geom.size = [float(cg.size[0]), float(cg.size[1]), float(cg.size[2])]
        cyl_geom.position = np.asarray(cg.pos, dtype=np.float32)
        cyl_geom.orientation = np.asarray(cg.quat, dtype=np.float32)
        cyl_geom.collision_mask = msd.CollisionMask.collide_with_all()
        friction = cg.friction if cg.friction else (1.0, 0.005, 0.0001)
        cyl_geom.physics_material.friction = [float(friction[0]), float(friction[1]), float(friction[2])]
        world.hierarchy.geoms.append(cyl_geom)

    fragment_paths = [
        _resolve_scene_fragment_path(fragment_file, robot_path) for fragment_file in fragment_files
    ]
    merged_robot_path = _materialize_robot_with_fragment_keyframes(robot_path, fragment_paths)
    try:
        robot_world = msd.from_file(str(merged_robot_path))
        _soften_equality_constraint_springs(robot_world, xml_path=merged_robot_path)
        world.attach(robot_world)
        # TODO(motrixsim): remove this once msd.World.attach carries keyframes.
        world.keyframes.extend(robot_world.keyframes)
    finally:
        _cleanup_temp_xml(merged_robot_path, robot_path)

    for fragment_path in fragment_paths:
        _attach_motrix_scene_fragment(world, fragment_path)
    if add_body_sensors:
        add_motrix_tracking_frame_sensors(world, base_name=base_name)

    if return_surface_sampler:
        return msd.build(world), generated.terrain_origins, generated.surface_sampler()
    return msd.build(world), generated.terrain_origins