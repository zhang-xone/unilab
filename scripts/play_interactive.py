"""MuJoCo interactive play script for trained policies.

This MuJoCo-only viewer tool opens a live MuJoCo window for a task owner config.
``--sim`` selects which owner config to read; playback visualization still uses
MuJoCo.

Usage:
    # Zero-action playback for a task/backend owner config
    uv run scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco

    # Policy playback and keyboard command control
    uv run scripts/play_interactive.py --algo ppo --task go2_joystick_rough --sim mujoco \
      interactive.action_mode=policy interactive.keyboard=true

    # Show target bodies / reward debug overlays
    uv run scripts/play_interactive.py --algo ppo --task g1_motion_tracking --sim mujoco \
      interactive.show_target_bodies=true \
      interactive.target_show_axes=true \
      interactive.show_reward_debug=true

Camera controls (MuJoCo viewer):
    Mouse drag     - rotate
    Scroll         - zoom
    Right-drag     - pan

Interactive keyboard shortcuts (when keyboard teleop enabled):
    Up/Down        - forward/backward (vx)
    Left/Right     - turn left/right (vyaw)
    M              - toggle mouse yaw control (鼠标左右划控制偏航)
    Enter          - full stop (zero all commands)
    Space          - pause/resume
    Backspace      - reset environment
    +/-            - speed up/down simulation

Mouse yaw control:
    When enabled (via ``M`` key or ``interactive.mouse_yaw=true``),
    moving the mouse horizontally controls the yaw-rate command (vyaw).
    Sensitivity is adjustable via ``interactive.mouse_yaw_sensitivity``.
"""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import argparse
import copy
import ctypes
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

try:
    import glfw as _glfw
except ImportError:
    _glfw = None

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.training import (
    ensure_registries,
    get_entrypoint_log_root,
    resolve_task_checkpoint_path,
)
from unilab.training.rsl_rl import (
    RslRlVecEnvWrapper,
    get_policy_obs_dims,
    normalize_ppo_train_cfg,
)
from unilab.visualization.interactive_playback import (
    _HORA_DISTILL_CHECKPOINT_UNAVAILABLE,
    KeyboardCommander,
    PlaybackControls,
    RslRlPlaybackConfig,
    create_appo_playback_session,
    create_hora_distill_playback_session,
    create_rsl_rl_playback_session,
    create_sac_playback_session,
    prepare_motion_overlay_selection,
    select_torch_device,
)

_KEY_ENTER, _KEY_KP_ENTER = 257, 335
_KEY_BACKSPACE = 259
_KEY_RIGHT, _KEY_LEFT, _KEY_DOWN, _KEY_UP = 262, 263, 264, 265
_COMMAND_OBS_VERIFY_COMMAND = np.array([0.37, -0.23, 0.19], dtype=np.float64)
_DEFAULT_CAMERA_DISTANCE = 2.0
_TERRAIN_FOLLOW_CAMERA_DISTANCE = 3.0
_FOLLOW_CAMERA_MAX_DISTANCE = 6.0
_VELOCITY_ARROW_HEIGHT = 0.6
_VELOCITY_ARROW_SCALE = 0.45
_VELOCITY_ARROW_WIDTH = 0.025
_VELOCITY_ARROW_LATERAL_OFFSET = 0.0
_VELOCITY_COMMAND_TASK_NAME_MARKERS = ("Joystick", "Walk")

_DEFAULT_MOUSE_YAW_SENSITIVITY = 0.015  # rad per pixel of mouse movement

ensure_registries()

from unilab.base import registry
from unilab.base.backend.mujoco.playback import resolve_render_play_model_files
from unilab.base.scene import SceneCfg
from unilab.structured_configs import PPOConfig as _StructuredPPOConfig

PPOConfig = _StructuredPPOConfig
_PLAYBACK_ENV_UNAVAILABLE = "playback_env_unavailable"

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)

import mujoco
import mujoco.viewer


@dataclass
class PlayInteractiveArgs:
    task: str
    load_run: str
    checkpoint: str | None
    action_mode: str
    policy_obs_mode: str
    algo_log_name: str
    log_root: str | None
    show_target_bodies: bool
    show_reward_debug: bool
    target_show_axes: bool
    target_body_names: str
    target_max_bodies: int
    target_marker_radius: float
    target_axis_length: float
    target_marker_alpha: float
    reward_debug_show_velocity: bool
    reward_debug_lin_vel_scale: float
    reward_debug_ang_vel_scale: float
    reward_debug_show_connectors: bool
    reward_debug_show_global_anchor: bool
    camera_follow_body: bool
    camera_focus_body_name: str
    camera_height_offset: float
    camera_distance: float | None
    camera_elevation: float | None
    camera_azimuth: float | None
    use_env_visual_model: bool
    speed: float
    start_paused: bool
    keyboard: bool = False
    keyboard_step_lin: float = 0.1
    keyboard_step_ang: float = 0.2
    require_keyboard_command_obs: bool = True
    mouse_yaw: bool = False
    mouse_yaw_sensitivity: float = 0.015
    mouse_yaw_decay: float = 0.85
    algo: str = "ppo"


def _infer_checkpoint_actor_input_dim(ckpt_path: str) -> int | None:
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = loaded.get("actor_state_dict")
    if not isinstance(state_dict, dict):
        return None

    # Common rsl-rl naming: "mlp.0.weight" or nested prefixes ending with ".0.weight".
    for key in ("mlp.0.weight", "actor.mlp.0.weight"):
        w = state_dict.get(key)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[1])

    for key, w in state_dict.items():
        if key.endswith(".0.weight") and isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[1])
    return None


def _backend_adapter(cfg: DictConfig, *, algo_name: str = "ppo"):
    from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
    from unilab.training import BackendAdapter

    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name=algo_name,
        scene_materializer=materialize_scene_visual_override,
    )


def _algo_config_dict(cfg: DictConfig | None) -> dict[str, Any]:
    """Return the composed PPO algo config as a plain dict.

    Args:
        cfg: Hydra config for the current playback run, or ``None`` when the
            script is driven through its legacy non-Hydra path.

    Returns:
        The resolved ``cfg.algo`` subtree as a mutable dict for rsl_rl.
    """
    if cfg is None:
        return cast(dict[str, Any], PPOConfig().to_dict())
    train_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(train_cfg_raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    return cast(dict[str, Any], train_cfg_raw)


SUPPORTED_INTERACTIVE_ALGOS = ("ppo", "ppo_sequence", "appo", "sac", "flashsac", "hora_distill", "ppo_him")
_CONFIG_ROOT_BY_ALGO = {
    "ppo": "ppo",
    "ppo_sequence": "ppo_sequence",
    "appo": "appo",
    "sac": "offpolicy",
    "flashsac": "offpolicy",
    "hora_distill": "hora_distill",
    "ppo_him": "ppo_him",
}
_OFFPOLICY_INTERACTIVE_ALGOS = {"sac", "flashsac"}


@dataclass(frozen=True)
class InteractiveCliArgs:
    algo: str
    task: str
    sim: str
    overrides: list[str]


def _override_key(override: str) -> str:
    key = override.split("=", 1)[0].strip()
    return key.lstrip("+~")


def _parse_interactive_cli(argv: Sequence[str]) -> InteractiveCliArgs:
    parser = argparse.ArgumentParser(
        prog="play_interactive.py",
        description="Open a MuJoCo viewer for an interactive policy playback config.",
    )
    parser.add_argument("--algo", choices=SUPPORTED_INTERACTIVE_ALGOS, default="ppo")
    parser.add_argument("--task", required=True, help="Task name, for example go2_joystick_flat.")
    parser.add_argument("--sim", required=True, help="Owner backend config name to read.")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides.")
    namespace = parser.parse_args(list(argv))

    task = str(namespace.task)
    sim = str(namespace.sim)
    if "/" in task:
        parser.error("--task must be a task name without '/'; pass backend via --sim.")
    if "/" in sim:
        parser.error("--sim must be a backend/config name without '/'.")

    extra_overrides = [str(item) for item in namespace.overrides]
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]
    overrides = _interactive_overrides_from_cli(task, sim, extra_overrides)
    return InteractiveCliArgs(
        algo=str(namespace.algo),
        task=task,
        sim=sim,
        overrides=overrides,
    )


def _interactive_overrides_from_cli(
    task: str, sim: str, extra_overrides: Sequence[str]
) -> list[str]:
    normalized = [f"task={task}/{sim}"]
    for override in extra_overrides:
        key = _override_key(str(override))
        if key in {"task", "training.sim_backend"}:
            raise SystemExit(
                f"{key} is controlled by --task/--sim; use explicit CLI flags instead."
            )
        normalized.append(str(override))
    return normalized


def _normalize_interactive_overrides(algo: str, overrides: list[str]) -> list[str]:
    normalized: list[str] = []
    has_algo_group = False

    for override in overrides:
        key = _override_key(override)
        if algo in _OFFPOLICY_INTERACTIVE_ALGOS and key == "algo":
            value = override.split("=", 1)[1] if "=" in override else ""
            if value != algo:
                raise SystemExit(
                    f"--algo {algo} cannot be combined with a non-{algo} Hydra algo group."
                )
            has_algo_group = True
        if algo in _OFFPOLICY_INTERACTIVE_ALGOS and key == "task" and "=" in override:
            value = override.split("=", 1)[1]
            if not value.startswith(f"{algo}/"):
                override = f"task={algo}/{value}"
        normalized.append(override)

    if algo in _OFFPOLICY_INTERACTIVE_ALGOS and not has_algo_group:
        normalized.insert(0, f"algo={algo}")
    return normalized


def _compose_interactive_config(algo: str, overrides: list[str]) -> DictConfig:
    config_group = _CONFIG_ROOT_BY_ALGO[algo]
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / config_group),
        version_base="1.3",
    ):
        return compose(
            config_name="config",
            overrides=_normalize_interactive_overrides(algo, overrides),
        )


def _select_playback_device(cfg: DictConfig | None) -> str:
    configured = OmegaConf.select(cfg, "training.device") if cfg is not None else None
    if configured not in (None, ""):
        return str(configured)
    return select_torch_device()


# ---------------------------------------------------------------------------
# Checkpoint resolution helpers
# ---------------------------------------------------------------------------


def resolve_checkpoint(
    task: str,
    load_run: str,
    checkpoint: str | None = None,
    algo_log_name: str = "rsl_rl_ppo",
    log_root: str | None = None,
) -> str | None:
    checkpoint_path, checkpoint_dir = resolve_task_checkpoint_path(
        ROOT_DIR,
        task_name=task,
        load_run=load_run,
        algo_log_name=algo_log_name,
        checkpoint=checkpoint,
        log_root=log_root,
    )
    if checkpoint_path is None:
        if checkpoint is not None and checkpoint_dir is not None:
            checkpoint_name = (
                f"model_{checkpoint}.pt" if str(checkpoint).isdigit() else str(checkpoint)
            )
            print(f"[play_interactive] Checkpoint not found: {checkpoint_dir / checkpoint_name}")
        elif checkpoint_dir is not None:
            print(f"[play_interactive] No model_*.pt files in {checkpoint_dir}")
        else:
            print(f"[play_interactive] Run not found for load_run={load_run}")
        return None

    print(f"[play_interactive] Loading checkpoint: {checkpoint_path}")
    return str(checkpoint_path)


# ---------------------------------------------------------------------------
# Interactive play
# ---------------------------------------------------------------------------


def _quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _add_sphere_marker(scene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    scene.ngeom += 1
    return True


def _add_axis_arrow(scene, p0: np.ndarray, p1: np.ndarray, width: float, rgba: np.ndarray) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros((3,), dtype=np.float64),
        np.zeros((3,), dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scene.ngeom += 1
    return True


def _add_vector_arrow(
    scene,
    origin: np.ndarray,
    vector: np.ndarray,
    scale: float,
    width: float,
    rgba: np.ndarray,
    min_len: float = 1e-6,
) -> bool:
    vec = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vec))
    if length < min_len:
        return True
    p0 = np.asarray(origin, dtype=np.float64)
    p1 = p0 + vec * scale
    return _add_axis_arrow(scene, p0, p1, width, rgba)


def _local_xy_to_world_arrow(body_xmat: np.ndarray, local_xy: np.ndarray) -> np.ndarray:
    rot = np.asarray(body_xmat, dtype=np.float64).reshape(3, 3)
    forward = rot[:, 0].copy()
    left = rot[:, 1].copy()
    forward[2] = 0.0
    left[2] = 0.0

    forward_norm = float(np.linalg.norm(forward))
    left_norm = float(np.linalg.norm(left))
    if forward_norm < 1e-9:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        forward /= forward_norm
    if left_norm < 1e-9:
        left = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        left /= left_norm

    xy = np.asarray(local_xy, dtype=np.float64).reshape(2)
    return forward * xy[0] + left * xy[1]


def _resolve_focus_body_id(mj_model, env, preferred_name: str) -> int:
    candidate_names: list[str] = []
    if preferred_name.strip():
        candidate_names.append(preferred_name.strip())

    cfg = getattr(env, "cfg", None)
    asset = getattr(cfg, "asset", None) if cfg is not None else None
    if asset is not None and getattr(asset, "base_name", None):
        candidate_names.append(str(asset.base_name))
    if cfg is not None and getattr(cfg, "base_name", None):
        candidate_names.append(str(cfg.base_name))

    candidate_names.extend(["base", "trunk", "pelvis", "torso", "torso_link"])

    for name in candidate_names:
        try:
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        except Exception:
            body_id = -1
        if body_id >= 0:
            return int(body_id)

    nbody = int(getattr(mj_model, "nbody", 1))
    return 1 if nbody > 1 else 0


def _has_generated_terrain(env: Any) -> bool:
    scene = getattr(getattr(env, "cfg", None), "scene", None)
    return getattr(scene, "terrain", None) is not None


def _default_viewer_camera_distance(mj_model, env: Any, *, follow_body: bool) -> float:
    model_extent = float(getattr(getattr(mj_model, "stat", None), "extent", 1.0))
    extent_distance = max(_DEFAULT_CAMERA_DISTANCE, 2.5 * model_extent)
    if not follow_body:
        return extent_distance
    if _has_generated_terrain(env):
        return _TERRAIN_FOLLOW_CAMERA_DISTANCE
    return min(extent_distance, _FOLLOW_CAMERA_MAX_DISTANCE)


def _available_backends_for_task(task_name: str) -> tuple[str, ...]:
    envs = registry.list_registered_envs()
    task_meta = envs.get(task_name, {})
    backends = task_meta.get("available_backends", ())
    if not isinstance(backends, list):
        return ()
    return tuple(str(backend) for backend in backends)


def _can_launch_glfw_viewer() -> bool:
    try:
        import glfw
    except Exception:
        return True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = bool(glfw.init())
    if ok:
        glfw.terminate()
    return ok


def _uses_native_mujoco_viewer_launch() -> bool:
    launch_fn = getattr(mujoco.viewer, "launch_passive", None)
    module_name = str(getattr(launch_fn, "__module__", ""))
    return module_name.startswith("mujoco")


def _render_motion_targets(
    viewer,
    motion_data,
    selected_indices: np.ndarray,
    marker_radius: float,
    marker_alpha: float,
    show_axes: bool,
    axis_length: float,
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0

    if motion_data is None:
        return

    body_pos = motion_data.body_pos_w[0]
    body_quat = motion_data.body_quat_w[0]

    point_rgba = np.array([0.1, 0.85, 1.0, marker_alpha], dtype=np.float32)
    x_rgba = np.array([1.0, 0.35, 0.35, marker_alpha], dtype=np.float32)
    y_rgba = np.array([0.35, 1.0, 0.35, marker_alpha], dtype=np.float32)
    z_rgba = np.array([0.35, 0.55, 1.0, marker_alpha], dtype=np.float32)

    for idx in selected_indices:
        if idx < 0 or idx >= body_pos.shape[0]:
            continue

        p = body_pos[idx]
        if not _add_sphere_marker(scene, p, marker_radius, point_rgba):
            break

        if show_axes:
            rot = _quat_to_rotmat_wxyz(body_quat[idx])
            px = p + rot[:, 0] * axis_length
            py = p + rot[:, 1] * axis_length
            pz = p + rot[:, 2] * axis_length
            if not _add_axis_arrow(scene, p, px, marker_radius * 0.4, x_rgba):
                break
            if not _add_axis_arrow(scene, p, py, marker_radius * 0.4, y_rgba):
                break
            if not _add_axis_arrow(scene, p, pz, marker_radius * 0.4, z_rgba):
                break


def _render_reward_debug_targets(
    viewer,
    info: dict,
    selected_indices: np.ndarray,
    marker_radius: float,
    marker_alpha: float,
    show_axes: bool,
    axis_length: float,
    show_vel: bool,
    lin_vel_scale: float,
    ang_vel_scale: float,
    show_connectors: bool,
    show_global_anchor: bool,
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0

    if not info:
        return

    motion_data = info.get("motion_data", None)
    robot_body_pos_w = info.get("robot_body_pos_w", None)
    robot_body_quat_w = info.get("robot_body_quat_w", None)
    robot_body_lin_vel_w = info.get("robot_body_lin_vel_w", None)
    robot_body_ang_vel_w = info.get("robot_body_ang_vel_w", None)
    ref_body_pos_w = info.get("reward_ref_body_pos_w", None)
    ref_body_quat_w = info.get("reward_ref_body_quat_w", None)

    if motion_data is None or robot_body_pos_w is None or robot_body_quat_w is None:
        return
    if ref_body_pos_w is None or ref_body_quat_w is None:
        return

    robot_pos = robot_body_pos_w[0]
    robot_quat = robot_body_quat_w[0]
    ref_pos = ref_body_pos_w[0]
    ref_quat = ref_body_quat_w[0]
    motion_pos = motion_data.body_pos_w[0]
    motion_quat = motion_data.body_quat_w[0]
    motion_lin_vel = motion_data.body_lin_vel_w[0]
    motion_ang_vel = motion_data.body_ang_vel_w[0]
    robot_lin_vel = robot_body_lin_vel_w[0] if robot_body_lin_vel_w is not None else None
    robot_ang_vel = robot_body_ang_vel_w[0] if robot_body_ang_vel_w is not None else None

    ref_rgba = np.array([0.15, 0.95, 0.95, marker_alpha], dtype=np.float32)
    robot_rgba = np.array([1.0, 0.62, 0.15, marker_alpha], dtype=np.float32)
    motion_rgba = np.array([0.55, 0.55, 1.0, marker_alpha * 0.7], dtype=np.float32)
    connector_rgba = np.array([1.0, 1.0, 1.0, marker_alpha * 0.55], dtype=np.float32)

    ref_lin_vel_rgba = np.array([0.0, 1.0, 1.0, marker_alpha], dtype=np.float32)
    robot_lin_vel_rgba = np.array([1.0, 0.95, 0.1, marker_alpha], dtype=np.float32)
    ref_ang_vel_rgba = np.array([0.45, 0.75, 1.0, marker_alpha], dtype=np.float32)
    robot_ang_vel_rgba = np.array([1.0, 0.45, 0.1, marker_alpha], dtype=np.float32)

    x_rgba = np.array([1.0, 0.35, 0.35, marker_alpha], dtype=np.float32)
    y_rgba = np.array([0.35, 1.0, 0.35, marker_alpha], dtype=np.float32)
    z_rgba = np.array([0.35, 0.55, 1.0, marker_alpha], dtype=np.float32)

    for idx in selected_indices:
        if idx < 0 or idx >= robot_pos.shape[0]:
            continue

        p_ref = ref_pos[idx]
        p_robot = robot_pos[idx]
        p_motion = motion_pos[idx]

        if not _add_sphere_marker(scene, p_ref, marker_radius, ref_rgba):
            break
        if not _add_sphere_marker(scene, p_robot, marker_radius, robot_rgba):
            break
        if not _add_sphere_marker(scene, p_motion, marker_radius * 0.85, motion_rgba):
            break

        if show_connectors:
            if not _add_axis_arrow(scene, p_ref, p_robot, marker_radius * 0.2, connector_rgba):
                break

        if show_axes:
            for p, q in (
                (p_ref, ref_quat[idx]),
                (p_robot, robot_quat[idx]),
                (p_motion, motion_quat[idx]),
            ):
                rot = _quat_to_rotmat_wxyz(q)
                px = p + rot[:, 0] * axis_length
                py = p + rot[:, 1] * axis_length
                pz = p + rot[:, 2] * axis_length
                if not _add_axis_arrow(scene, p, px, marker_radius * 0.35, x_rgba):
                    break
                if not _add_axis_arrow(scene, p, py, marker_radius * 0.35, y_rgba):
                    break
                if not _add_axis_arrow(scene, p, pz, marker_radius * 0.35, z_rgba):
                    break

        if show_vel:
            if not _add_vector_arrow(
                scene,
                p_motion,
                motion_lin_vel[idx],
                lin_vel_scale,
                marker_radius * 0.28,
                ref_lin_vel_rgba,
            ):
                break
            if robot_lin_vel is not None and not _add_vector_arrow(
                scene,
                p_robot,
                robot_lin_vel[idx],
                lin_vel_scale,
                marker_radius * 0.28,
                robot_lin_vel_rgba,
            ):
                break
            if not _add_vector_arrow(
                scene,
                p_motion,
                motion_ang_vel[idx],
                ang_vel_scale,
                marker_radius * 0.24,
                ref_ang_vel_rgba,
            ):
                break
            if robot_ang_vel is not None and not _add_vector_arrow(
                scene,
                p_robot,
                robot_ang_vel[idx],
                ang_vel_scale,
                marker_radius * 0.24,
                robot_ang_vel_rgba,
            ):
                break

    if show_global_anchor:
        anchor_idx = int(info.get("anchor_body_idx", 0))
        if 0 <= anchor_idx < robot_pos.shape[0]:
            anchor_motion_p = motion_pos[anchor_idx]
            anchor_motion_q = motion_quat[anchor_idx]
            anchor_robot_p = robot_pos[anchor_idx]
            anchor_robot_q = robot_quat[anchor_idx]

            anchor_motion_rgba = np.array([0.95, 0.2, 0.95, marker_alpha], dtype=np.float32)
            anchor_robot_rgba = np.array([1.0, 0.2, 0.2, marker_alpha], dtype=np.float32)
            anchor_radius = marker_radius * 1.35

            _add_sphere_marker(scene, anchor_motion_p, anchor_radius, anchor_motion_rgba)
            _add_sphere_marker(scene, anchor_robot_p, anchor_radius, anchor_robot_rgba)
            _add_axis_arrow(
                scene,
                anchor_motion_p,
                anchor_robot_p,
                marker_radius * 0.3,
                connector_rgba,
            )

            if show_axes:
                for p, q in ((anchor_motion_p, anchor_motion_q), (anchor_robot_p, anchor_robot_q)):
                    rot = _quat_to_rotmat_wxyz(q)
                    px = p + rot[:, 0] * (axis_length * 1.25)
                    py = p + rot[:, 1] * (axis_length * 1.25)
                    pz = p + rot[:, 2] * (axis_length * 1.25)
                    _add_axis_arrow(scene, p, px, marker_radius * 0.45, x_rgba)
                    _add_axis_arrow(scene, p, py, marker_radius * 0.45, y_rgba)
                    _add_axis_arrow(scene, p, pz, marker_radius * 0.45, z_rgba)


def _render_velocity_arrows(
    viewer,
    viz_data,
    focus_body_id: int,
    env: Any,
    *,
    height: float,
    scale: float,
    width: float,
    lateral_offset: float,
) -> None:
    state = getattr(env, "state", None)
    info = getattr(state, "info", None) if state is not None else None
    commands = info.get("commands") if isinstance(info, dict) else None
    if not isinstance(commands, np.ndarray) or commands.ndim != 2 or commands.shape[1] < 3:
        return

    try:
        local_linvel = env.get_local_linvel()
    except AttributeError:
        return
    if (
        not isinstance(local_linvel, np.ndarray)
        or local_linvel.ndim != 2
        or local_linvel.shape[1] < 2
    ):
        return

    body_xmat = np.asarray(viz_data.xmat[focus_body_id], dtype=np.float64)
    origin = np.asarray(viz_data.xpos[focus_body_id], dtype=np.float64).copy()
    origin[2] += float(height)
    side = _local_xy_to_world_arrow(body_xmat, np.array([0.0, 1.0], dtype=np.float64))

    target_vec = _local_xy_to_world_arrow(body_xmat, commands[0, :2])
    current_vec = _local_xy_to_world_arrow(body_xmat, local_linvel[0, :2])
    target_origin = origin + side * float(lateral_offset)
    current_origin = origin - side * float(lateral_offset)

    target_rgba = np.array([0.1, 0.95, 0.15, 0.9], dtype=np.float32)
    current_rgba = np.array([0.1, 0.45, 1.0, 0.9], dtype=np.float32)
    _add_vector_arrow(
        viewer.user_scn,
        target_origin,
        target_vec,
        scale,
        width,
        target_rgba,
    )
    _add_vector_arrow(
        viewer.user_scn,
        current_origin,
        current_vec,
        scale,
        width,
        current_rgba,
    )


def _load_mujoco_model_file_for_viewer(model_file: str):
    if Path(model_file).suffix.lower() == ".mjb":
        return mujoco.MjModel.from_binary_path(str(model_file))
    return mujoco.MjModel.from_xml_path(str(model_file))


def _load_resolved_visual_viewer_model(env: Any):
    try:
        with tempfile.TemporaryDirectory(prefix="unilab-interactive-viewer-") as tmp_dir:
            model_files = resolve_render_play_model_files(env, num_envs=1, tmp_dir=tmp_dir)
            model_file = model_files[0] if isinstance(model_files, list) else model_files
            print(
                f"[play_interactive] Using resolved visual playback model for viewer: {model_file}"
            )
            return _load_mujoco_model_file_for_viewer(str(model_file))
    except Exception as exc:
        print(
            "[play_interactive] WARNING: failed to resolve visual playback model; "
            f"falling back to visual model ({exc})."
        )
        return None


def _load_viewer_model(env: Any, *, use_env_visual_model: bool):
    import mujoco

    backend = getattr(env, "_backend", None)
    backend_visual_model_file = getattr(backend, "scene_visual_model_file", None)
    if backend_visual_model_file:
        resolved = _load_resolved_visual_viewer_model(env)
        if resolved is not None:
            return resolved
        print(
            f"[play_interactive] Using backend visual model for viewer: {backend_visual_model_file}"
        )
        return mujoco.MjModel.from_xml_path(str(backend_visual_model_file))

    if use_env_visual_model:
        cfg_scene = getattr(getattr(env, "cfg", None), "scene", None)
        if cfg_scene is not None and not isinstance(cfg_scene, SceneCfg):
            raise TypeError("env.cfg.scene must be a SceneCfg")
        model_file = None if cfg_scene is None else cfg_scene.model_file
        if model_file:
            try:
                resolved = _load_resolved_visual_viewer_model(env)
                if resolved is not None:
                    return resolved
                print(f"[play_interactive] Using configured visual model for viewer: {model_file}")
                return mujoco.MjModel.from_xml_path(str(model_file))
            except Exception as exc:
                print(
                    "[play_interactive] WARNING: failed to load configured visual model; "
                    f"falling back to playback model ({exc})."
                )

    try:
        playback_model = env.get_playback_model()
    except NotImplementedError as exc:
        raise AttributeError("Environment does not expose a playback model contract") from exc
    if isinstance(playback_model, str):
        print(f"[play_interactive] Using playback model for viewer: {playback_model}")
        return mujoco.MjModel.from_xml_path(playback_model)
    print("[play_interactive] Using backend playback model for viewer.")
    return playback_model


def _build_playback_config(args, *, num_envs: int = 1) -> RslRlPlaybackConfig:
    return RslRlPlaybackConfig(
        task=str(args.task),
        load_run=str(args.load_run),
        checkpoint=getattr(args, "checkpoint", None),
        action_mode=str(args.action_mode),
        policy_obs_mode=str(args.policy_obs_mode),
        algo_log_name=str(getattr(args, "algo_log_name", "rsl_rl_ppo")),
        log_root=getattr(args, "log_root", None),
        num_envs=num_envs,
        speed=float(getattr(args, "speed", 1.0)),
        start_paused=bool(getattr(args, "start_paused", False)),
    )


def _build_keyboard_commander(env: Any, args) -> KeyboardCommander | None:
    """Set up keyboard velocity teleop, or return None when unsupported/disabled."""
    if not bool(getattr(args, "keyboard", False)):
        return None

    state = getattr(env, "state", None)
    command_arr = state.info.get("commands") if state is not None else None
    cmds_cfg = getattr(getattr(env, "cfg", None), "commands", None)
    if not isinstance(command_arr, np.ndarray) or cmds_cfg is None:
        print("[play_interactive] interactive.keyboard ignored: task has no velocity 'commands'.")
        return None

    cmds_cfg.heading_command = False
    cmds_cfg.resampling_time = 0.0

    height_limit = getattr(cmds_cfg, "height_limit", None)
    commander = KeyboardCommander.from_vel_limit(
        cmds_cfg.vel_limit,
        step_lin=float(getattr(args, "keyboard_step_lin", 0.1)),
        step_ang=float(getattr(args, "keyboard_step_ang", 0.2)),
        height_limit=height_limit,
    )
    env.state.info["commands"][:, :commander.command.shape[0]] = commander.command
    return commander


def _state_has_velocity_commands(env: Any) -> bool:
    state = getattr(env, "state", None)
    info = getattr(state, "info", None) if state is not None else None
    command_arr = info.get("commands") if isinstance(info, dict) else None
    return (
        isinstance(command_arr, np.ndarray)
        and command_arr.ndim == 2
        and command_arr.shape[0] > 0
        and command_arr.shape[1] >= 3
    )


def _is_locomotion_env(env: Any) -> bool:
    return type(env).__module__.startswith("unilab.envs.locomotion")


def _is_velocity_command_locomotion_task(env: Any) -> bool:
    if not _is_locomotion_env(env):
        return False
    cfg = getattr(env, "cfg", None)
    candidate_names = [
        type(env).__name__,
        type(cfg).__name__ if cfg is not None else "",
        type(env).__module__,
        type(cfg).__module__ if cfg is not None else "",
    ]
    return any(
        marker in candidate
        for candidate in candidate_names
        for marker in _VELOCITY_COMMAND_TASK_NAME_MARKERS
    )


def _should_render_velocity_arrows(env: Any, *, reset_fn=None) -> bool:
    if not _is_velocity_command_locomotion_task(env):
        return False
    if not _state_has_velocity_commands(env):
        return False
    if reset_fn is None:
        return _state_policy_obs_contains_command(env)
    return _policy_obs_contains_command(env, reset_fn=reset_fn)


def _row_contains_contiguous_vector(
    row: np.ndarray,
    vector: np.ndarray,
    *,
    atol: float = 1.0e-6,
) -> bool:
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    target = np.asarray(vector, dtype=np.float64).reshape(-1)
    if target.size == 0 or values.size < target.size:
        return False
    for start in range(values.size - target.size + 1):
        if np.allclose(values[start : start + target.size], target, atol=atol, rtol=0.0):
            return True
    return False


def _state_policy_obs_contains_command(env: Any) -> bool:
    if not _state_has_velocity_commands(env):
        return False

    state = env.state
    obs = getattr(state, "obs", None)
    actor_obs = obs.get("obs") if isinstance(obs, dict) else None
    if not isinstance(actor_obs, np.ndarray) or actor_obs.ndim != 2 or actor_obs.shape[0] == 0:
        return False

    command = np.asarray(state.info["commands"][0, :3], dtype=np.float64)
    if np.linalg.norm(command) <= 1.0e-9:
        return False
    return _row_contains_contiguous_vector(actor_obs[0], command)


def _policy_obs_contains_command(env: Any, *, reset_fn) -> bool:
    if _state_policy_obs_contains_command(env):
        return True

    cmds_cfg = getattr(getattr(env, "cfg", None), "commands", None)
    if cmds_cfg is None or not hasattr(cmds_cfg, "vel_limit"):
        return False

    original_vel_limit = cmds_cfg.vel_limit
    probe = _COMMAND_OBS_VERIFY_COMMAND.tolist()
    try:
        cmds_cfg.vel_limit = [probe, probe]
        reset_fn()
        return _state_policy_obs_contains_command(env)
    finally:
        cmds_cfg.vel_limit = original_vel_limit
        reset_fn()


def _handle_command_key(commander: KeyboardCommander, keycode: int) -> None:
    if keycode == _KEY_UP:
        commander.nudge(commander.AXIS_VX, +1.0)
    elif keycode == _KEY_DOWN:
        commander.nudge(commander.AXIS_VX, -1.0)
    elif keycode == _KEY_LEFT:
        commander.nudge(commander.AXIS_VYAW, +1.0)
    elif keycode == _KEY_RIGHT:
        commander.nudge(commander.AXIS_VYAW, -1.0)
    elif keycode in (_KEY_ENTER, _KEY_KP_ENTER):
        commander.zero()
    elif keycode in (ord("R"), ord("r")) and commander.has_height:
        commander.nudge(commander.AXIS_HEIGHT, +1.0)
    elif keycode in (ord("T"), ord("t")) and commander.has_height:
        commander.nudge(commander.AXIS_HEIGHT, -1.0)
    else:
        return
    print(f"[play_interactive] {commander.describe()}")


def _print_keyboard_legend(commander: KeyboardCommander | None, env: Any = None) -> None:
    print("[play_interactive] Keyboard teleop ENABLED (drive style):")
    print("  Up / Down    : forward / backward (vx)")
    print("  Left / Right : turn left / right  (vyaw)")
    if commander is not None and commander.has_height:
        print("  R / T        : height up / down")
    if env is not None:
        cmds = getattr(getattr(env, "state", None), "info", {}).get("commands")
        if isinstance(cmds, np.ndarray) and cmds.shape[1] >= 5:
            print("  H            : trigger jump")
    print("  M            : toggle mouse yaw (鼠标左右划控制偏航)")
    print("  Enter        : full stop")


def _create_sequence_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    env_factory,
    algo_config: dict[str, Any],
    root_dir: str | Path,
    device: str | None,
    checkpoint_resolver,
    entrypoint_log_root,
    log,
):
    from unilab.training.rsl_rl_sequence import UniLabSequenceVecEnv
    from unilab.algos.torch.ppo.rsl_rl.runners import OnPolicyRunner as SequenceRunner
    from unilab.visualization.interactive_playback import RslRlPlaybackSession

    device_name = select_torch_device() if device is None else str(device)
    env = env_factory(int(playback_cfg.num_envs))

    obs_history_length = int(algo_config.get("obs_history_length", 5))
    seq_env = UniLabSequenceVecEnv(env, device=device_name, obs_history_length=obs_history_length)

    policy_obs_mode = "flat"

    checkpoint_path = None
    policy = None
    if playback_cfg.action_mode == "policy":
        checkpoint_path = checkpoint_resolver(
            playback_cfg.task,
            playback_cfg.load_run,
            playback_cfg.checkpoint,
            playback_cfg.algo_log_name,
            playback_cfg.log_root,
        )
        if checkpoint_path is None:
            log("WARNING: no checkpoint found - falling back to zero actions.")
        else:
            log(f"Loading checkpoint: {checkpoint_path}")
            log_dir = str(
                entrypoint_log_root(
                    Path(root_dir),
                    algo_log_name=playback_cfg.algo_log_name,
                    log_root=playback_cfg.log_root,
                )
                / playback_cfg.task
                / "play_temp"
            )
            train_cfg = {
                "runner": dict(algo_config.get("runner", {})),
                "algorithm": dict(algo_config.get("algorithm", {})),
                "policy": dict(algo_config.get("policy", {})),
            }
            train_cfg["runner"]["logger"] = "none"
            runner = SequenceRunner(seq_env, train_cfg, log_dir=log_dir, device=device_name)
            runner.load(checkpoint_path, load_optimizer=False)
            policy = runner.get_inference_policy(device=device_name)

    class _SeqAdapter:
        def __init__(self, seq_env, device):
            self._seq_env = seq_env
            self._device = device
            self._obs_history = None

        def reset(self):
            obs, self._obs_history = self._seq_env.reset()
            return obs, {}

        def step(self, actions):
            obs, priv, rew, dones, infos, self._obs_history = self._seq_env.step(actions)
            return obs, rew, dones, infos

    adapter = _SeqAdapter(seq_env, device_name)

    wrapped_policy = None
    if policy is not None:
        def _policy_wrapper(obs):
            actions, _ = policy(obs, adapter._obs_history)
            return actions
        wrapped_policy = _policy_wrapper

    log(f"Action mode: {playback_cfg.action_mode}")
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=adapter,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        policy=wrapped_policy,
        num_envs=playback_cfg.num_envs,
    )
    return session, policy_obs_mode, checkpoint_path


def play_interactive(args, cfg: DictConfig | None = None, *, algo: str | None = None):
    device = _select_playback_device(cfg)
    print(f"[play_interactive] Device: {device}")
    algo = str(algo or getattr(args, "algo", "ppo"))

    # Always use a single env for interactive view
    available_backends = _available_backends_for_task(args.task)
    if available_backends and "mujoco" not in available_backends:
        print(
            "[play_interactive] Task does not support MuJoCo backend: "
            f"{args.task}. Available backends: {available_backends or ('<none>',)}. "
            "This script only supports MuJoCo viewer mode."
        )
        return

    def _create_env(num_envs: int):
        if cfg is None:
            return registry.make(args.task, num_envs=num_envs, sim_backend="mujoco")
        from unilab.training import create_env

        if algo in _OFFPOLICY_INTERACTIVE_ALGOS:
            from train_offpolicy import build_offpolicy_env_cfg_override

            env_cfg_override = build_offpolicy_env_cfg_override(algo, cfg)
        else:
            env_cfg_override = _backend_adapter(cfg, algo_name=algo).build_task_env_cfg_override()
        try:
            return create_env(
                cfg,
                num_envs=num_envs,
                env_cfg_override=env_cfg_override,
                sim_backend="mujoco",
                task_name=args.task,
            )
        except ValueError as exc:
            if "does not support simulation backend 'mujoco'" in str(exc):
                print(
                    "[play_interactive] Task does not support MuJoCo backend: "
                    f"{args.task}. Available backends: {available_backends or ('<none>',)}. "
                    "This script only supports MuJoCo viewer mode."
                )
                raise RuntimeError(_PLAYBACK_ENV_UNAVAILABLE) from exc
            raise

    try:
        playback_cfg = _build_playback_config(args, num_envs=1)
        if algo == "ppo":
            wrapper_cls = RslRlVecEnvWrapper
            if cfg is not None:
                from unilab.algos.torch.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime

                wrapper_cls = resolve_rsl_rl_ppo_runtime(
                    _algo_config_dict(cfg),
                    default_wrapper_cls=RslRlVecEnvWrapper,
                ).wrapper_cls
            session = create_rsl_rl_playback_session(
                playback_cfg=playback_cfg,
                env_factory=_create_env,
                algo_config=_algo_config_dict(cfg),
                root_dir=ROOT_DIR,
                device=device,
                checkpoint_resolver=resolve_checkpoint,
                checkpoint_input_dim_reader=_infer_checkpoint_actor_input_dim,
                entrypoint_log_root=get_entrypoint_log_root,
                wrapper_cls=wrapper_cls,
                runner_cls=OnPolicyRunner,
                policy_obs_dims_getter=get_policy_obs_dims,
                train_cfg_normalizer=normalize_ppo_train_cfg,
                log=lambda message: print(f"[play_interactive] {message}"),
            )
        elif algo == "ppo_sequence":
            session = _create_sequence_playback_session(
                playback_cfg=playback_cfg,
                env_factory=_create_env,
                algo_config=_algo_config_dict(cfg),
                root_dir=ROOT_DIR,
                device=device,
                checkpoint_resolver=resolve_checkpoint,
                entrypoint_log_root=get_entrypoint_log_root,
                log=lambda message: print(f"[play_interactive] {message}"),
            )
        elif algo == "appo":
            if cfg is None:
                raise ValueError("APPO interactive playback requires a composed Hydra config.")
            session = create_appo_playback_session(
                playback_cfg=playback_cfg,
                cfg=cfg,
                rl_cfg=_algo_config_dict(cfg),
                env_factory=_create_env,
                root_dir=ROOT_DIR,
                device=device,
                wrapper_cls=RslRlVecEnvWrapper,
                log=lambda message: print(f"[play_interactive] {message}"),
            )
        elif algo in _OFFPOLICY_INTERACTIVE_ALGOS:
            if cfg is None:
                raise ValueError(f"{algo} interactive playback requires a composed Hydra config.")
            session = create_sac_playback_session(
                playback_cfg=playback_cfg,
                cfg=cfg,
                env_factory=_create_env,
                root_dir=ROOT_DIR,
                device=device,
                algo_name=algo,
                log=lambda message: print(f"[play_interactive] {message}"),
            )
        elif algo == "hora_distill":
            if cfg is None:
                raise ValueError(
                    "HORA distill interactive playback requires a composed Hydra config."
                )
            session = create_hora_distill_playback_session(
                playback_cfg=playback_cfg,
                cfg=cfg,
                root_dir=ROOT_DIR,
                device=device,
                log=lambda message: print(f"[play_interactive] {message}"),
            )
        elif algo == "ppo_him":
            from unilab.algos.torch.him_ppo.runner import HIMOnPolicyRunner

            def _him_train_cfg_normalizer(train_cfg):
                return dict(copy.deepcopy(train_cfg))

            session = create_rsl_rl_playback_session(
                playback_cfg=playback_cfg,
                env_factory=_create_env,
                algo_config=_algo_config_dict(cfg),
                root_dir=ROOT_DIR,
                device=device,
                checkpoint_resolver=resolve_checkpoint,
                checkpoint_input_dim_reader=_infer_checkpoint_actor_input_dim,
                entrypoint_log_root=get_entrypoint_log_root,
                wrapper_cls=RslRlVecEnvWrapper,
                runner_cls=HIMOnPolicyRunner,
                policy_obs_dims_getter=get_policy_obs_dims,
                train_cfg_normalizer=_him_train_cfg_normalizer,
                log=lambda message: print(f"[play_interactive] {message}"),
            )
        else:
            raise ValueError(f"Unsupported interactive playback algo: {algo}")
    except RuntimeError as exc:
        if str(exc) in {_PLAYBACK_ENV_UNAVAILABLE, _HORA_DISTILL_CHECKPOINT_UNAVAILABLE}:
            return
        raise
    playback_session = session[0]
    env = playback_session.env

    if _uses_native_mujoco_viewer_launch() and not _can_launch_glfw_viewer():
        print(
            "[play_interactive] GLFW viewer initialization failed (no usable display). "
            "Set DISPLAY correctly, or run this command in a desktop session."
        )
        return
    overlay = prepare_motion_overlay_selection(
        env,
        show_target_bodies=bool(args.show_target_bodies),
        show_reward_debug=bool(args.show_reward_debug),
        target_body_names=str(args.target_body_names),
        target_max_bodies=int(args.target_max_bodies),
        log=lambda message: print(f"[play_interactive] {message}"),
    )

    if overlay.enabled:
        print(
            "[play_interactive] Target visualization enabled "
            f"({overlay.selected_indices.size} bodies, axes={args.target_show_axes})."
        )
    if args.show_reward_debug:
        print(
            "[play_interactive] Reward debug overlay enabled "
            f"(vel={args.reward_debug_show_velocity}, connectors={args.reward_debug_show_connectors}, "
            f"global_anchor={args.reward_debug_show_global_anchor})."
        )
    # Dedicated MjData for the viewer (never touches the rollout workers)
    use_env_visual_model = bool(getattr(args, "use_env_visual_model", True))
    mj_model = _load_viewer_model(env, use_env_visual_model=use_env_visual_model)

    viz_data = mujoco.MjData(mj_model)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    ctrl_dt = env.cfg.ctrl_dt

    playback_session.reset()
    render_velocity_arrows = str(args.action_mode) == "policy" and _should_render_velocity_arrows(
        env, reset_fn=playback_session.reset
    )
    if render_velocity_arrows:
        print("[play_interactive] Velocity arrows enabled (green=target, blue=current).")
    if bool(getattr(args, "keyboard", False)) and bool(
        getattr(args, "require_keyboard_command_obs", True)
    ):
        if not _state_has_velocity_commands(env):
            print(
                "[play_interactive] interactive.keyboard unavailable: "
                "task state has no velocity 'commands'."
            )
            return
        if not _policy_obs_contains_command(env, reset_fn=playback_session.reset):
            print(
                "[play_interactive] interactive.keyboard unavailable: "
                "policy obs does not contain the velocity command."
            )
            return
    controls = PlaybackControls(
        paused=bool(getattr(args, "start_paused", False)),
        speed=float(getattr(args, "speed", 1.0)),
    )

    commander = _build_keyboard_commander(env, args)
    if commander is not None:
        env.set_autoreset(False)
    env._play_mode = True

    mouse_yaw_active = bool(getattr(args, "mouse_yaw", False))
    mouse_yaw_sensitivity = float(getattr(args, "mouse_yaw_sensitivity", 0.015))
    mouse_yaw_decay = float(getattr(args, "mouse_yaw_decay", 0.85))
    mouse_last_x = 0.0
    mouse_last_y = 0.0

    limp_mode = False
    _limp_original_pre_step_fn = env._backend._pre_step_control_fn if hasattr(env, "_backend") else None

    def _limp_zero_torque_wrapper(backend, policy_ctrl):
        if _limp_original_pre_step_fn is not None:
            _limp_original_pre_step_fn(backend, policy_ctrl)
        return np.zeros(policy_ctrl.shape, dtype=policy_ctrl.dtype)

    _x11 = None
    _x11_display = None
    _x11_root = None
    _x11_screen_w = 0
    _x11_screen_h = 0
    _xtst = None
    try:
        _x11 = ctypes.CDLL("libX11.so.6")
        _x11.XOpenDisplay.restype = ctypes.c_void_p
        _x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        _x11_display = _x11.XOpenDisplay(None)
        if _x11_display:
            _x11.XDefaultRootWindow.restype = ctypes.c_void_p
            _x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
            _x11_root = _x11.XDefaultRootWindow(_x11_display)
            _x11.XDefaultScreen.restype = ctypes.c_int
            _x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
            _screen = _x11.XDefaultScreen(_x11_display)
            _x11.XDisplayWidth.restype = ctypes.c_int
            _x11.XDisplayWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
            _x11_screen_w = _x11.XDisplayWidth(_x11_display, _screen)
            _x11.XDisplayHeight.restype = ctypes.c_int
            _x11.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
            _x11_screen_h = _x11.XDisplayHeight(_x11_display, _screen)
            _x11.XQueryPointer.restype = ctypes.c_int
            _x11.XQueryPointer.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_uint),
            ]
            _x11.XFlush.restype = ctypes.c_int
            _x11.XFlush.argtypes = [ctypes.c_void_p]
        _xtst = ctypes.CDLL("libXtst.so.6")
        _xtst.XTestFakeMotionEvent.restype = ctypes.c_int
        _xtst.XTestFakeMotionEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong,
        ]
    except Exception:
        _x11_display = None

    def _on_key(keycode: int) -> None:
        nonlocal mouse_yaw_active, mouse_last_x, mouse_last_y, limp_mode
        if keycode == ord(" "):
            paused = controls.toggle_pause()
            status = "paused" if paused else "resumed"
            print(f"[play_interactive] {status} (space)")
        elif keycode in (ord("N"), ord("n")):
            controls.request_single_step()
            if not controls.paused:
                controls.pause()
                print("[play_interactive] paused for single-step mode (n)")
            print("[play_interactive] single step requested (n)")
        elif keycode in (ord("+"), ord("=")):
            controls.set_speed(controls.speed * 1.25)
            print(f"[play_interactive] speed={controls.speed:.2f}x")
        elif keycode in (ord("-"), ord("_")):
            controls.set_speed(controls.speed / 1.25)
            print(f"[play_interactive] speed={controls.speed:.2f}x")
        elif keycode in (ord("M"), ord("m")):
            mouse_yaw_active = not mouse_yaw_active
            mouse_last_x = 0.0
            mouse_last_y = 0.0
            status = "ON (mouse 左右划控制 vyaw)" if mouse_yaw_active else "OFF"
            print(f"[play_interactive] 鼠标偏航控制: {status}")
        elif keycode in (ord("L"), ord("l")):
            limp_mode = not limp_mode
            if limp_mode:
                if hasattr(env, "_backend") and _limp_original_pre_step_fn is not None:
                    env._backend.set_pre_step_control(_limp_zero_torque_wrapper)
                print("[play_interactive] 无力模式: ON (ctrl 直接归零)")
            else:
                if hasattr(env, "_backend") and _limp_original_pre_step_fn is not None:
                    env._backend.set_pre_step_control(_limp_original_pre_step_fn)
                print("[play_interactive] 无力模式: OFF (恢复策略控制)")
        elif commander is not None and keycode == _KEY_BACKSPACE:
            playback_session.reset()
            commander.zero()
            print("[play_interactive] reset (backspace)")
        elif keycode in (ord("H"), ord("h")) and env is not None:
            cmds = env.state.info.get("commands")
            if isinstance(cmds, np.ndarray) and cmds.shape[1] >= 5:
                current_jumph = float(getattr(env, "_jumph", np.zeros(1))[0])
                if current_jumph > 0.0:
                    env._jumph[:] = 0.0
                    if hasattr(env, "_jump_vz_target"):
                        env._jump_vz_target[:] = 0.0
                    cmds[:, 4] = 0.0
                    for attr in (
                        "_jump_in_progress", "_jump_need_tuck", "_jump_phase",
                    ):
                        arr = getattr(env, attr, None)
                        if arr is not None:
                            arr[:] = False if attr != "_jump_phase" else 0
                    for attr in ("_jump_phase_time", "_jump_step_counter",
                                 "_jump_ref_vel_z", "_jump_ref_phase",
                                 "_jump_ref_height", "_jump_ref_release_vel_z",
                                 "_jump_push_max_vel_z",
                                 "_jump_assist_selected"):
                        arr = getattr(env, attr, None)
                        if arr is not None:
                            arr[:] = 0
                    if hasattr(env, "_jump_debug_enter_count"):
                        env._jump_debug_enter_count = 0
                    print("[play_interactive] jump cancelled (H)")
                else:
                    rc = getattr(getattr(env, "_cfg", None), "reward_config", None)
                    play_jumph = float(getattr(rc, "jump_height_play", 0.10))
                    play_jumph = max(play_jumph, 0.02)
                    env._jumph[:] = play_jumph
                    if hasattr(env, "_jump_vz_target"):
                        env._jump_vz_target[:] = np.sqrt(2.0 * 9.81 * play_jumph)
                    cmds[:, 4] = play_jumph
                    print(f"[play_interactive] jump triggered (H) jumph={play_jumph:.2f}")
        elif commander is not None:
            _handle_command_key(commander, keycode)

    print("[play_interactive] Opening viewer — close the window or press Esc to quit.")
    print("[play_interactive] Controls: Space=pause/resume, N=single-step, +/-=speed")
    if commander is not None:
        _print_keyboard_legend(commander, env)

    with mujoco.viewer.launch_passive(mj_model, viz_data, key_callback=_on_key) as viewer:
        focus_body_id = _resolve_focus_body_id(
            mj_model, env, getattr(args, "camera_focus_body_name", "")
        )

        # Initialize camera to a reasonable default and keep lookat on robot base.
        has_cam = hasattr(viewer, "cam")
        if has_cam:
            if getattr(args, "camera_distance", None) is not None:
                viewer.cam.distance = float(args.camera_distance)
            else:
                viewer.cam.distance = _default_viewer_camera_distance(
                    mj_model,
                    env,
                    follow_body=bool(getattr(args, "camera_follow_body", True)),
                )
            if getattr(args, "camera_elevation", None) is not None:
                viewer.cam.elevation = float(args.camera_elevation)
            if getattr(args, "camera_azimuth", None) is not None:
                viewer.cam.azimuth = float(args.camera_azimuth)

        with torch.inference_mode():
            while viewer.is_running():
                t0 = time.perf_counter()

                # Mouse yaw control: read cursor position via X11 and map horizontal
                # movement to vyaw when mouse_yaw_active is enabled.
                if mouse_yaw_active and commander is not None and _x11_display is not None:
                    try:
                        root_ret = ctypes.c_void_p()
                        child_ret = ctypes.c_void_p()
                        rx = ctypes.c_int()
                        ry = ctypes.c_int()
                        wx = ctypes.c_int()
                        wy = ctypes.c_int()
                        mask = ctypes.c_uint()
                        ret = _x11.XQueryPointer(
                            _x11_display, _x11_root,
                            ctypes.byref(root_ret), ctypes.byref(child_ret),
                            ctypes.byref(rx), ctypes.byref(ry),
                            ctypes.byref(wx), ctypes.byref(wy),
                            ctypes.byref(mask),
                        )
                        if ret:
                            mx, my = rx.value, ry.value
                            if mouse_last_x != 0.0 or mouse_last_y != 0.0:
                                dx = mx - mouse_last_x
                                if abs(dx) > 0.5:
                                    new_yaw = commander.command[commander.AXIS_VYAW] - dx * mouse_yaw_sensitivity
                                    commander.command[commander.AXIS_VYAW] = float(
                                        np.clip(new_yaw, commander.low[commander.AXIS_VYAW], commander.high[commander.AXIS_VYAW])
                                    )
                                    mouse_last_x = mx
                                    mouse_last_y = my
                                else:
                                    commander.command[commander.AXIS_VYAW] *= mouse_yaw_decay
                                    cx = _x11_screen_w // 2
                                    cy = _x11_screen_h // 2
                                    if _xtst is not None and (abs(mx - cx) > 2 or abs(my - cy) > 2):
                                        new_x = int(mx + (cx - mx) * (1.0 - mouse_yaw_decay))
                                        new_y = int(my + (cy - my) * (1.0 - mouse_yaw_decay))
                                        _xtst.XTestFakeMotionEvent(_x11_display, -1, new_x, new_y, 0)
                                        _x11.XFlush(_x11_display)
                                        mouse_last_x = new_x
                                        mouse_last_y = new_y
                                    else:
                                        mouse_last_x = mx
                                        mouse_last_y = my
                            else:
                                mouse_last_x = mx
                                mouse_last_y = my
                    except Exception:
                        pass

                # Write the command before stepping so this step's obs follow it.
                if commander is not None and env.state is not None:
                    env.state.info["commands"][:, :commander.command.shape[0]] = commander.command

                playback_session.advance(controls)

                # Auto-clear H-jump command on landing
                _jumph = getattr(env, "_jumph", None)
                if _jumph is not None and _jumph[0] > 0:
                    info = playback_session.info
                    was_in_flight_arr = info.get("was_in_flight")
                    if was_in_flight_arr is not None:
                        was_in_flight = bool(np.asarray(was_in_flight_arr).flat[0])
                        if not hasattr(env, "_play_was_in_flight"):
                            env._play_was_in_flight = False
                        if env._play_was_in_flight and not was_in_flight:
                            _jumph[:] = 0.0
                            cmds_arr = info.get("commands")
                            if cmds_arr is not None and cmds_arr.shape[1] >= 5:
                                cmds_arr[:, 4] = 0.0
                            if hasattr(env, "_jump_in_progress"):
                                env._jump_in_progress[:] = False
                            print("[play_interactive] jump auto-cleared on landing")
                        env._play_was_in_flight = was_in_flight

                # Push env state[0] into viz_data and refresh scene
                phys = playback_session.physics_state()[0].astype(np.float64)
                mujoco.mj_setState(mj_model, viz_data, phys, state_spec)
                mujoco.mj_forward(mj_model, viz_data)

                if has_cam and bool(getattr(args, "camera_follow_body", True)):
                    base_pos = viz_data.xpos[focus_body_id]
                    viewer.cam.lookat[0] = float(base_pos[0])
                    viewer.cam.lookat[1] = float(base_pos[1])
                    viewer.cam.lookat[2] = float(
                        base_pos[2] + float(getattr(args, "camera_height_offset", 0.15))
                    )

                if overlay.enabled:
                    if args.show_reward_debug:
                        _render_reward_debug_targets(
                            viewer,
                            playback_session.info,
                            overlay.selected_indices,
                            marker_radius=args.target_marker_radius,
                            marker_alpha=args.target_marker_alpha,
                            show_axes=args.target_show_axes,
                            axis_length=args.target_axis_length,
                            show_vel=args.reward_debug_show_velocity,
                            lin_vel_scale=args.reward_debug_lin_vel_scale,
                            ang_vel_scale=args.reward_debug_ang_vel_scale,
                            show_connectors=args.reward_debug_show_connectors,
                            show_global_anchor=args.reward_debug_show_global_anchor,
                        )
                    else:
                        motion_data = env.state.info.get("motion_data", None)
                        _render_motion_targets(
                            viewer,
                            motion_data,
                            overlay.selected_indices,
                            marker_radius=args.target_marker_radius,
                            marker_alpha=args.target_marker_alpha,
                            show_axes=args.target_show_axes,
                            axis_length=args.target_axis_length,
                        )
                else:
                    viewer.user_scn.ngeom = 0

                if render_velocity_arrows:
                    _render_velocity_arrows(
                        viewer,
                        viz_data,
                        focus_body_id,
                        env,
                        height=_VELOCITY_ARROW_HEIGHT,
                        scale=_VELOCITY_ARROW_SCALE,
                        width=_VELOCITY_ARROW_WIDTH,
                        lateral_offset=_VELOCITY_ARROW_LATERAL_OFFSET,
                    )

                viewer.sync()

                # Real-time pacing
                elapsed = time.perf_counter() - t0
                target_dt = controls.target_dt(ctrl_dt)
                if target_dt - elapsed > 0:
                    time.sleep(target_dt - elapsed)

    print("[play_interactive] Done.")


def _normalize_checkpoint_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text in {"-1", "None", "null"} else text


def _build_play_args(cfg: DictConfig, *, algo: str = "ppo") -> PlayInteractiveArgs:
    return PlayInteractiveArgs(
        task=str(cfg.training.task_name),
        load_run=str(cfg.algo.load_run),
        checkpoint=_normalize_checkpoint_value(OmegaConf.select(cfg, "algo.checkpoint")),
        action_mode=str(cfg.interactive.action_mode),
        policy_obs_mode=str(cfg.interactive.policy_obs_mode),
        algo_log_name=str(cfg.algo.algo_log_name),
        log_root=(
            str(cfg.training.log_root)
            if OmegaConf.select(cfg, "training.log_root") is not None
            else None
        ),
        show_target_bodies=bool(cfg.interactive.show_target_bodies),
        show_reward_debug=bool(cfg.interactive.show_reward_debug),
        target_show_axes=bool(cfg.interactive.target_show_axes),
        target_body_names=str(cfg.interactive.target_body_names),
        target_max_bodies=int(cfg.interactive.target_max_bodies),
        target_marker_radius=float(cfg.interactive.target_marker_radius),
        target_axis_length=float(cfg.interactive.target_axis_length),
        target_marker_alpha=float(cfg.interactive.target_marker_alpha),
        reward_debug_show_velocity=bool(cfg.interactive.reward_debug_show_velocity),
        reward_debug_lin_vel_scale=float(cfg.interactive.reward_debug_lin_vel_scale),
        reward_debug_ang_vel_scale=float(cfg.interactive.reward_debug_ang_vel_scale),
        reward_debug_show_connectors=bool(cfg.interactive.reward_debug_show_connectors),
        reward_debug_show_global_anchor=bool(cfg.interactive.reward_debug_show_global_anchor),
        camera_follow_body=bool(cfg.interactive.camera_follow_body),
        camera_focus_body_name=str(cfg.interactive.camera_focus_body_name),
        camera_height_offset=float(cfg.interactive.camera_height_offset),
        camera_distance=(
            float(cfg.interactive.camera_distance)
            if OmegaConf.select(cfg, "interactive.camera_distance") is not None
            else None
        ),
        camera_elevation=(
            float(cfg.interactive.camera_elevation)
            if OmegaConf.select(cfg, "interactive.camera_elevation") is not None
            else None
        ),
        camera_azimuth=(
            float(cfg.interactive.camera_azimuth)
            if OmegaConf.select(cfg, "interactive.camera_azimuth") is not None
            else None
        ),
        use_env_visual_model=bool(cfg.interactive.use_env_visual_model),
        speed=float(OmegaConf.select(cfg, "interactive.speed", default=1.0)),
        start_paused=bool(OmegaConf.select(cfg, "interactive.start_paused", default=False)),
        keyboard=bool(OmegaConf.select(cfg, "interactive.keyboard", default=False)),
        keyboard_step_lin=float(
            OmegaConf.select(cfg, "interactive.keyboard_step_lin", default=0.1)
        ),
        keyboard_step_ang=float(
            OmegaConf.select(cfg, "interactive.keyboard_step_ang", default=0.2)
        ),
        require_keyboard_command_obs=bool(
            OmegaConf.select(cfg, "interactive.require_keyboard_command_obs", default=True)
        ),
        mouse_yaw=bool(OmegaConf.select(cfg, "interactive.mouse_yaw", default=False)),
        mouse_yaw_sensitivity=float(
            OmegaConf.select(cfg, "interactive.mouse_yaw_sensitivity", default=0.015)
        ),
        mouse_yaw_decay=float(
            OmegaConf.select(cfg, "interactive.mouse_yaw_decay", default=0.85)
        ),
        algo=algo,
    )


def main(argv: list[str] | None = None) -> None:
    parsed = _parse_interactive_cli(sys.argv[1:] if argv is None else argv)
    cfg = _compose_interactive_config(parsed.algo, parsed.overrides)
    play_interactive(_build_play_args(cfg, algo=parsed.algo), cfg)


if __name__ == "__main__":
    main()