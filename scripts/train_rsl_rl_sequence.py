"""Train with RSL-RL Sequence Encoder (ActorCriticSequence)."""

import datetime
import os
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.algos.torch.ppo.rsl_rl.runners import OnPolicyRunner
from unilab.training import (
    apply_configured_training_seed,
    create_env,
    ensure_registries,
    get_latest_checkpoint,
    get_latest_run,
    get_log_root,
    log_playback_plan,
    parse_checkpoint_path,
    should_run_playback,
)
from unilab.training.experiment import ExperimentTracker
from unilab.training.rsl_rl_sequence import UniLabSequenceVecEnv
from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
from unilab.training.backend_adapter import BackendAdapter
from unilab.utils.device import get_default_device


def _backend_adapter(cfg: DictConfig) -> BackendAdapter:
    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo",
        scene_materializer=materialize_scene_visual_override,
    )


def _build_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], _backend_adapter(cfg).build_task_env_cfg_override())


def _build_algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    return cast(dict[str, Any], raw)


def _allow_shape_mismatch() -> bool:
    """续训时是否允许丢弃 shape 不一致的权重（会重置 critic）。

    默认不允许：shape 变化通常意味着 env 配置变了（如 terrain_scan），
    静默丢弃会让价值函数从头开始。确实需要时设
    ``UNILAB_ALLOW_SHAPE_MISMATCH=1``。
    """
    return os.environ.get("UNILAB_ALLOW_SHAPE_MISMATCH", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _build_train_cfg(cfg: DictConfig) -> dict[str, Any]:
    algo = _build_algo_config_dict(cfg)

    runner = dict(algo.get("runner", {}))
    runner.setdefault("policy_class_name", "ActorCriticSequence")
    runner.setdefault("algorithm_class_name", "PPO")
    runner.setdefault("num_steps_per_env", algo["num_steps_per_env"])
    runner.setdefault("save_interval", algo["save_interval"])

    algorithm = dict(algo.get("algorithm", {}))

    policy = dict(algo.get("policy", {}))
    policy.setdefault("obs_normalization", bool(algo.get("empirical_normalization", False)))

    return {"runner": runner, "algorithm": algorithm, "policy": policy}


def _format_play_checkpoint_error(
    cfg: DictConfig,
    *,
    task_log_root: Path,
    load_path: Path | None,
    load_path_dir: Path | None,
) -> str:
    selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    checkpoint_hint = (
        f" algo.checkpoint={selected_checkpoint!r}"
        if selected_checkpoint not in (None, "", -1, "-1")
        else ""
    )

    if load_path_dir is not None and load_path is None and checkpoint_hint:
        reason = f"Requested checkpoint was not found under resolved_run={load_path_dir}."
    elif not task_log_root.exists():
        reason = "Task log root does not exist."
    else:
        latest_run = get_latest_run(task_log_root)
        if latest_run is None:
            reason = "No run directories were found under the task log root."
        elif get_latest_checkpoint(latest_run) is None:
            reason = f"Resolved latest run has no model_*.pt checkpoint files: {latest_run}."
        else:
            reason = "Requested run or checkpoint could not be resolved."

    return (
        "Could not resolve a checkpoint for play mode. "
        f"{reason} task={cfg.training.task_name} task_log_root={task_log_root} "
        f"algo.load_run={cfg.algo.load_run!r}{checkpoint_hint}."
        " Use algo.load_run=<run-dir-or-checkpoint-path> "
        "and optionally algo.checkpoint=<iteration-or-filename>."
    )


def _resolve_play_num_steps(cfg: DictConfig) -> int | None:
    play_steps = OmegaConf.select(cfg, "training.play_steps", default=None)
    if play_steps is None:
        return None
    return int(play_steps)


def _build_play_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], _backend_adapter(cfg).build_play_env_cfg_override())


def play_rsl_rl_sequence(cfg: DictConfig, device: str) -> str | None:
    """Play mode for RSL-RL Sequence Encoder."""
    task_log_root = get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name)
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or load_path_dir is None or not load_path.exists():
        print(
            _format_play_checkpoint_error(
                cfg,
                task_log_root=task_log_root,
                load_path=load_path,
                load_path_dir=load_path_dir,
            )
        )
        return None

    print(f"Loading latest model: {load_path}")
    _ckpt_keys = set(torch.load(load_path, map_location="cpu", weights_only=True).keys())
    if "model_state_dict" not in _ckpt_keys and "actor_state_dict" not in _ckpt_keys:
        print(
            f"Checkpoint at {load_path} is not an rsl-rl checkpoint "
            f"(found keys: {_ckpt_keys}). Aborting play."
        )
        return None

    env_cfg_override = _build_play_env_cfg_override(cfg)

    obs_history_length = int(cfg.algo.get("obs_history_length", 5))
    env = create_env(
        cfg,
        num_envs=cfg.training.play_env_num,
        env_cfg_override=env_cfg_override,
    )
    vec_env = UniLabSequenceVecEnv(env, device=device, obs_history_length=obs_history_length)

    train_cfg = _build_train_cfg(cfg)
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"

    runner = cast(
        Any,
        OnPolicyRunner(cast(Any, vec_env), train_cfg, log_dir=None, device=device),
    )
    # 推理只用 actor/encoder，critic 维度不一致可以安全忽略
    runner.load(str(load_path), allow_shape_mismatch=True)
    policy = runner.get_inference_policy(device=device)
    num_steps = _resolve_play_num_steps(cfg)
    output_video = Path(load_path_dir) / "play_video.mp4"
    playback_mode: str | None = None

    _step_state: dict[str, Any] = {}

    def _init_fn():
        obs, obs_history = vec_env.reset()
        _step_state["obs_history"] = obs_history
        return obs

    def _step_fn(obs):
        actions, _ = policy(obs, _step_state["obs_history"])
        new_obs, _priv, _rew, _dones, _infos, obs_history = vec_env.step(actions)
        _step_state["obs_history"] = obs_history
        return new_obs

    play_video_path = None
    def _log_plan(plan) -> None:
        nonlocal playback_mode
        playback_mode = plan.mode
        log_playback_plan(plan)

    try:
        with torch.inference_mode():
            play_video_path = env.run_playback_mode(
                play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
                play_steps=num_steps,
                output_video=output_video,
                render_spacing=float(
                    getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
                ),
                render_offset_mode=str(getattr(env.cfg, "render_offset_mode", "grid")),
                initialize=_init_fn,
                step=_step_fn,
                camera_kwargs={
                    "cam_distance": cfg.training.cam_distance,
                    "cam_elevation": cfg.training.cam_elevation,
                    "cam_azimuth": cfg.training.cam_azimuth,
                    "cam_lookat": getattr(cfg.training, "cam_lookat", None),
                    "cam_tracking": getattr(cfg.training, "cam_tracking", False),
                    "cam_tracking_env_idx": getattr(cfg.training, "cam_tracking_env_idx", 0),
                    "cam_tracking_extra_envs": getattr(cfg.training, "cam_tracking_extra_envs", 2),
                },
                on_plan=_log_plan,
            )
    except Exception as e:
        if cfg.training.sim_backend == "motrix" and "RenderClosedError" in str(type(e).__name__):
            print("Render window closed.")
        else:
            raise
    if playback_mode != "none" and num_steps is not None:
        print("Done.")
    return play_video_path


@hydra.main(version_base="1.3", config_path="../conf/ppo_sequence", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    device = get_default_device()
    print(f"Using device: {device}")

    max_iterations = cfg.algo.max_iterations
    if cfg.training.num_timesteps:
        n_steps_per_iter = cfg.algo.num_steps_per_env * cfg.algo.num_envs
        max_iterations = max(1, int(cfg.training.num_timesteps / n_steps_per_iter))
        print(
            f"Overriding max_iterations to {max_iterations} based on "
            f"num_timesteps {cfg.training.num_timesteps}"
        )

    if not cfg.training.play_only:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root = get_log_root(ROOT_DIR, cfg)
        log_dir = str(
            Path(log_root) / cfg.training.task_name / f"{timestamp}_{cfg.training.sim_backend}"
        )
    else:
        log_dir = None

    tracker = None
    if not cfg.training.play_only and log_dir is not None:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name="ppo_sequence",
            task_name=cfg.training.task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=device,
            seed_info=seed_info,
        )
        tracker.start()

    try:
        play_video_path = None
        if not cfg.training.play_only:
            env_cfg_override = _build_env_cfg_override(cfg)
            env = create_env(
                cfg,
                num_envs=cfg.algo.num_envs,
                env_cfg_override=env_cfg_override,
            )

            obs_history_length = int(cfg.algo.get("obs_history_length", 5))
            vec_env = UniLabSequenceVecEnv(env, device=device, obs_history_length=obs_history_length)

            train_cfg = _build_train_cfg(cfg)
            runner = OnPolicyRunner(vec_env, train_cfg, log_dir=log_dir, device=device)

            if cfg.algo.load_run != "-1":
                resume_path, _ = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
                if resume_path:
                    print(f"Resuming from {resume_path}")
                    runner.load(str(resume_path), allow_shape_mismatch=_allow_shape_mismatch())

            runner.learn(num_learning_iterations=max_iterations)
            env.close()

        if should_run_playback(
            play_only=cfg.training.play_only,
            no_play=cfg.training.no_play,
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
        ):
            play_video_path = play_rsl_rl_sequence(cfg, device)
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()