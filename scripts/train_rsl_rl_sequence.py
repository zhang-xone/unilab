"""Train with RSL-RL Sequence Encoder (ActorCriticSequence)."""

import datetime
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
    get_log_root,
    parse_checkpoint_path,
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


@hydra.main(version_base="1.3", config_path="../conf/ppo_sequence", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
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

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = get_log_root(ROOT_DIR, cfg)
    log_dir = str(
        Path(log_root) / cfg.training.task_name / f"{timestamp}_{cfg.training.sim_backend}"
    )

    tracker = ExperimentTracker(
        root_dir=ROOT_DIR,
        log_dir=log_dir,
        algo_name="ppo_sequence",
        task_name=cfg.training.task_name,
        sim_backend=cfg.training.sim_backend,
        training_cfg=cfg.training,
        full_cfg=cfg,
        device=device,
        seed_info=None,
    )
    tracker.start()

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
            runner.load(str(resume_path))

    runner.learn(num_learning_iterations=max_iterations)


if __name__ == "__main__":
    main()