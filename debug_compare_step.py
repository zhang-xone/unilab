"""对比 jump 和 joystick 的首步 obs / reward。

用法:
    python debug_compare_step.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
from omegaconf import OmegaConf
from unilab.base.registry import make, ensure_registries
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.reward import extract_reward_config


def _to_env_override(cfg):
    """从 Hydra cfg 提取 env_cfg_override."""
    override = extract_reward_config(cfg)
    env_section = OmegaConf.to_container(cfg.env, resolve=True)
    if isinstance(env_section, dict):
        override.update(env_section)
    return override


def compare():
    ensure_registries()

    # 加载两份 config
    base = OmegaConf.load(ROOT_DIR / "conf/ppo_sequence/config.yaml")
    OmegaConf.set_struct(base, False)

    jump_yaml = OmegaConf.load(ROOT_DIR / "conf/ppo_sequence/task/balance_infantry_jump_flat/mujoco.yaml")
    joy_yaml = OmegaConf.load(ROOT_DIR / "conf/ppo_sequence/task/balance_infantry_joystick_flat/mujoco.yaml")

    jump_cfg = OmegaConf.merge(base, jump_yaml)
    joy_cfg = OmegaConf.merge(base, joy_yaml)

    jump_override = _to_env_override(jump_cfg)
    joy_override = _to_env_override(joy_cfg)

    # 相同 seed
    np.random.seed(42)

    num_envs = 4
    jump_env = make("BalanceInfantryJumpFlat", num_envs=num_envs, env_cfg_override=jump_override)
    joy_env = make("BalanceInfantryJoystickFlat", num_envs=num_envs, env_cfg_override=joy_override)

    jump_env.init_state()
    joy_env.init_state()

    rng = np.random.default_rng(42)
    action_dim = jump_env.action_space.shape[0]
    actions = rng.uniform(-1, 1, (num_envs, action_dim)).astype(np.float32)
    # 共享相同 action
    actions2 = actions.copy()

    print(f"\n=== Obs Delay Steps ===")
    print(f"Jump:    {jump_env._obs_delay_steps}  max_delay={jump_env._max_delay}")
    print(f"Joystick:{joy_env._obs_delay_steps}  max_delay={joy_env._max_delay}")

    for step in range(5):
        s_j = jump_env.step(actions)
        s_yt = joy_env.step(actions2)

        # 检查 terminated
        t_j = s_j.terminated if hasattr(s_j, 'terminated') else np.zeros(num_envs, dtype=bool)
        t_yt = s_yt.terminated if hasattr(s_yt, 'terminated') else np.zeros(num_envs, dtype=bool)

        print(f"\n{'='*60}")
        print(f"Step {step+1}")
        print(f"{'='*60}")
        print(f"Jump    terminated: {t_j}")
        print(f"Joystick terminated: {t_yt}")

        # obs 对比
        obs_j = s_j.obs["obs"]
        obs_yt = s_yt.obs["obs"]
        print(f"\n--- Actor Obs ---")
        print(f"Jump    sum={obs_j.sum():.4f}, mean={obs_j.mean():.6f}, max={obs_j.max():.6f}, min={obs_j.min():.6f}")
        print(f"Joystick sum={obs_yt.sum():.4f}, mean={obs_yt.mean():.6f}, max={obs_yt.max():.6f}, min={obs_yt.min():.6f}")
        print(f"Diff    max_abs={np.abs(obs_j - obs_yt).max():.6f}, equal={np.allclose(obs_j, obs_yt, atol=1e-5)}")

        # critic obs 对比
        critic_j = s_j.obs.get("critic", None)
        critic_yt = s_yt.obs.get("critic", None)
        if critic_j is not None and critic_yt is not None:
            print(f"\n--- Critic Obs ---")
            print(f"Jump    sum={critic_j.sum():.4f}, mean={critic_j.mean():.6f}, dim={critic_j.shape[1]}")
            print(f"Joystick sum={critic_yt.sum():.4f}, mean={critic_yt.mean():.6f}, dim={critic_yt.shape[1]}")

        # reward 对比
        r_j = s_j.reward
        r_yt = s_yt.reward
        print(f"\n--- Reward ---")
        print(f"Jump    sum={r_j.sum():.4f}, mean={r_j.mean():.6f}, max={r_j.max():.6f}, min={r_j.min():.6f}")
        print(f"Joystick sum={r_yt.sum():.4f}, mean={r_yt.mean():.6f}, max={r_yt.max():.6f}, min={r_yt.min():.6f}")
        print(f"Diff    abs={np.abs(r_j - r_yt).max():.6f}")

        # obs 逐维差异
        diff = np.abs(obs_j - obs_yt)
        max_diff_idx = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
        print(f"\n--- Obs 逐维最大差异 ---")
        print(f"Index {max_diff_idx}: jump={obs_j[max_diff_idx]:.6f}, joystick={obs_yt[max_diff_idx]:.6f}, diff={diff[max_diff_idx]:.6f}")

        # 打印前 20 维
        print(f"\n--- Actor Obs[0, :20] ---")
        print(f"Jump:    {np.array2string(obs_j[0, :20], precision=6, suppress_small=False)}")
        print(f"Joystick:{np.array2string(obs_yt[0, :20], precision=6, suppress_small=False)}")

        actions = rng.uniform(-1, 1, (num_envs, action_dim)).astype(np.float32)
        actions2 = actions.copy()

    jump_env.close()
    joy_env.close()


if __name__ == "__main__":
    compare()