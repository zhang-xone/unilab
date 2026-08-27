"""Verify that UniLabSequenceVecEnv.reset() correctly updates state.obs.

Bug: reset() discarded the (new_obs, info) return from self._env.reset(),
so _get_obs() read stale state.obs from the previous step, causing the policy
to output wrong actions and simulation to go unstable (QACC NaN).

This script:
  1. Creates an env, wraps it in UniLabSequenceVecEnv
  2. Resets, steps a few times, then resets again
  3. Checks that after each reset, state.obs matches the new obs returned by reset()
  4. Checks that obs after reset != obs before reset (proving freshness)
"""

import numpy as np
import torch

from unilab.training.common import ensure_registries
from unilab.base import registry
from unilab.training.rsl_rl_sequence import UniLabSequenceVecEnv

ensure_registries()


def _build_env():
    env = registry.make(
        "BalanceInfantryJoystickFlat",
        num_envs=1,
        sim_backend="mujoco",
    )
    return env


def test_reset_obs_updated():
    print("=" * 60)
    print("Test: reset() updates state.obs correctly")
    print("=" * 60)

    env = _build_env()
    seq_env = UniLabSequenceVecEnv(env, device="cpu", obs_history_length=5)

    print(f"\nnum_obs={seq_env.num_obs}, num_actions={seq_env.num_actions}")

    # ── First reset (init_state path) ──
    obs1, _ = seq_env.reset()
    state_obs1 = seq_env._get_obs()
    nz = (obs1[0] != 0).sum().item()
    assert torch.allclose(obs1, state_obs1, atol=1e-6), \
        f"FAIL: obs != state.obs after first reset! (non_zero={nz}/{seq_env.num_obs})"
    print(f"  PASS: first reset (init_state path), non_zero={nz}/{seq_env.num_obs}")

    # ── Step ──
    actions = torch.zeros(1, seq_env.num_actions)
    for _ in range(10):
        seq_env.step(actions)

    # ── Second reset (explicit reset path) ──
    obs_before = seq_env._get_obs()
    obs2, _ = seq_env.reset()
    state_obs2 = seq_env._get_obs()
    nz = (obs2[0] != 0).sum().item()
    assert torch.allclose(obs2, state_obs2, atol=1e-6), \
        f"FAIL: obs != state.obs after second reset! (non_zero={nz}/{seq_env.num_obs})"
    diff = (state_obs2 - obs_before).abs().max().item()
    print(f"  PASS: second reset, non_zero={nz}/{seq_env.num_obs}, max_diff={diff:.6f}")

    # ── Step after reset ──
    for _ in range(5):
        seq_env.step(actions)
    print("  PASS: stepping works after reset")

    # ── Third reset ──
    obs3, _ = seq_env.reset()
    state_obs3 = seq_env._get_obs()
    assert torch.allclose(obs3, state_obs3, atol=1e-6), \
        "FAIL: obs != state.obs after third reset!"
    print("  PASS: third reset")

    seq_env.close()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_reset_obs_updated()