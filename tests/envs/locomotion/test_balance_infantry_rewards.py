"""BalanceInfantry 奖励注册回归：yaml 里有权重、代码里必须真的有实现。

回归背景：``wheel_power`` 在 yaml 里配了 -0.03，但函数没注册进 ``_reward_fns``，
``run_reward_dispatch`` 会静默跳过（``name not in fns``），奖励从未生效。
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

from unilab.envs.locomotion.common.rewards import run_reward_dispatch  # noqa: E402

try:  # pragma: no cover - 平台相关
    from mujoco.batch_env import BatchEnvPool as _BatchEnvPool  # noqa: F401
except Exception:  # pragma: no cover
    pytest.skip("mujoco.batch_env not available", allow_module_level=True)

from unilab.base.registry import ensure_registries  # noqa: E402


def _wheel_power_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        info={
            "torques": np.array([[0.0, 0.0, 0.0, 0.0, 2.0, 2.0]], dtype=np.float64),
            "contact_filt": np.ones((1, 2), dtype=bool),
            "lw_vel": np.zeros(1, dtype=np.float64),
            "rw_vel": np.zeros(1, dtype=np.float64),
        },
        num_envs=1,
    )


def test_wheel_power_is_registered_and_finite() -> None:
    ensure_registries()
    from unilab.base import registry

    env = registry.make("BalanceInfantryJoystickFlat", sim_backend="mujoco", num_envs=1)
    try:
        assert "wheel_power" in env._reward_fns, "wheel_power 未注册，yaml 权重会被静默忽略"

        ctx = _wheel_power_ctx()
        penalty = env._reward_fns["wheel_power"](ctx)
        assert penalty.shape == (1,)
        assert np.all(np.isfinite(penalty))
        assert np.all(penalty >= 0.0)

        # 走 dispatch：yaml 的 -0.03 必须真的被应用
        reward = run_reward_dispatch(
            scales={"wheel_power": -0.03},
            fns=env._reward_fns,
            ctx=ctx,
            info={},
            enable_log=False,
            ctrl_dt=0.01,
        )
        assert np.all(np.isfinite(reward))
        assert np.all(reward < 0.0), "wheel_power 权重没有生效"
    finally:
        env.close()
