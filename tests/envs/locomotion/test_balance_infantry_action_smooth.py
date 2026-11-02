"""``action_smooth`` 奖励：必须用**真实动作历史**做二阶差分。

回归背景：旧实现把 critic 用的关节 target 历史 (``_action_t2``) 当成
"上上步动作"，量纲不一致：

* 恒定动作 a_t = a 也会被罚 ``(a - 2a + target)²`` ≈ 1.0；
* 梯度等价于让 ``a_t ≈ 2a_{t-1} - target``，权重越大越容易激励二步振荡。

这里锁住三件事：
1. 恒定 / 线性斜坡动作 → 惩罚为 0；
2. 交替动作 → 惩罚为正且等于解析值；
3. 旧的 target 历史即使传进来也不能被使用。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.envs.locomotion.balance_infantry.joystick import BalanceInfantryJoystickEnv


def _penalty(current, last, prev2, **extra) -> float:
    env = object.__new__(BalanceInfantryJoystickEnv)
    info = {
        "current_actions": np.asarray(current, dtype=np.float64).reshape(1, -1),
        "last_actions": np.asarray(last, dtype=np.float64).reshape(1, -1),
        "_prev2_actions": np.asarray(prev2, dtype=np.float64).reshape(1, -1),
    }
    info.update(extra)
    return float(env._reward_action_smooth(SimpleNamespace(info=info))[0])


def test_constant_action_has_zero_penalty() -> None:
    a = np.full(6, 0.3)
    assert _penalty(a, a, a) == pytest.approx(0.0, abs=1e-12)


def test_linear_ramp_has_zero_penalty() -> None:
    # a_t - 2a_{t-1} + a_{t-2} = 0
    assert _penalty(np.full(6, 0.3), np.full(6, 0.2), np.full(6, 0.1)) == pytest.approx(
        0.0, abs=1e-12
    )


def test_alternating_action_is_penalized() -> None:
    # (+1, -1, +1) → current - 2*last + prev2 = 1 - 2*(-1) + 1 = 4
    # → 每条腿 4² = 16，4 条腿 = 64
    current = np.full(6, 1.0)
    last = np.full(6, -1.0)
    prev2 = np.full(6, 1.0)
    assert _penalty(current, last, prev2) == pytest.approx(64.0, rel=1e-12)


def test_legs_only() -> None:
    # 轮子 (4,5) 的抖动不计入
    current = np.zeros(6)
    last = np.zeros(6)
    prev2 = np.zeros(6)
    current[4:] = 1.0
    last[4:] = -1.0
    prev2[4:] = 1.0
    assert _penalty(current, last, prev2) == pytest.approx(0.0, abs=1e-12)


def test_stale_target_buffer_is_ignored() -> None:
    """旧的 `_action_t2`（关节 target）即使还在 info 里也不能影响结果。"""
    a = np.full(6, 0.5)
    assert _penalty(a, a, a, _action_t2=np.full(6, 0.25)) == pytest.approx(0.0, abs=1e-12)
