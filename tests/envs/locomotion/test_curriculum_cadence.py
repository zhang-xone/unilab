"""命令课程：更新节奏按 env reset，且与 num_envs / 全局窗口解耦。"""

from types import SimpleNamespace

import numpy as np

from unilab.envs.locomotion.common.curriculum import CurriculumManager


def _cfg(threshold: int = 2, threshold_bad: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        commands=SimpleNamespace(
            vel_limit=[[-3.0, 0.0, -5.0], [3.0, 0.0, 5.0]],
            cmd_curriculum_enabled=True,
            initial_vel_range=[-1.0, 1.0],
            initial_yaw_range=1.0,
            cmd_curriculum_sustain_threshold=threshold,
            cmd_curriculum_sustain_threshold_bad=threshold_bad,
            jump_curriculum_enabled=False,
        ),
        reward_config=SimpleNamespace(
            tracking_sigma_lin=0.25,
            tracking_sigma_ang=0.25,
            tracking_sigma_yaw=0.25,
            jump_curriculum_height_threshold=0.03,
        ),
        domain_rand=None,
    )


def _reset(cm: CurriculumManager, env_id: int, tracking: float, steps: int = 10) -> None:
    n = cm._num_envs
    ids = np.array([env_id], dtype=np.int64)
    tracking_sum = np.zeros(n, dtype=np.float64)
    tracking_sum[ids] = tracking * steps
    steps_arr = np.full(n, steps, dtype=np.float64)
    zeros = np.zeros(n, dtype=np.float64)
    cm.on_reset(ids, zeros, zeros, zeros, zeros, steps_arr, zeros, zeros, zeros, tracking_sum)


def test_num_envs_does_not_change_update_trigger() -> None:
    small = CurriculumManager(_cfg(threshold=1), 8)
    large = CurriculumManager(_cfg(threshold=1), 2048)
    _reset(small, 0, 0.9)
    _reset(large, 0, 0.9)

    assert small.cmd_vel_range_env[0, 1] > 1.0
    assert large.cmd_vel_range_env[0, 1] > 1.0


def test_promotion_uses_configured_per_env_sustain_threshold() -> None:
    cm = CurriculumManager(_cfg(threshold=3), 8)
    start = float(cm.cmd_vel_range_env[0, 1])

    _reset(cm, 0, 0.9)
    _reset(cm, 0, 0.9)
    assert float(cm.cmd_vel_range_env[0, 1]) == start

    _reset(cm, 0, 0.9)
    assert float(cm.cmd_vel_range_env[0, 1]) > start


def test_demotion_uses_configured_per_env_bad_threshold() -> None:
    cm = CurriculumManager(_cfg(threshold=1, threshold_bad=3), 8)
    _reset(cm, 0, 0.9)
    high = float(cm.cmd_vel_range_env[0, 1])
    assert high > 1.0

    _reset(cm, 0, 0.0)
    _reset(cm, 0, 0.0)
    assert float(cm.cmd_vel_range_env[0, 1]) == high

    _reset(cm, 0, 0.0)
    assert float(cm.cmd_vel_range_env[0, 1]) < high
