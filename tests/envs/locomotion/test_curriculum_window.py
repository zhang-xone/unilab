"""Per-env command curriculum tests.

Regression: command ranges must be promoted/demoted for each resetting env,
not for a global tracking average over a fixed step window. A single good env
must not lift every env's command range.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.envs.locomotion.common.curriculum import CurriculumManager


def _manager(
    num_envs: int = 8,
    threshold: int = 1,
    threshold_bad: int = 1,
    enabled: bool = True,
    min_cmd_speed: float = 0.0,
) -> CurriculumManager:
    dr = SimpleNamespace(curriculum_enabled=False)
    rc = SimpleNamespace(
        tracking_sigma_lin=0.25,
        tracking_sigma_ang=0.25,
        tracking_sigma_yaw=0.25,
        jump_curriculum_height_threshold=0.03,
        powerless_duration=0.0,
        powerless_curriculum_threshold=0.0,
        powerless_probability=0.0,
    )
    cmds = SimpleNamespace(
        vel_limit=[[-3.0, 0.0, -5.0], [3.0, 0.0, 5.0]],
        cmd_curriculum_enabled=enabled,
        jump_curriculum_enabled=False,
        initial_vel_range=[-1.0, 1.0],
        initial_yaw_range=1.0,
        cmd_curriculum_sustain_threshold=threshold,
        cmd_curriculum_sustain_threshold_bad=threshold_bad,
        cmd_curriculum_min_cmd_speed=min_cmd_speed,
    )
    cfg = SimpleNamespace(commands=cmds, domain_rand=dr, reward_config=rc)
    return CurriculumManager(cfg, num_envs)


def _reset(
    cm: CurriculumManager,
    env_ids: list[int],
    tracking: float | list[float],
    steps: int = 10,
    cmd_speed: float | list[float] = 0.0,
) -> None:
    ids = np.asarray(env_ids, dtype=np.int64)
    n = cm._num_envs
    tracking_arr = np.asarray(tracking, dtype=np.float64)
    if tracking_arr.ndim == 0:
        tracking_arr = np.full(len(ids), float(tracking_arr))
    tracking_sum = np.zeros(n, dtype=np.float64)
    tracking_sum[ids] = tracking_arr * steps
    steps_arr = np.full(n, steps, dtype=np.float64)
    zeros = np.zeros(n, dtype=np.float64)
    # episode 内累计的 |cmd_vx|（fwd 桶），均值 = cmd_speed
    cmd_speed_arr = np.asarray(cmd_speed, dtype=np.float64)
    if cmd_speed_arr.ndim == 0:
        cmd_speed_arr = np.full(len(ids), float(cmd_speed_arr))
    cmd_vel_sum_fwd = np.zeros(n, dtype=np.float64)
    cmd_vel_sum_fwd[ids] = cmd_speed_arr * steps
    cm.on_reset(
        ids,
        zeros,
        cmd_vel_sum_fwd,
        zeros,
        zeros,
        steps_arr,
        zeros,
        zeros,
        zeros,
        tracking_sum,
    )


def test_promotes_only_the_env_that_is_good() -> None:
    cm = _manager(num_envs=4, threshold=1)
    _reset(cm, [0, 1, 2], [0.9, 0.0, 0.5])

    # env 0 passes; env 1 fails but is already at the minimum; env 2 is neutral.
    assert cm.cmd_vel_range_env[0, 1] > 1.0
    assert cm.cmd_vel_range_env[1, 1] == 1.0
    assert cm.cmd_vel_range_env[2, 1] == 1.0
    assert cm.cmd_yaw_range_env[0, 1] > 1.0
    assert cm.cmd_yaw_range_env[1, 1] == 1.0


def test_global_average_does_not_promote_everyone() -> None:
    cm = _manager(num_envs=8, threshold=1)
    # Global mean here is 0.45, which the old global curriculum would not
    # promote, but env 0 itself is above 0.7.
    _reset(cm, [0, 1], [0.9, 0.0])

    assert cm.cmd_vel_range_env[0, 1] > 1.0
    assert np.all(cm.cmd_vel_range_env[2:, 1] == 1.0)


def test_sustain_threshold_is_per_env_episodes() -> None:
    cm = _manager(num_envs=4, threshold=2)
    _reset(cm, [0], 0.9)
    assert cm.cmd_vel_range_env[0, 1] == 1.0

    _reset(cm, [0], 0.9)
    assert cm.cmd_vel_range_env[0, 1] > 1.0

    # A neutral episode resets the per-env counter, so a later good episode
    # alone must not promote.
    cm = _manager(num_envs=4, threshold=2)
    _reset(cm, [0], 0.9)
    _reset(cm, [0], 0.5)
    _reset(cm, [0], 0.9)
    assert cm.cmd_vel_range_env[0, 1] == 1.0


def test_demotes_only_the_bad_env() -> None:
    cm = _manager(num_envs=4, threshold=1, threshold_bad=1)
    _reset(cm, [0, 1], [0.9, 0.9])
    promoted = cm.cmd_vel_range_env[:, 1].copy()
    assert promoted[0] > 1.0 and promoted[1] > 1.0

    _reset(cm, [0], 0.0)
    assert cm.cmd_vel_range_env[0, 1] < promoted[0]
    assert cm.cmd_vel_range_env[1, 1] == promoted[1]


def test_step_alone_does_not_change_command_ranges() -> None:
    cm = _manager(num_envs=4, threshold=1)
    before = cm.cmd_vel_range_env.copy()
    cm.step(10_000_000)
    np.testing.assert_allclose(cm.cmd_vel_range_env, before)


def test_disabled_command_curriculum_uses_full_ranges_and_never_updates() -> None:
    cm = _manager(num_envs=2, enabled=False)
    np.testing.assert_allclose(cm.cmd_vel_range_env, [[-3.0, 3.0], [-3.0, 3.0]])
    np.testing.assert_allclose(cm.cmd_yaw_range_env, [[-5.0, 5.0], [-5.0, 5.0]])

    _reset(cm, [0], 0.9)
    np.testing.assert_allclose(cm.cmd_vel_range_env[0], [-3.0, 3.0])


# ---------------------------------------------------------------------------
# |cmd_vx| 筛选：原地自转 / 站立 episode 不参与升/降速
# ---------------------------------------------------------------------------


def test_zero_vx_episodes_do_not_participate_when_filter_enabled() -> None:
    """vx=0 的 episode 天然 tracking≈1，绝不能推动速度课程。"""
    cm = _manager(num_envs=2, min_cmd_speed=0.5)

    # tracking=0.99（原地自转/站立都能拿到），但整段没有速度指令
    _reset(cm, [0, 1], 0.99, cmd_speed=0.0)

    np.testing.assert_allclose(cm.cmd_vel_range_env[:, 1], 1.0)
    assert cm.pop_cmd_curriculum_stats() == (0, 0)


def test_zero_vx_episode_never_demotes_a_promoted_range() -> None:
    """有指令的 episode 升上去之后，随后的自转 episode 不能把它降回来。"""
    cm = _manager(num_envs=2, min_cmd_speed=0.5)

    _reset(cm, [0], 0.9, cmd_speed=1.2)
    promoted = cm.cmd_vel_range_env[0, 1]
    assert promoted > 1.0

    # 自转 episode：tracking 掉到 0 也不该降速
    _reset(cm, [0], 0.0, cmd_speed=0.05)
    assert cm.cmd_vel_range_env[0, 1] == promoted


def test_velocity_episodes_still_drive_the_curriculum() -> None:
    cm = _manager(num_envs=2, min_cmd_speed=0.5)

    _reset(cm, [0], 0.9, cmd_speed=1.2)
    promoted = cm.cmd_vel_range_env[0, 1]
    assert promoted > 1.0
    assert cm.pop_cmd_curriculum_stats() == (1, 0)

    _reset(cm, [0], 0.0, cmd_speed=1.2)
    assert cm.cmd_vel_range_env[0, 1] < promoted
    assert cm.pop_cmd_curriculum_stats() == (0, 1)


def test_zero_vx_episode_resets_the_streak_counter() -> None:
    cm = _manager(num_envs=2, threshold=2, min_cmd_speed=0.5)

    _reset(cm, [0], 0.9, cmd_speed=1.2)  # streak = 1
    _reset(cm, [0], 0.9, cmd_speed=0.0)  # 不参与，但会把连续计数清零
    _reset(cm, [0], 0.9, cmd_speed=1.2)  # 只回到 streak = 1

    assert cm.cmd_vel_range_env[0, 1] == 1.0


def test_filter_is_opt_in_for_backward_compatibility() -> None:
    """默认 min_cmd_speed=0.0：保持旧行为，自转 episode 仍然参与判定。"""
    cm = _manager(num_envs=2)  # 不传 min_cmd_speed

    _reset(cm, [0], 0.9, cmd_speed=0.0)
    assert cm.cmd_vel_range_env[0, 1] > 1.0

