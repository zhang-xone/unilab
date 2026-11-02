"""按地形分层的命令范围：必须夹紧到真实边界，打印也要用真实边界。

回归背景：

* ``get_terrain_cmdvx_limits`` 原先所有地形都返回 ±3.0，与 ``vel_limit``
  的 vx 上界相等，于是 ``min(课程范围, 地形上限)`` 恒等于课程范围——"按地形
  分层"从未生效；
* ``[info] vx_range`` 原先打印 ``cmd_vel_range``（课程名义范围），看不出每个
  地形真正能采到多少命令；现在打印的是地形夹紧后的 per-env 真实边界。
"""

import numpy as np
import pytest

from unilab.envs.locomotion.balance_infantry import joystick as _flat
from unilab.envs.locomotion.balance_infantry import joystick_vmc as _vmc


@pytest.fixture(params=[_flat, _vmc], ids=["flat", "vmc"])
def J(request):
    """同一组契约测试跑两个 env 模块，防止 joystick_vmc 漂移。"""
    return request.param


def _patch_limits(J, monkeypatch, limits: dict[str, tuple[float, float]]) -> None:
    """注入一张可控的地形 → (low, high) 表，避免依赖真实数值。"""

    def fake(terrain_types):
        lo = np.array([limits[t][0] for t in terrain_types], dtype=np.float64)
        hi = np.array([limits[t][1] for t in terrain_types], dtype=np.float64)
        return lo, hi

    monkeypatch.setattr(J, "get_terrain_cmdvx_limits", fake)


def test_bounds_are_clipped_to_terrain_limits(J, monkeypatch):
    _patch_limits(J, monkeypatch, {"flat": (-1.5, 1.5)})
    lo, hi = J.effective_cmd_vx_bounds(
        np.array([-2.5, -2.5]), np.array([2.5, 2.5]), ["flat", "flat"]
    )
    np.testing.assert_allclose(lo, [-1.5, -1.5])
    np.testing.assert_allclose(hi, [1.5, 1.5])


def test_bounds_never_widen_beyond_curriculum_range(J, monkeypatch):
    """地形上限大于课程范围时不能放大——这正是当前 ±3.0 的退化情形。"""
    _patch_limits(J, monkeypatch, {"flat": (-3.0, 3.0)})
    lo, hi = J.effective_cmd_vx_bounds(np.array([-1.2]), np.array([1.2]), ["flat"])
    np.testing.assert_allclose(lo, [-1.2])
    np.testing.assert_allclose(hi, [1.2])


def test_bounds_are_per_env(J, monkeypatch):
    """不同地形的 env 拿到不同的边界，不能合并成一个全局标量。"""
    _patch_limits(
        J, monkeypatch, {"flat": (-2.0, 2.0), "pyramid_stairs": (-0.6, 1.2)}
    )
    lo, hi = J.effective_cmd_vx_bounds(
        np.array([-2.5, -2.5]),
        np.array([2.5, 2.5]),
        ["flat", "pyramid_stairs"],
    )
    np.testing.assert_allclose(lo, [-2.0, -0.6])
    np.testing.assert_allclose(hi, [2.0, 1.2])
    assert lo[0] != lo[1]
    assert hi[0] != hi[1]


def test_no_terrain_types_returns_curriculum_range(J):
    lo, hi = J.effective_cmd_vx_bounds(
        np.array([-1.0, -2.0]), np.array([1.0, 2.0]), None
    )
    np.testing.assert_allclose(lo, [-1.0, -2.0])
    np.testing.assert_allclose(hi, [1.0, 2.0])


def test_returned_arrays_do_not_alias_curriculum_storage(J):
    """返回的是副本：记录/打印时改写不能污染课程状态。"""
    cur_low = np.array([-1.0])
    cur_high = np.array([1.0])
    lo, hi = J.effective_cmd_vx_bounds(cur_low, cur_high, None)
    lo[0] = -9.0
    hi[0] = 9.0
    np.testing.assert_allclose(cur_low, [-1.0])
    np.testing.assert_allclose(cur_high, [1.0])


@pytest.mark.parametrize(
    "terrain", ["flat", "pyramid_stairs", "pyramid_stairs_inv", "hf_pyramid_slope_inv"]
)
def test_real_table_is_a_hard_cap_not_a_widening(J, terrain):
    """真实表只能收紧、不能放宽：这保证打印值不会大于课程范围。"""
    cur_low = np.array([-2.5, -2.5])
    cur_high = np.array([2.5, 2.5])
    lo, hi = J.effective_cmd_vx_bounds(cur_low, cur_high, [terrain, terrain])
    assert np.all(lo >= cur_low - 1e-9)
    assert np.all(hi <= cur_high + 1e-9)
