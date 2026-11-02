"""resume 时 checkpoint 与模型 shape 不一致必须显式失败。

回归背景：旧逻辑对 shape 不一致的 key 直接丢弃并继续（critic 会被重新初始化、
optimizer 也会被跳过），而且只打印一行 warning。env 配置一改（terrain_scan /
观测布局），续训就会在“看似正常”的情况下损坏价值函数。
"""

import pytest
import torch

from unilab.algos.torch.ppo.rsl_rl.runners.on_policy_runner import (
    drop_shape_mismatched_keys,
)


def test_identical_shapes_pass() -> None:
    ckpt = {"actor.0.weight": torch.zeros(128, 32)}
    model = {"actor.0.weight": torch.zeros(128, 32)}
    assert drop_shape_mismatched_keys(ckpt, model) == {}
    assert "actor.0.weight" in ckpt


def test_shape_mismatch_raises_by_default() -> None:
    ckpt = {"critic.0.weight": torch.zeros(256, 92)}
    model = {"critic.0.weight": torch.zeros(256, 89)}
    with pytest.raises(RuntimeError, match=r"critic\.0\.weight"):
        drop_shape_mismatched_keys(ckpt, model)
    # 报错信息里要能看出两边 shape，方便定位是哪个配置变了
    with pytest.raises(RuntimeError, match=r"92[\s\S]*89|89[\s\S]*92"):
        drop_shape_mismatched_keys(ckpt, model)


def test_allow_shape_mismatch_drops_only_that_key() -> None:
    ckpt = {
        "actor.0.weight": torch.zeros(128, 32),
        "critic.0.weight": torch.zeros(256, 92),
    }
    model = {
        "actor.0.weight": torch.zeros(128, 32),
        "critic.0.weight": torch.zeros(256, 89),
    }
    mismatched = drop_shape_mismatched_keys(ckpt, model, allow_shape_mismatch=True)
    assert list(mismatched) == ["critic.0.weight"]
    assert "critic.0.weight" not in ckpt
    assert "actor.0.weight" in ckpt


def test_missing_or_unexpected_keys_are_not_shape_mismatch() -> None:
    ckpt = {"a": torch.zeros(2), "only_in_ckpt": torch.zeros(5)}
    model = {"a": torch.zeros(2), "only_in_model": torch.zeros(7)}
    # 缺失/多余 key 交给 load_state_dict(strict=False) 处理，不算 shape 冲突
    assert drop_shape_mismatched_keys(ckpt, model) == {}
