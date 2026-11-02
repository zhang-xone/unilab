"""BalanceInfantryJoystickVMC —— 虚拟空间 (L0, θ) 动作空间测试。

覆盖三个风险边界：
1. action → 虚拟参考量 (L0_ref/θ_ref/wheel_ref) 的分组、缩放与限幅；
2. 虚拟空间 PD + 重力前馈 → 五连杆雅可比 → 关节力矩的符号/数值一致性
   （用有限差分验证 τ = Jᵀ[F, T] 的功率一致性）；
3. 真实后端上的 reset / step 冒烟：复位后参考量为标称站姿、力矩有限。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from unilab.base.np_env import NpEnvState
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.balance_infantry.base import (
    ControlConfig,
    VMCConfig,
    WheelTorqueEnvelope,
    WheelTorqueSpeedConfig,
    forward_kinematics_fivebar,
)
from unilab.envs.locomotion.balance_infantry.joystick import (
    BalanceInfantryJoystickEnv,
)
from unilab.envs.locomotion.balance_infantry.joystick_vmc import (
    BalanceInfantryJoystickVMCEnv,
    build_balance_backend_reset_randomization,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _vmc_cfg(**overrides: Any) -> VMCConfig:
    base = dict(
        enabled=True,
        kp_theta=100.0,
        kd_theta=5.0,
        kp_l0=1500.0,
        kd_l0=80.0,
        feedforward_force=80.0,
        action_scale_theta=0.7,
        action_scale_l0=0.15,
        action_scale_wheel=10.0,
        default_L0=0.28,
        theta_clip=0.5,
        L0_clip=(0.15, 0.45),
    )
    base.update(overrides)
    return VMCConfig(**base)


def _fake_env(vmc: VMCConfig, num_envs: int = 1) -> Any:
    env = object.__new__(BalanceInfantryJoystickVMCEnv)
    env._cfg = SimpleNamespace(
        control_config=ControlConfig(
            vmc=vmc,
            clip_actions=100.0,  # 与任务 yaml 一致：轮子动作几乎不限幅
            action_scale_wheel=vmc.action_scale_wheel,
        ),
        debug_enabled=False,
    )
    env._num_envs = num_envs
    env._np_dtype = np.float64
    env._wheel_envelope = WheelTorqueEnvelope(env._cfg.control_config.wheel_torque_speed)
    # 与真实 env 的 _init_control_gains 保持一致：虚拟空间 PD 增益是 env 的属性，
    # 不是每次从 cfg 读，这里必须补上，否则 _compute_vmc_torques 取不到。
    # apply_action 里的动作历史（action_smooth 二阶差分用）
    env._prev_actions = np.zeros((num_envs, 6), dtype=np.float64)
    env._prev2_actions = np.zeros((num_envs, 6), dtype=np.float64)
    env._vmc_kp_theta = np.full(num_envs, vmc.kp_theta, dtype=np.float64)
    env._vmc_kd_theta = np.full(num_envs, vmc.kd_theta, dtype=np.float64)
    env._vmc_kp_l0 = np.full(num_envs, vmc.kp_l0, dtype=np.float64)
    env._vmc_kd_l0 = np.full(num_envs, vmc.kd_l0, dtype=np.float64)
    return env


def _nominal_leg_state(num_envs: int = 1) -> dict[str, np.ndarray]:
    """标称站姿 (φ₁=150°, φ₄=30°) 下的 VMC 状态量。"""
    phi1 = np.full((num_envs, 2), 5.0 * np.pi / 6.0)
    phi4 = np.full((num_envs, 2), np.pi / 6.0)
    L0, phi0, phi2, phi3, sigma1 = forward_kinematics_fivebar(
        phi1, phi4, 0.215, 0.254, 0.254, 0.215, 0.0
    )
    return {
        "phi1": phi1,
        "phi4": phi4,
        "L0": L0,
        "phi0": phi0,
        "phi2": phi2,
        "phi3": phi3,
        "sigma1": sigma1,
    }


# ---------------------------------------------------------------------------
# 1. action → 虚拟参考量
# ---------------------------------------------------------------------------


def test_apply_action_maps_leg_actions_to_virtual_references() -> None:
    vmc = _vmc_cfg()
    env = _fake_env(vmc, num_envs=2)
    env._leg_action_clip = 1.0
    env._action_smoothing = np.ones(2)
    env._prev_filtered_actions = np.zeros((2, 6))
    env._action_t2 = np.zeros((2, 6))
    env._action_t3 = np.zeros((2, 6))
    env._prev_targets_buffer = np.zeros((2, 6))
    env._current_targets = np.zeros((2, 6))
    env._vmc_targets = np.zeros((2, 2, 3))
    state = NpEnvState(
        obs={},
        reward=np.zeros(2),
        terminated=np.zeros(2, dtype=bool),
        truncated=np.zeros(2, dtype=bool),
        info={},
    )

    actions = np.array(
        [
            [0.5, 1.0, -0.5, -1.0, 2.0, -2.0],
            [2.0, -2.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    targets = env.apply_action(actions, state)

    # 左腿/L0: 0.28 + 0.5*0.15；右腿/L0: 0.28 - 0.5*0.15
    np.testing.assert_allclose(targets[:, 0], [0.355, 0.28 + 0.15])
    np.testing.assert_allclose(targets[:, 2], [0.205, 0.28])
    # θ 被 theta_clip=±0.5 限幅
    np.testing.assert_allclose(targets[:, 1], [0.5, -0.5])
    np.testing.assert_allclose(targets[:, 3], [-0.5, 0.0])
    # 轮子 = action * action_scale_wheel
    np.testing.assert_allclose(targets[:, 4], [20.0, 0.0])
    np.testing.assert_allclose(targets[:, 5], [-20.0, 0.0])

    # (leg, slot) 布局：[:, :, 0]=L0_ref, [:, :, 1]=θ_ref, [:, :, 2]=wheel_ref
    np.testing.assert_allclose(env._vmc_targets[:, 0, 0], targets[:, 0])
    np.testing.assert_allclose(env._vmc_targets[:, 1, 0], targets[:, 2])
    np.testing.assert_allclose(env._vmc_targets[:, 0, 1], targets[:, 1])
    np.testing.assert_allclose(env._vmc_targets[:, 1, 1], targets[:, 3])
    np.testing.assert_allclose(env._vmc_targets[:, 0, 2], targets[:, 4])
    np.testing.assert_allclose(env._vmc_targets[:, 1, 2], targets[:, 5])

    clipped_expected = actions.copy()
    clipped_expected[:, :4] = np.clip(clipped_expected[:, :4], -1.0, 1.0)  # leg clip
    clipped_expected[:, 4:6] = np.clip(clipped_expected[:, 4:6], -100.0, 100.0)
    np.testing.assert_array_equal(state.info["current_actions"], clipped_expected)
    np.testing.assert_array_equal(state.info["last_actions"], np.zeros((2, 6)))
    np.testing.assert_array_equal(env._current_targets, targets)


def test_apply_action_clips_l0_reference_to_configured_range() -> None:
    vmc = _vmc_cfg(action_scale_l0=1.0)  # 让 L0_ref 越界，验证 L0_clip 生效
    env = _fake_env(vmc, num_envs=1)
    env._leg_action_clip = 100.0
    env._action_smoothing = np.ones(1)
    env._prev_filtered_actions = np.zeros((1, 6))
    env._action_t2 = np.zeros((1, 6))
    env._action_t3 = np.zeros((1, 6))
    env._prev_targets_buffer = np.zeros((1, 6))
    env._current_targets = np.zeros((1, 6))
    env._vmc_targets = np.zeros((1, 2, 3))
    state = NpEnvState(
        obs={},
        reward=np.zeros(1),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.zeros(1, dtype=bool),
        info={},
    )

    targets = env.apply_action(np.array([[5.0, 0.0, -5.0, 0.0, 0.0, 0.0]]), state)

    assert targets[0, 0] == pytest.approx(vmc.L0_clip[1])
    assert targets[0, 2] == pytest.approx(vmc.L0_clip[0])


# ---------------------------------------------------------------------------
# 2. 虚拟 PD + 前馈 → 雅可比 → 关节力矩
# ---------------------------------------------------------------------------


def test_vmc_torques_match_virtual_pd_plus_feedforward() -> None:
    vmc = _vmc_cfg()
    env = _fake_env(vmc)
    leg = _nominal_leg_state()
    env._vmc_phi1 = leg["phi1"]
    env._vmc_phi4 = leg["phi4"]
    env._vmc_phi0 = leg["phi0"]
    env._vmc_phi2 = leg["phi2"]
    env._vmc_phi3 = leg["phi3"]
    env._vmc_sigma1 = leg["sigma1"]
    env._vmc_L0 = leg["L0"].copy()
    env._vmc_L0_dot = np.zeros((1, 2))
    env._theta_vmc_l = np.zeros(1)
    env._theta_vmc_r = np.zeros(1)
    env._vmc_theta_dot = np.zeros((1, 2))
    env._vmc_ff = np.array([vmc.feedforward_force])
    env._vmc_F = np.zeros((1, 2))
    env._vmc_T = np.zeros((1, 2))
    env._vmc_F_ff = np.zeros((1, 2))

    # ΔL0 = +1cm -> F = kp_l0*0.01 + ff；θ_ref=θ -> Tp = 0
    env._vmc_l0_ref = leg["L0"] + 0.01
    env._vmc_theta_ref = np.zeros((1, 2))

    taus = env._compute_vmc_torques(None)

    expected_f = vmc.kp_l0 * 0.01 + vmc.feedforward_force
    np.testing.assert_allclose(env._vmc_F, expected_f, rtol=1e-12)
    np.testing.assert_allclose(env._vmc_T, 0.0, atol=1e-12)
    np.testing.assert_allclose(env._vmc_F_ff, vmc.feedforward_force, rtol=1e-12)
    assert np.all(np.isfinite(taus))

    # 有限差分验证 τ = Jᵀ[F, Tp]：J00=dL0/dφ1, J01=dφ0/dφ1, J10=dL0/dφ4, J11=dφ0/dφ4
    eps = 1e-6
    phi1 = leg["phi1"][0]
    phi4 = leg["phi4"][0]
    F = expected_f
    T = 0.0

    def _fk(p1: np.ndarray, p4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        L0, phi0, *_ = forward_kinematics_fivebar(
            p1.reshape(1, 2), p4.reshape(1, 2), 0.215, 0.254, 0.254, 0.215, 0.0
        )
        return L0[0], phi0[0]

    d_phi1 = np.full(2, eps)  # 对每条腿的 φ1 同时做前向差分
    d_phi4 = np.full(2, eps)  # 对每条腿的 φ4 同时做前向差分
    L0_p1, phi0_p1 = _fk(phi1 + d_phi1, phi4)
    L0_m1, phi0_m1 = _fk(phi1 - d_phi1, phi4)
    L0_p4, phi0_p4 = _fk(phi1, phi4 + d_phi4)
    L0_m4, phi0_m4 = _fk(phi1, phi4 - d_phi4)

    dL0_dphi1 = (L0_p1 - L0_m1) / (2 * eps)
    dphi0_dphi1 = (phi0_p1 - phi0_m1) / (2 * eps)
    dL0_dphi4 = (L0_p4 - L0_m4) / (2 * eps)
    dphi0_dphi4 = (phi0_p4 - phi0_m4) / (2 * eps)

    expected_tau1 = dL0_dphi1 * F + dphi0_dphi1 * T
    expected_tau4 = dL0_dphi4 * F + dphi0_dphi4 * T
    np.testing.assert_allclose(taus[0, 2], expected_tau1[0], rtol=2e-4, atol=1e-6)  # LR ← τ1
    np.testing.assert_allclose(taus[0, 0], expected_tau4[0], rtol=2e-4, atol=1e-6)  # LF ← τ4
    np.testing.assert_allclose(taus[0, 3], expected_tau1[1], rtol=2e-4, atol=1e-6)  # RR ← τ1
    np.testing.assert_allclose(taus[0, 1], expected_tau4[1], rtol=2e-4, atol=1e-6)  # RF ← τ4


def test_vmc_theta_pd_direction_is_consistent_with_pitch_free_swing() -> None:
    """θ > θ_ref 时 Tp>0（φ0 空间），且 τ 会朝增大 φ0/减小 θ 的方向出力。"""
    vmc = _vmc_cfg(kp_l0=0.0, kd_l0=0.0, feedforward_force=0.0, kd_theta=0.0)
    env = _fake_env(vmc)
    leg = _nominal_leg_state()
    env._vmc_phi1 = leg["phi1"]
    env._vmc_phi4 = leg["phi4"]
    env._vmc_phi0 = leg["phi0"]
    env._vmc_phi2 = leg["phi2"]
    env._vmc_phi3 = leg["phi3"]
    env._vmc_sigma1 = leg["sigma1"]
    env._vmc_L0 = leg["L0"].copy()
    env._vmc_L0_dot = np.zeros((1, 2))
    env._theta_vmc_l = np.array([0.05])
    env._theta_vmc_r = np.array([0.05])
    env._vmc_theta_dot = np.zeros((1, 2))
    env._vmc_ff = np.array([0.0])
    env._vmc_F = np.zeros((1, 2))
    env._vmc_T = np.zeros((1, 2))
    env._vmc_F_ff = np.zeros((1, 2))
    env._vmc_l0_ref = leg["L0"].copy()
    env._vmc_theta_ref = np.zeros((1, 2))

    taus = env._compute_vmc_torques(None)

    np.testing.assert_allclose(env._vmc_F, 0.0, atol=1e-12)
    np.testing.assert_allclose(env._vmc_T, vmc.kp_theta * 0.05, rtol=1e-12)
    # 标称对称构型下 J01 = dφ0/dφ1 > 0，τ1 = J01*Tp > 0（增大 φ0、减小 θ）
    assert np.all(taus[:, [2, 3]] > 0.0)
    assert np.all(taus[:, [0, 1]] > 0.0)


def _fake_env_at_theta(deg_l: float, deg_r: float) -> Any:
    """标称构型、给定摆角 θ 的假 env（只够跑 VMC 力矩计算）。"""
    vmc = _vmc_cfg()
    env = _fake_env(vmc)
    leg = _nominal_leg_state()
    env._vmc_phi1 = leg["phi1"]
    env._vmc_phi4 = leg["phi4"]
    env._vmc_phi0 = leg["phi0"]
    env._vmc_phi2 = leg["phi2"]
    env._vmc_phi3 = leg["phi3"]
    env._vmc_sigma1 = leg["sigma1"]
    env._vmc_L0 = leg["L0"].copy()
    env._vmc_L0_dot = np.zeros((1, 2))
    env._vmc_theta_dot = np.zeros((1, 2))
    env._vmc_ff = np.array([vmc.feedforward_force])
    env._vmc_F = np.zeros((1, 2))
    env._vmc_T = np.zeros((1, 2))
    env._vmc_F_ff = np.zeros((1, 2))
    env._vmc_l0_ref = leg["L0"].copy()
    env._vmc_theta_ref = np.zeros((1, 2))
    env._theta_vmc_l = np.array([np.deg2rad(deg_l)])
    env._theta_vmc_r = np.array([np.deg2rad(deg_r)])
    return env


def test_vmc_normal_path_still_fades_feedforward_beyond_60deg() -> None:
    """正常控制按原样淡出：|θ| > 60° 时前馈为 0（无力期的改动不能碰它）。"""
    env = _fake_env_at_theta(-60.7, -67.2)

    env._compute_vmc_torques(None)

    np.testing.assert_allclose(env._vmc_F_ff, 0.0, atol=1e-12)
    # 同样的姿态下，无力期专用路径**不淡出**，腿仍有伸展力矩
    powerless_taus = env._compute_vmc_feedforward_torques()
    assert np.all(np.abs(powerless_taus[:, :4]) > 1.0)
    assert np.all(powerless_taus[:, 4:] == 0.0)


# ---------------------------------------------------------------------------
# 3. 真实后端冒烟：reset 后的虚拟参考量 / step 后的力矩
# ---------------------------------------------------------------------------


def test_vmc_env_reset_initializes_virtual_refs_and_steps() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available (platform/libstdc++ issue)")

    ensure_registries()
    from unilab.base import registry

    env = registry.make("BalanceInfantryJoystickVMC", num_envs=2, sim_backend="mujoco")
    env_obj: Any = env
    try:
        env_obj.init_state()
        default_l0 = float(env_obj._cfg.control_config.vmc.default_L0)

        # 复位后虚拟参考量回到标称站姿，而不是 joint 版本的全零
        np.testing.assert_allclose(env_obj._vmc_targets[:, :, 0], default_l0)
        np.testing.assert_allclose(env_obj._vmc_targets[:, :, 1], 0.0)
        np.testing.assert_allclose(env_obj._vmc_targets_fifo[:, :, :, 0], default_l0)
        np.testing.assert_allclose(env_obj._vmc_l0_ref, default_l0)

        zeros = np.zeros((env_obj.num_envs, 6))
        for _ in range(3):
            env.step(zeros)

        assert np.all(np.isfinite(env_obj._last_motor_ctrl))
        assert np.all(np.isfinite(env.state.obs["obs"]))
        # 零动作下虚拟参考保持标称；腿部力矩来自 VMC（前馈 > 0）
        np.testing.assert_allclose(env_obj._vmc_targets[:, :, 0], default_l0)
        np.testing.assert_allclose(env_obj._vmc_targets[:, :, 1], 0.0)
        assert np.all(env_obj._vmc_F_ff > 0.0)
        assert np.any(env_obj._last_motor_ctrl[:, :4] != 0.0)
    finally:
        env.close()


_ALL_BALANCE_TASKS = [
    "BalanceInfantryJoystickFlat",
    "BalanceInfantryJoystickVMC",
    "BalanceInfantryJumpFlat",
    "BalanceInfantryJumpvf",
    "BalanceInfantryJumpFlatPretrain",
]


@pytest.mark.parametrize("task", _ALL_BALANCE_TASKS)
def test_powerless_period_keeps_unfaded_feedforward(task: str, monkeypatch: Any) -> None:
    """无力期只保留"不淡出"的重力前馈：腿仍蹬出，轮驱为 0。

    正常控制里 |θ| 越大前馈越淡（60° 归零），但无力期不淡出 —— 用来模拟真实
    机器人断电后内部机械力把腿蹬出去。五个 balance 任务行为必须一致。
    """
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available (platform/libstdc++ issue)")

    ensure_registries()
    from unilab.base import registry

    env: Any = registry.make(task, num_envs=1, sim_backend="mujoco")
    try:
        env._cfg.reward_config.powerless_probability = 1.0  # reset 后必然进入无力期
        monkeypatch.setattr(
            type(env._curriculum),
            "effective_powerless_duration",
            property(lambda self: 99.0),
        )
        env.reset(np.arange(1))
        for _ in range(2):
            env.step(np.zeros((1, env.action_space.shape[-1])))

        assert np.all(env._powerless_active), "无力期未生效，测试前提不成立"
        applied = np.asarray(env._last_motor_ctrl)
        expected = np.asarray(env._compute_vmc_feedforward_torques())
        np.testing.assert_allclose(applied[:, :4], expected[:, :4], atol=1e-4)
        np.testing.assert_allclose(applied[:, 4:6], 0.0, atol=1e-2)  # 只剩执行器噪声
        assert np.all(np.abs(applied[:, :4]) > 1.0)
    finally:
        env.close()


def test_wheel_ground_friction_assigns_same_absolute_value_to_wheels_and_floor() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available (platform/libstdc++ issue)")

    ensure_registries()
    from unilab.base import registry

    env = registry.make("BalanceInfantryJoystickVMC", num_envs=1, sim_backend="mujoco")
    env_obj: Any = env
    try:
        env_obj.init_state()
        dr = env_obj._cfg.domain_rand
        dr.randomize_ground_friction = False
        dr.randomize_wheel_ground_friction = True
        dr.wheel_ground_friction_range = [0.5, 0.5]

        payload = build_balance_backend_reset_randomization(env_obj, 1)
        assert payload is not None and payload.geom_friction is not None
        geom_friction = np.asarray(payload.geom_friction[0])
        for geom_name in ("left_wheel_geom", "right_wheel_geom"):
            geom_id = env_obj._backend.get_geom_id(geom_name)
            assert geom_friction[geom_id, 0] == pytest.approx(0.5)
        ground_id = env_obj._backend.get_geom_id(env_obj._cfg.scene.terrain.geom_name)
        assert geom_friction[ground_id, 0] == pytest.approx(0.5)
    finally:
        env.close()


def test_dr_couples_body_mass_and_inertia_and_uses_per_joint_multipliers() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available (platform/libstdc++ issue)")

    ensure_registries()
    from unilab.base import registry

    env = registry.make("BalanceInfantryJoystickVMC", num_envs=1, sim_backend="mujoco")
    env_obj: Any = env
    try:
        env_obj.init_state()
        num_envs = env_obj.num_envs
        assert env_obj._torque_scale.shape == (num_envs, 6)
        assert env_obj._leg_kp.shape == (num_envs, 4)
        assert env_obj._wheel_kd.shape == (num_envs, 2)

        dr = env_obj._cfg.domain_rand
        dr.randomize_base_mass = True
        dr.added_mass_range = [1.0, 1.0]
        dr.randomize_body_mass = False
        dr.randomize_inertia = True
        dr.inertia_multiplier_range = [0.5, 0.5]

        payload = build_balance_backend_reset_randomization(env_obj, 1)
        assert payload is not None and payload.body_mass is not None and payload.body_inertia is not None
        base_id = env_obj._backend.get_body_ids([env_obj._cfg.asset.base_name])[0]
        base_mass = env_obj._backend.get_body_mass()[base_id]
        base_inertia = env_obj._backend._model.body_inertia[base_id]

        # 同一个 body 的 mass 和 inertia 共用一个 scale；base delta 先加再乘
        assert payload.base_mass_delta is None
        assert payload.body_mass[0, base_id] == pytest.approx((base_mass + 1.0) * 0.5)
        np.testing.assert_allclose(payload.body_inertia[0, base_id], base_inertia * 0.5)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 4. 速度追踪形状：Fudan enhance 形状 + 无接触门控
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_cls",
    [BalanceInfantryJoystickEnv, BalanceInfantryJoystickVMCEnv],
    ids=["joystick", "joystick_vmc"],
)
def test_precise_tracking_is_signed_enhance_and_ignores_contact(env_cls: Any) -> None:
    from unilab.envs.locomotion.common.rewards import RewardContext

    n = 2
    commands = np.zeros((n, 5))
    commands[:, 0] = 1.0
    commands[:, 2] = 0.5
    info = {
        "commands": commands,
        "body_vx": np.array([1.0, 0.7]),  # 误差 0.0 / 0.3 m/s
        "body_vy": np.zeros(n),
        "gravity_body": np.tile(np.array([0.0, 0.0, -1.0]), (n, 1)),
        # 旧实现会因接触把 tracking 归零；新实现必须照常计算
        "collision_stacked": np.ones((n, 3)),
    }
    ctx = RewardContext(
        info=info,
        linvel=np.zeros((n, 3)),
        gyro=np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.2]]),  # 角速度误差 0 / 0.3
        dof_pos=np.zeros((n, 6)),
        num_envs=n,
        tracking_sigma=0.25,
        tracking_sigma_lin=0.25,
        tracking_sigma_ang=0.25,
        tracking_sigma_dynamic=False,
    )

    lin_precise = env_cls._reward_tracking_lin_vel_precise(None, ctx)
    ang_precise = env_cls._reward_tracking_ang_vel_precise(None, ctx)
    lin_base = env_cls._reward_tracking_lin_vel(None, ctx)

    expected = np.exp(-0.3**2 / 0.025) - 1.0
    np.testing.assert_allclose(lin_precise, [0.0, expected], atol=1e-12)
    np.testing.assert_allclose(ang_precise, [0.0, expected], atol=1e-12)
    assert np.all(lin_precise <= 0.0)
    assert np.all(ang_precise <= 0.0)
    # 接触状态下基础项不再归零；静态 sigma 不再对 |cmd|<=1 减半，直接用 0.25
    np.testing.assert_allclose(lin_base, [1.0, np.exp(-0.09 / 0.25)], atol=1e-12)


# ---------------------------------------------------------------------------
# 5. 轮子扭矩-转速（功率）包络
# ---------------------------------------------------------------------------


def test_wheel_torque_speed_envelope() -> None:
    # 解析式：τ_max = min(4, 100/|ω|)，|ω| >= 200 时为 0
    envelope = WheelTorqueEnvelope(
        WheelTorqueSpeedConfig(
            enabled=True, stall_torque=4.0, max_power=100.0, max_speed=200.0
        )
    )
    torque = np.array([[10.0, -10.0], [10.0, -10.0]])
    omega = np.array([[0.0, 25.0], [50.0, 400.0]])
    limited = envelope.limit(torque, omega)
    np.testing.assert_allclose(limited, [[4.0, -4.0], [2.0, 0.0]])

    # 关闭时完全不变
    disabled = WheelTorqueEnvelope(WheelTorqueSpeedConfig(enabled=False))
    np.testing.assert_array_equal(disabled.limit(torque, omega), torque)

    # 实测曲线优先：[(0,4), (30,2), (60,0)]，超过 60 为 0
    curved = WheelTorqueEnvelope(
        WheelTorqueSpeedConfig(
            enabled=True,
            torque_speed_curve=[(0.0, 4.0), (30.0, 2.0), (60.0, 0.0)],
        )
    )
    np.testing.assert_allclose(
        curved.limit(np.array([[10.0, -10.0]]), np.array([[0.0, 15.0]])),
        [[4.0, -3.0]],
    )
    np.testing.assert_allclose(
        curved.limit(np.array([[10.0]]), np.array([[90.0]])), [[0.0]]
    )

    # 速度目标夹紧
    clamp_env = WheelTorqueEnvelope(
        WheelTorqueSpeedConfig(enabled=True, max_wheel_speed=50.0)
    )
    np.testing.assert_allclose(
        clamp_env.clamp_target(np.array([[100.0, -100.0]])), [[50.0, -50.0]]
    )
