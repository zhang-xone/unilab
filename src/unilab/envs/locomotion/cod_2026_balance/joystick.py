from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import mujoco

from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_quat_apply, np_yaw_to_quat
from unilab.dr import DomainRandomizationCapabilities, ResetPlan, IntervalRandomizationPlan
from unilab.dr.dr_utils import build_interval_push_plan, zero_actions

from unilab.envs.locomotion.common.commands import (
    Commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common import rewards as common_rewards
from unilab.envs.locomotion.common.rewards import (
    RewardContext,
    run_reward_dispatch,
)

from unilab.envs.locomotion.cod_2026_balance.base import (
    NUM_COD2026_ACTIONS,
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
    JOINT_ZERO_OFFSET,
    LEG_JOINT_INDICES,
    WHEEL_JOINT_INDICES,
    Cod2026BaseCfg,
    Cod2026BaseEnv,
    compute_cod2026_motor_ctrl,
)


# =====================================================================
# 配置类
# =====================================================================


@dataclass
class Cod2026JoystickSensor:
    gyro: str = "gyro"
    accel: str = "accel"
    local_linvel: str = "local_linvel"
    upvector: str = "upvector"
    base_pos: str = "base_pos"
    right_wheel: str = "right_wheel_pos"
    left_wheel: str = "left_wheel_pos"


@dataclass
class Cod2026RewardConfig:
    """Cod2026 奖励配置 —— 参考 go2w/go2/g1 的标准设计：scales + fns(ctx) + dispatch。

    设计原则：
      - 量级差 < 1000 倍（避免 base_height=-3000 这种独大）
      - 高度 / 姿态用 square(z_err) 线性惩罚（标准形式，不是高斯）
      - 速度追踪用标准高斯 exp(-err²/sigma²)
      - 无离散脉冲惩罚（termination）
      - alive 给小正分替代"别死"的激励
    """

    scales: dict[str, float] = field(
        default_factory=lambda: {
            # ── 主任务（正奖励，量级 1~5） ──────────────────────
            "tracking_lin_vel": 5.0,
            "tracking_ang_vel": 0.5,
            "body_accel_tracking": 1.5,
            # ── 姿态 / 高度（惩罚，量级 10~200） ───────────────────
            "orientation": -3.0,
            "base_height": -100.0,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.02,

            # ── Cod2026 特有：腿 / 对称（惩罚） ───────────────────
            "swing_vertical": -5.0,
            "swing_symmetry": -10.0,
            "swing_vel": -1.0,

            # ── 平滑 / 能耗（小惩罚，量级 1e-5~1e-3） ─────────────
            "action_rate": -0.002,
            "dof_torques_l2": -0.00005,
            "dof_acc_l2": -1.0e-7,
            "joint_power": -0.00005,

            # ── 存活（小正分） ───────────────────────────────────
            "alive": 0.5,

            # ── 经验性小惩罚 ─────────────────────────────────────
            "roll_on_turn": -3.0,

            # ── 站立姿态保持（低速/姿态恶化时启用） ────────────────
            "stand": -30.0,
        }
    )
    base_height_target: float = 0.31
    base_height_sigma: float = 0.05      # 高度奖励的 σ：err=5cm 拿 37%
    vertex_clearance_target: float = 0.10  # 离地间隙：0.1m 以上拿满
    vertex_clearance_sigma: float = 0.03   # 离地间隙 σ：低3cm拿37%
    tracking_sigma: float = 0.25
    tracking_sigma_lin: float = 0.2
    tracking_sigma_ang: float = 0.1
    tilt_limit: float = 25.0 * np.pi / 180.0
    n_steps_delay: int = 50
    # orientation_stand 的启用阈值
    stand_cmd_linvel_threshold: float = 0.1  # |cmd_vx| < 此值 → 认为应该站立
    stand_cmd_angvel_threshold: float = 0.1  # |cmd_yaw| < 此值 → 认为应该站立
    stand_tilt_threshold: float = 7.5 * np.pi / 180.0  # tilt > 此值 → 姿态已恶化


@registry.envcfg("Cod2026JoystickFlat")
@dataclass
class Cod2026JoystickCfg(Cod2026BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file="/home/xxx/ubuntu_software/unilab/UniLab/src/unilab/assets/robots/cod_2026_balance/scene_flat.xml"
        )
    )
    max_episode_seconds: float = 20.0
    sensor: Cod2026JoystickSensor = field(default_factory=Cod2026JoystickSensor)
    commands: Commands = field(
        default_factory=lambda: Commands(
            vel_limit=[[-2.5, -2.5, -2.5], [2.5, 2.5, 2.5]],
            resampling_time=10.0,
            rel_standing_envs=0.1,
        )
    )
    domain_rand: DomainRandConfig = field(
        default_factory=lambda: DomainRandConfig(
            push_robots=True,
            push_interval=200,
            max_force=[15.0, 15.0, 2.0],
            push_body_name="base",
            curriculum_enabled=True,
            curriculum_warmup_steps=2_000,
            curriculum_full_steps=500_000,
            curriculum_initial_scale=0.05,
            curriculum_shape="smooth",
        )
    )
    reward_config: Cod2026RewardConfig = field(default_factory=Cod2026RewardConfig)


# =====================================================================
# 领域随机化 Provider
# =====================================================================


class Cod2026JoystickDomainRandomizationProvider(LocomotionDRProvider):

    def _get_curriculum_scale(self, env: Any) -> float:
        dr = getattr(env.cfg, "domain_rand", None)
        if dr is None or not getattr(dr, "curriculum_enabled", False):
            return 1.0
        step_counter = getattr(env, "step_counter", 0)
        warmup = int(getattr(dr, "curriculum_warmup_steps", 0))
        full = int(getattr(dr, "curriculum_full_steps", 1))
        initial = float(getattr(dr, "curriculum_initial_scale", 0.0))
        shape = str(getattr(dr, "curriculum_shape", "linear"))
        if step_counter <= warmup:
            return float(initial)
        if step_counter >= full:
            return 1.0
        progress = float(step_counter - warmup) / float(max(1, full - warmup))
        if shape == "smooth":
            t = float(progress)
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            normalized = 3.0 * t * t - 2.0 * t * t * t
        elif shape == "quadratic":
            t = float(progress)
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            normalized = t * t
        else:
            normalized = float(progress)
            normalized = 0.0 if normalized < 0.0 else 1.0 if normalized > 1.0 else normalized
        return float(initial + (1.0 - initial) * normalized)

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        if capabilities is not None and getattr(env.cfg.domain_rand, "push_robots", False):
            if not capabilities.supports_interval_push:
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support interval push"
                )
            if not capabilities.supports_interval_body_force:
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support interval body force"
                )

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        domain_rand = getattr(env.cfg, "domain_rand", None)
        if domain_rand is None or not getattr(domain_rand, "push_robots", False):
            return None
        if not hasattr(self, "_push_remaining_steps"):
            self._push_remaining_steps = 0
            self._push_base_force = None
            self._push_leg_body_ids = None
            self._push_leg_force = None
        if self._push_remaining_steps > 0:
            self._push_remaining_steps -= 1
            return IntervalRandomizationPlan(
                push_perturbation_limit=self._push_base_force,
                body_ids=self._push_leg_body_ids,
                body_force=self._push_leg_force,
            )
        scale = self._get_curriculum_scale(env)
        pi_init = getattr(domain_rand, "push_interval_initial", None)
        pi_final = getattr(domain_rand, "push_interval_final", None)
        if pi_init is not None and pi_final is not None:
            current_push_interval = int(pi_init - scale * (pi_init - pi_final))
        else:
            current_push_interval = int(getattr(domain_rand, "push_interval", 200))
        if step_counter % current_push_interval != 0:
            return None

        backend = env._backend
        num_envs = backend._num_envs
        model = backend._model
        base_force_base = domain_rand.max_force
        base_force = [float(f) * scale for f in base_force_base]

        LEG_BODY_NAMES = ["Right_rear_child1_link", "Left_rear_child1_link"]
        leg_force_cfg = getattr(domain_rand, "leg_max_force", None)
        if leg_force_cfg is not None:
            LEG_MAX_FORCE_BASE = [float(x) for x in leg_force_cfg]
        else:
            LEG_MAX_FORCE_BASE = [20.0, 20.0, 3.0]
        LEG_MAX_FORCE = [f * scale for f in LEG_MAX_FORCE_BASE]

        leg_body_ids = []
        for bn in LEG_BODY_NAMES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
            if bid >= 0:
                leg_body_ids.append(bid)
        leg_body_ids = np.array(leg_body_ids, dtype=np.int32)

        if len(leg_body_ids) > 0:
            leg_fx = np.random.uniform(-LEG_MAX_FORCE[0], LEG_MAX_FORCE[0], (num_envs, len(leg_body_ids)))
            leg_fy = np.random.uniform(-LEG_MAX_FORCE[1], LEG_MAX_FORCE[1], (num_envs, len(leg_body_ids)))
            leg_fz = np.random.uniform(-LEG_MAX_FORCE[2], LEG_MAX_FORCE[2], (num_envs, len(leg_body_ids)))
            body_force = np.zeros((num_envs, len(leg_body_ids), 3), dtype=np.float64)
            body_force[:, :, 0] = leg_fx
            body_force[:, :, 1] = leg_fy
            body_force[:, :, 2] = leg_fz
        else:
            body_force = None

        duration = int(np.random.randint(3, 9))
        self._push_remaining_steps = duration
        self._push_base_force = base_force
        self._push_leg_body_ids = leg_body_ids if len(leg_body_ids) > 0 else None
        self._push_leg_force = body_force

        return IntervalRandomizationPlan(
            push_perturbation_limit=base_force,
            body_ids=self._push_leg_body_ids,
            body_force=body_force,
        )

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        scale = self._get_curriculum_scale(env)  # 姿态/摩擦等用原来的步数增长

        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))

        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        _facing = np.random.uniform(-np.pi , np.pi , (num_reset,))
        _backward = np.random.uniform(np.pi - np.pi , np.pi + np.pi , (num_reset,))
        _face_mask = np.random.uniform(size=(num_reset,)) < 0.5
        yaw = np.where(_face_mask, _facing, _backward)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))

        raw_w = np.random.uniform(0.0, 1.0, (num_reset, 4))
        w = raw_w.copy()
        for _ in range(3):
            w = np.clip(w, 0.0, 1.0)
            s = w.sum(axis=1, keepdims=True)
            s = np.where(s < 1e-6, 1.0, s)
            w = w / s * 2.0
        w = np.clip(w, 0.0, 1.0)
        w_pitch = w[:, 0]
        w_roll = w[:, 1]
        w_lin = w[:, 2]
        w_ang = w[:, 3]

        pitch_base = 0.15
        roll_base = 0.15
        pitch = np.random.uniform(-pitch_base * w_pitch * scale, +pitch_base * w_pitch * scale, (num_reset,))
        roll = np.random.uniform(-roll_base * w_roll * scale, +roll_base * w_roll * scale, (num_reset,))
        cpt, spt = np.cos(pitch / 2), np.sin(pitch / 2)
        q_pitch = np.stack([cpt, np.zeros(num_reset), spt, np.zeros(num_reset)], axis=1)
        cpr, spr = np.cos(roll / 2), np.sin(roll / 2)
        q_roll = np.stack([cpr, spr, np.zeros(num_reset), np.zeros(num_reset)], axis=1)
        qpos[:, 3:7] = np_quat_mul(np_quat_mul(q_roll, q_pitch), qpos[:, 3:7])

        wheel_vel_base = 2.0
        wheel_qvel_offsets = np.array([12, 19], dtype=np.int32)
        wheel_vel = np.random.uniform(-wheel_vel_base * scale, wheel_vel_base * scale, (num_reset, len(wheel_qvel_offsets)))
        qvel[:, wheel_qvel_offsets] = qvel[:, wheel_qvel_offsets] + wheel_vel

        swing_vel_base = 3.0
        leg_qvel_offsets = np.array([6, 10, 13, 17], dtype=np.int32)
        swing_vel = np.random.uniform(-swing_vel_base * scale, swing_vel_base * scale, (num_reset, len(leg_qvel_offsets)))
        qvel[:, 6] = qvel[:, 6] + swing_vel[:, 0]
        qvel[:, 10] = qvel[:, 10] + swing_vel[:, 1]
        qvel[:, 13] = qvel[:, 13] + swing_vel[:, 2]
        qvel[:, 17] = qvel[:, 17] + swing_vel[:, 3]

        linvel_base = 0.5
        angvel_base = 0.5
        w_lin_3 = np.broadcast_to(w_lin[:, np.newaxis], (num_reset, 3))
        w_ang_3 = np.broadcast_to(w_ang[:, np.newaxis], (num_reset, 3))
        qvel[:, 0:3] = np.asarray(
            np.random.uniform(-linvel_base * w_lin_3 * scale, +linvel_base * w_lin_3 * scale, size=(num_reset, 3)),
            dtype=get_global_dtype(),
        )
        qvel[:, 3:6] = np.asarray(
            np.random.uniform(-angvel_base * w_ang_3 * scale, +angvel_base * w_ang_3 * scale, size=(num_reset, 3)),
            dtype=get_global_dtype(),
        )

        commands = self._sample_commands(env, num_reset)

        init_actions = np.zeros((num_reset, NUM_COD2026_ACTIONS), dtype=get_global_dtype())
        info_updates: dict[str, Any] = {
            "commands": commands,
            "current_actions": init_actions,
            "last_actions": init_actions.copy(),
        }

        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=None,
        )

    def _compute_reset_obs(self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel):
        vx = linvel[:, 0]
        rw_all, lw_all = env.get_wheel_pos()
        rw = rw_all[env_ids]
        lw = lw_all[env_ids]
        return env._compute_obs(info_updates, gyro, gravity, dof_pos, dof_vel, rw, lw, vx=vx)

    def _sample_commands(self, env, num_reset):
        dtype = get_global_dtype()
        vx_low = float(env._cmd_vel_range[0, 0])
        vx_high = float(env._cmd_vel_range[0, 1])
        yaw_low = float(env._cfg.commands.vel_limit[0][2])
        yaw_high = float(env._cfg.commands.vel_limit[1][2])
        commands = np.asarray(
            np.random.uniform(
                low=[vx_low, 0.0, yaw_low],
                high=[vx_high, 0.0, yaw_high],
                size=(num_reset, 3),
            ),
            dtype=dtype,
        )
        zero_small_xy_commands(commands, threshold=0.05)

        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing, :3] = 0.0
        if getattr(env.cfg.commands, "heading_command", False):
            commands[:, 2] = 0.0

        if hasattr(env.cfg.commands, "height_limit"):
            hlim = np.asarray(env.cfg.commands.height_limit, dtype=dtype)
            height = np.asarray(
                np.random.uniform(float(hlim[0]), float(hlim[1]), size=(num_reset, 1)),
                dtype=dtype,
            )
            commands = np.concatenate([commands, height], axis=1)

        return commands
# =====================================================================
# 主环境类
# =====================================================================


@registry.env("Cod2026JoystickFlat", sim_backend="mujoco")
class Cod2026JoystickEnv(Cod2026BaseEnv):
    _cfg: Cod2026JoystickCfg

    def __init__(self, cfg: Cod2026JoystickCfg | None = None,
                 num_envs: int = 1, backend_type: str = "mujoco"):
        cfg = cfg or Cod2026JoystickCfg()
        backend = create_backend(
            backend_type, cfg.scene, num_envs, cfg.sim_dt,
            base_name=cfg.asset.base_name,
            add_body_sensors=True,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()

        try:
            self._right_wheel_body_id = self._backend.get_body_ids(["Right_Wheel_link"])[0]
            self._left_wheel_body_id = self._backend.get_body_ids(["Left_Wheel_link"])[0]
        except Exception:
            self._right_wheel_body_id = None
            self._left_wheel_body_id = None

        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)

        leg_kp_arr = np.asarray(self._cfg.control_config.leg_kp, dtype=self._np_dtype)
        leg_kd_arr = np.asarray(self._cfg.control_config.leg_kd, dtype=self._np_dtype)
        wheel_cfg = np.asarray(self._cfg.control_config.wheel_kd, dtype=self._np_dtype)
        self._motor_kp = np.broadcast_to(leg_kp_arr, (num_envs, NUM_LEG_ACTIONS)).copy()
        self._motor_kd = np.broadcast_to(leg_kd_arr, (num_envs, NUM_LEG_ACTIONS)).copy()
        self._wheel_kd = np.broadcast_to(wheel_cfg, (num_envs, NUM_WHEEL_ACTIONS)).copy()
        self._last_motor_ctrl = np.zeros((num_envs, NUM_COD2026_ACTIONS), dtype=self._np_dtype)

        self._backend.set_pre_step_control(self._pre_step_motor_control)

        self._init_domain_randomization(Cod2026JoystickDomainRandomizationProvider())
        self._init_reward_functions()

        # 延迟终止 / 差分加速度所需缓冲
        self._vertex_touch_count = np.zeros(num_envs, dtype=np.int32)
        self._body_low_count = np.zeros(num_envs, dtype=np.int32)
        self._tilt_count = np.zeros(num_envs, dtype=np.int32)
        self._last_dof_vel_for_acc = np.zeros((num_envs, NUM_COD2026_ACTIONS), dtype=self._np_dtype)
        self._last_rw_body = None   
        self._last_lw_body = None
        self._episode_steps = np.zeros(num_envs, dtype=np.int32)
        self._term_vertex = 0
        self._term_low = 0
        self._term_tilt = 0
        self._total_steps = 0
        self._term_window_steps = 0
        self._hit_vertex = 0
        self._hit_low = 0
        self._hit_tilt = 0

        self._episode_tracking_reward = np.zeros(num_envs, dtype=self._np_dtype)
        self._episode_vel_sum = np.zeros(num_envs, dtype=self._np_dtype)
        self._episode_cmd_vel_sum = np.zeros(num_envs, dtype=self._np_dtype)
        self._pbrs_lin_vel_prev = np.zeros(num_envs, dtype=self._np_dtype)
        self._pbrs_ang_vel_prev = np.zeros(num_envs, dtype=self._np_dtype)
        vel_limit = self._cfg.commands.vel_limit
        self._cmd_vel_curriculum_max = float(vel_limit[1][0])
        self._cmd_vel_range = np.array([
            [-0.2, 0.2],
        ], dtype=self._np_dtype)
        self._cmd_vel_ref = float(self._cmd_vel_range[0, 1])  # 保存初始最大值 0.2
        self._curriculum_good_accum = 0.0
        self._curriculum_bad_accum = 0.0
        self._curriculum_update_counter = 0
        self._curriculum_update_interval = 50
        self._curriculum_consecutive_good = 0
        self._curriculum_consecutive_bad = 0
        self._curriculum_sustain_threshold = 5

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 33}

    # -----------------------------------------------------------------
    # reset：清零缓存
    # -----------------------------------------------------------------

    def reset(self, env_indices: np.ndarray):
        env_ids = np.asarray(env_indices, dtype=np.int32)

        self._update_cmd_curriculum(env_ids)

        obs, info = super().reset(env_ids)
        self._vertex_touch_count[env_ids] = 0
        self._body_low_count[env_ids] = 0
        self._tilt_count[env_ids] = 0
        self._last_dof_vel_for_acc[env_ids] = 0.0
        cur_n = getattr(self, "_last_rw_body", None)
        total = cur_n.shape[0] if isinstance(cur_n, np.ndarray) else 0
        if total == 0 or total < int(env_ids.max()) + 1:
            n = len(self._vertex_touch_count)
            self._last_rw_body = np.zeros(n, dtype=get_global_dtype())
            self._last_lw_body = np.zeros(n, dtype=get_global_dtype())
            self._swing_first_step = np.ones(n, dtype=bool)
        self._swing_first_step[env_ids] = True
        self._episode_steps[env_ids] = 0
        self._episode_tracking_reward[env_ids] = 0.0
        self._episode_vel_sum[env_ids] = 0.0
        self._episode_cmd_vel_sum[env_ids] = 0.0
        self._pbrs_lin_vel_prev[env_ids] = 0.0
        self._pbrs_ang_vel_prev[env_ids] = 0.0
        return obs, info
    
    def _update_cmd_curriculum(self, env_ids: np.ndarray) -> None:
        if len(env_ids) == 0:
            return

        vel_sum = self._episode_vel_sum[env_ids]
        cmd_vel_sum = self._episode_cmd_vel_sum[env_ids]
        tracking_ratio = np.where(cmd_vel_sum > 0.01, vel_sum / cmd_vel_sum, 0.0)

        good = tracking_ratio > 0.55
        bad = ~good
        self._curriculum_good_accum += float(np.mean(good))
        self._curriculum_bad_accum += float(np.mean(bad))

        self._curriculum_update_counter += 1
        if self._curriculum_update_counter < self._curriculum_update_interval:
            return
        self._curriculum_update_counter = 0

        good_ratio = self._curriculum_good_accum / self._curriculum_update_interval
        bad_ratio = self._curriculum_bad_accum / self._curriculum_update_interval
        self._curriculum_good_accum = 0.0
        self._curriculum_bad_accum = 0.0

        v_max = self._cmd_vel_curriculum_max
        step = 0.05
        min_range = 0.2

        if good_ratio > bad_ratio:
            self._curriculum_consecutive_good += 1
            self._curriculum_consecutive_bad = 0
        else:
            self._curriculum_consecutive_bad += 1
            self._curriculum_consecutive_good = 0

        if self._curriculum_consecutive_good >= self._curriculum_sustain_threshold:
            self._curriculum_consecutive_good = 0
            self._cmd_vel_range[0, 0] = np.clip(self._cmd_vel_range[0, 0] - step, -v_max, -min_range)
            self._cmd_vel_range[0, 1] = np.clip(self._cmd_vel_range[0, 1] + step, min_range, v_max)
        elif self._curriculum_consecutive_bad >= self._curriculum_sustain_threshold:
            self._curriculum_consecutive_bad = 0
            self._cmd_vel_range[0, 0] = np.clip(self._cmd_vel_range[0, 0] + step, -v_max, -min_range)
            self._cmd_vel_range[0, 1] = np.clip(self._cmd_vel_range[0, 1] - step, min_range, v_max)
 
    # -----------------------------------------------------------------
    # action → PD torque
    # -----------------------------------------------------------------

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.asarray(
            np.clip(
                actions,
                -self._cfg.control_config.clip_actions,
                self._cfg.control_config.clip_actions,
            ),
            dtype=self._np_dtype,
        )
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped_actions))
        state.info["current_actions"] = clipped_actions

        targets = np.zeros((clipped_actions.shape[0], NUM_COD2026_ACTIONS), dtype=self._np_dtype)
        for idx in [0, 1, 3, 4]:
            targets[:, idx] = (
                exec_actions := clipped_actions
            )[:, idx] * self._cfg.control_config.action_scale + self.default_angles[idx]
        for idx in [2, 5]:
            targets[:, idx] = clipped_actions[:, idx] * self._cfg.control_config.wheel_action_scale
        return targets

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        #policy_ctrl[:, 2] = 0.0
        #policy_ctrl[:, 5] = 0.0
        joint_pos = self.get_dof_pos()
        joint_vel = self.get_dof_vel()
        motor_ctrl = compute_cod2026_motor_ctrl(
            policy_ctrl,
            joint_pos,
            joint_vel,
            self._motor_kp,
            self._motor_kd,
            self._wheel_kd,
            self._ctrl_lower,
            self._ctrl_upper,
            self._last_motor_ctrl,
        )
        self._debug_joint_pos = joint_pos
        self._debug_joint_vel = joint_vel
        self._debug_motor_ctrl = motor_ctrl
        return motor_ctrl

    # -----------------------------------------------------------------
    # 传感器辅助
    # -----------------------------------------------------------------

    def get_accel(self) -> np.ndarray:
        return np.asarray(self._backend.get_sensor_data(self._cfg.sensor.accel), dtype=get_global_dtype())

    def get_upvector(self) -> np.ndarray:
        return np.asarray(self._backend.get_sensor_data(self._cfg.sensor.upvector), dtype=get_global_dtype())

    def get_wheel_pos(self) -> tuple[np.ndarray, np.ndarray]:
        rw = np.asarray(self._backend.get_sensor_data(self._cfg.sensor.right_wheel), dtype=get_global_dtype())
        lw = np.asarray(self._backend.get_sensor_data(self._cfg.sensor.left_wheel), dtype=get_global_dtype())
        return rw, lw

    def get_base_pos(self) -> np.ndarray:
        return np.asarray(self._backend.get_sensor_data(self._cfg.sensor.base_pos), dtype=get_global_dtype())

    def get_base_quat(self) -> np.ndarray:
        return np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())

    def get_world_linvel(self) -> np.ndarray:
        return np.asarray(self._backend.get_base_lin_vel(), dtype=get_global_dtype())

    def _get_body_min_z(self, base_pos: np.ndarray, upvector: np.ndarray) -> np.ndarray:
        """
        用几何公式计算主体碰撞 box 最低点的世界 z 坐标。
        基于 XML: <geom name="base_box" type="box" size="0.28 0.12 0.10" pos="-0.02 0 -0.1"/>

        box 中心 (body 坐标系): c = (-0.02, 0.00, -0.10)
        box 半轴: h = (0.28, 0.12, 0.10)

        对 box 任意顶点 v_body = c + (±hx, ±hy, ±hz):
            v_world_z = base_z + upvector · v_body

        8 个顶点中最低的 z (数学精确解):
            min_z = base_z + upvector · c - (|ux|*hx + |uy|*hy + |uz|*hz)

        返回 (num_envs,)：每个 env 中 box 最低点的世界 z 坐标。
        当 min_z < 0.08m 时，认为主体已碰地（22°跌倒时最低点约 5cm）。
        """
        box_center = np.asarray([-0.02, 0.0, -0.10], dtype=self._np_dtype)
        box_half = np.asarray([0.28, 0.12, 0.10], dtype=self._np_dtype)

        base_z = base_pos[:, 2]  # (num_envs,)
        # upvector · box_center: body 坐标系中 box 中心相对原点的 z 偏移
        center_offset = np.sum(upvector * box_center, axis=1)  # (num_envs,)
        # |upvector| · box_half: 旋转后的最大 z 方向延伸
        vertex_extent = np.sum(np.abs(upvector) * box_half, axis=1)  # (num_envs,)

        return base_z + center_offset - vertex_extent  # box 最低点的世界 z

    # -----------------------------------------------------------------
    # observation
    # -----------------------------------------------------------------

    def _compute_obs(self, info, gyro, upvector, dof_pos, dof_vel, rw, lw, vx=None):
        n = gyro.shape[0]
        dtype = get_global_dtype()
        last_actions = np.asarray(info.get("last_actions", np.zeros((n, NUM_COD2026_ACTIONS), dtype=dtype)), dtype=dtype)
        commands = np.asarray(info.get("commands", np.zeros((n, 4), dtype=dtype)), dtype=dtype)

        rw_theta = info.get("swing_rw_theta")
        lw_theta = info.get("swing_lw_theta")
        if rw_theta is None:
            rw_theta = -np.arctan2(rw[:, 0], -rw[:, 2])
        if lw_theta is None:
            lw_theta = -np.arctan2(lw[:, 0], -lw[:, 2])

        rw_x, rw_z = rw[:, 0], rw[:, 2]
        lw_x, lw_z = lw[:, 0], lw[:, 2]

        if vx is None:
            vx = np.zeros((n, 1), dtype=dtype)
        elif vx.ndim == 1:
            vx = vx.reshape(-1, 1)

        obs_flat = np.asarray(np.concatenate([
            gyro, upvector, dof_pos, dof_vel, last_actions, commands,
            rw_theta.reshape(-1, 1),
            lw_theta.reshape(-1, 1),
            np.sqrt(rw_x**2 + rw_z**2).reshape(-1, 1),
            np.sqrt(lw_x**2 + lw_z**2).reshape(-1, 1),
            vx,
        ], axis=1), dtype=dtype)

        obs_scaled = np.clip(obs_flat, -100.0, 100.0).astype(dtype, copy=False)
        return {"obs": obs_scaled}

    # =================================================================
    # 奖励系统（go2w 规范：scales + fns(ctx) + dispatch）
    # =================================================================

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            # ── 标准共享（与 go2w/go2/g1 一致） ───────────────────
            "lin_vel_z": common_rewards.lin_vel_z,
            "ang_vel_xy": common_rewards.ang_vel_xy,
            "orientation": common_rewards.orientation,  # 倾斜惩罚：g_x²+g_y²
            "action_rate": common_rewards.action_rate,
            "alive": common_rewards.alive,
            "dof_torques_l2": common_rewards.dof_torques_l2,
            "dof_acc_l2": common_rewards.dof_acc_l2,
            "joint_power": common_rewards.joint_power,
            # ── Cod2026 特有：
            # ——速度追踪奖励 ——————————————————
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "tracking_lin_vel_pbrs": self._reward_tracking_lin_vel_pbrs,
            "tracking_ang_vel_pbrs": self._reward_tracking_ang_vel_pbrs,
            # ——高度奖励 ——————————————————
            "base_height": self._reward_base_height,
            # ——腿摆角 / 对称惩罚 ───────────────────────
            "swing_vertical": self._reward_swing_vertical,
            "swing_symmetry": self._reward_swing_symmetry,
            "swing_vel": self._reward_swing_vel,
            # ── 轮子速度直接跟随命令 ────────────
            "wheel_target": self._reward_wheel_target,
            # ── 倒立摆侧倾稳定 ───────────────────────────────────
            "roll_on_turn": self._reward_roll_on_turn,
            # ── 机体加速度跟踪（单独一项，不与 θ_tg 耦合） ─────
            "body_accel_tracking": self._reward_body_accel_tracking,
            # ── 轮子空转/打滑惩罚 ───────────────────────────
            "wheel_slip": self._reward_wheel_slip,
            # ── 离地高度奖励 ────────────────────────────────────
            "vertex_clearance": self._reward_vertex_clearance,
        }

    # ── A. 姿态 / 高度 ──────────────────────────────────────────────

    def _reward_roll_on_turn(self, ctx: RewardContext) -> np.ndarray:
        """
        常驻 roll 控制：不转向时也惩罚 roll（基线），转向时加权。
        roll = arcsin(gravity[:, 1])，直接反映真实侧倾角度。
        公式：clip( (0.3 + |cmd_ang|) × |roll| / 0.1, 0, 1 ) —— 不转向时 0.3×|roll|，转向时更强，
        最大惩罚 clip 到 1，避免跟 orientation 叠加后惩罚失控。
        """
        cmd_ang = ctx.info["commands"][:, 2] 
        g = ctx.gravity
        assert g is not None
        roll = np.arcsin(np.clip(g[:, 1], -1.0, 1.0))
        raw = (0.3 + np.abs(cmd_ang)) * np.abs(roll) / 0.1
        return np.clip(raw, 0.0, 1.0)
    def _reward_vertex_clearance(self, ctx: RewardContext) -> np.ndarray:
        """离地间隙奖励：body_box 最低点离地越高越好。
        
        target: 0.12m (安全距离)
        sigma: 0.03m (3cm 偏差拿 37%)
        
        vertex_z > 0.15m → 接近满分
        vertex_z = 0.12m → 0.5 (目标)
        vertex_z = 0.09m → 0.18 (危险)
        vertex_z < 0.05m → 几乎零分 (即将触地)
        """
        rc = self._cfg.reward_config
        target = float(getattr(rc, "vertex_clearance_target", 0.15))
        sigma = float(getattr(rc, "vertex_clearance_sigma", 0.03))
        
        body_vertex_z = np.asarray(ctx.info["body_vertex_z"], dtype=get_global_dtype())
        
        # 只惩罚低于 target 的部分
        shortfall = np.maximum(0.0, target - body_vertex_z)
        
        # 高斯衰减： shortfall=0 → 1.0, shortfall=sigma → 0.368
        return np.square(shortfall) / (sigma * sigma)
    def _reward_termination(self, ctx: RewardContext) -> np.ndarray:
        """终止时的离散大惩罚：terminated=True 的环境扣 1，其他不扣。"""
        terminated = ctx.info.get("terminated")
        if terminated is None:
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())
        return np.asarray(terminated, dtype=get_global_dtype())

    def _reward_base_height(self, ctx: RewardContext) -> np.ndarray:
        """高度目标奖励（高斯衰减）：越接近目标高度拿越多。

        σ = rc.base_height_sigma（YAML 可调，默认 0.05m）。
        err=0cm  → 1.0
        err=5cm  → exp(-1)  ≈ 0.368
        err=10cm → exp(-4)  ≈ 0.018
        err=15cm → exp(-9)  ≈ 0.0001（几乎为 0，不必再加硬截止）

        scale 通常为负数，所以这一项实际是「高度偏离目标越远 → 扣得越多」。
        """
        rc = self._cfg.reward_config
        sigma = float(getattr(rc, "base_height_sigma", 0.04))
        actual_height = np.asarray(ctx.base_height, dtype=get_global_dtype())
        target_height = np.asarray(ctx.info["commands"][:, 3], dtype=get_global_dtype())
        err = actual_height - target_height
        return np.exp(-np.square(err) / (sigma * sigma))


    def _soft_posture_gate(
        self,
        tilt: np.ndarray,
        theta_err: np.ndarray,
        base_z: np.ndarray,
        *,
        low: float = 0.6,
        high: float = 1.2,
        tilt_center: float = 6.0,
        tilt_width: float = 4.0,
        swing_center: float = 18.0,
        swing_width: float = 6.0,
        height_low_center: float = 0.26,
        height_low_width: float = 0.014,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """连续软姿态门控：三个 sigmoid 的乘积，从 low 平滑过渡到 high。

        tilt / swing_err 越小、|base_z - target_z| 越小 → gate → high（姿态好）。
        tilt / swing_err 越大、|base_z - target_z| 越大 → gate → low（姿态差）。
        每个 sigmoid 在 center 处输出 0.5，width 控制过渡快慢。
        高度维度使用相对误差 |base_z − target_z|，适配动态目标高度。
        """
        dtype = get_global_dtype()
        g_tilt = 1.0 / (1.0 + np.exp((np.rad2deg(tilt) - tilt_center) / tilt_width))
        g_swing = 1.0 / (1.0 + np.exp((np.rad2deg(theta_err) - swing_center) / swing_width))
        g_height = 1.0 / (1.0 + np.exp((height_low_center - base_z) / height_low_width))
        combined = g_tilt * g_swing * g_height
        gate = np.asarray(low + (high - low) * combined, dtype=dtype)
        return gate, g_tilt, g_swing, g_height

    def _reward_tracking_lin_vel(self, ctx: RewardContext) -> np.ndarray:
        """世界坐标系下的速度追踪：将机体坐标系指令旋转到世界坐标系，再与世界坐标系速度比较。

        好处：转向时速度追踪更稳定，不会因 pitch 变化导致机体坐标系速度分量突变。
        """
        commands = ctx.info["commands"]
        vx = ctx.info["body_vx"]
        vy = ctx.info["body_vy"]
        upvector = ctx.info["upvector"]
        cos_pitch = np.cos(np.arcsin(np.clip(-upvector[:, 0], -1.0, 1.0)))
        vx_horiz = vx * cos_pitch
        lin_vel_error = np.square(commands[:, 0] - vx_horiz) + np.square(commands[:, 1] - vy)
        sigma_base = ctx.tracking_sigma_lin if ctx.tracking_sigma_lin is not None else ctx.tracking_sigma
        cmd_ref = self._cmd_vel_ref
        sigma = np.maximum(sigma_base, sigma_base * np.abs(commands[:, 0]) / cmd_ref)
        raw = np.exp(-lin_vel_error / (sigma * sigma))

        _tilt = ctx.info["tilt"]
        _rw = ctx.info["swing_rw_theta"]
        _lw = ctx.info["swing_lw_theta"]
        _tg = ctx.info["swing_target_theta"]
        _theta_err = np.maximum(np.abs(_rw - _tg), np.abs(_lw - _tg))
        _base_z = ctx.info["base_z"]
        _target_z = ctx.info["commands"][:, 3]

        gating, g_tilt, g_swing, g_height = self._soft_posture_gate(_tilt, _theta_err, _base_z, low=0.6, high=1.2)
        ctx.info["linvel_gate"] = gating
        ctx.info["linvel_g_tilt"] = g_tilt
        ctx.info["linvel_g_swing"] = g_swing
        ctx.info["linvel_g_height"] = g_height
        return np.asarray(raw * gating, dtype=get_global_dtype())

    def _reward_tracking_ang_vel(self, ctx: RewardContext) -> np.ndarray:
        """和 common_rewards.tracking_ang_vel 一样的高斯形式，只是在末尾加姿态门控。

        作用：避免 policy 为了追角速度指令（转向）而甩倾斜/乱晃腿——
        姿态好才给全额（×1.2），姿态差打折扣（×0.6），和直线速度的门控保持一致。
        """
        commands = ctx.info["commands"]
        ang_vel_error = np.square(commands[:, 2] - ctx.gyro[:, 2])
        sigma_base = ctx.tracking_sigma_ang if ctx.tracking_sigma_ang is not None else ctx.tracking_sigma
        cmd_ref = float(self._cfg.commands.vel_limit[1][2])
        sigma = np.maximum(sigma_base, sigma_base * np.abs(commands[:, 2]) / cmd_ref)
        raw = np.exp(-ang_vel_error / (sigma * sigma))

        _tilt = ctx.info["tilt"]
        _rw = ctx.info["swing_rw_theta"]
        _lw = ctx.info["swing_lw_theta"]
        _tg = ctx.info["swing_target_theta"]
        _theta_err = np.maximum(np.abs(_rw - _tg), np.abs(_lw - _tg))
        _base_z = ctx.info["base_z"]
        _target_z = ctx.info["commands"][:, 3]

        gating, _, _, _ = self._soft_posture_gate(_tilt, _theta_err, _base_z, low=0.6, high=1.2)
        return np.asarray(raw * gating, dtype=get_global_dtype())

    def _reward_swing_vertical(self, ctx: RewardContext) -> np.ndarray:
        """腿杆偏离动态目标摆角的惩罚（加姿态门控）。

        用绝对值而不是平方——平方在弧度小时惩罚太小，方向反了几乎感觉不到。
        姿态门控：tilt > 10° / theta_err > 8° / base_z < 0.28 → 姿态不稳，
        降低摆腿惩罚强度（×0.6），让它先回正再去追摆角目标，
        避免"前倾时被 swing_vertical 钉死在错误姿态上"。
        """
        err_rw = ctx.info["swing_err_rw"]
        err_lw = ctx.info["swing_err_lw"]
        raw = np.abs(err_rw) + np.abs(err_lw)

        _tilt = ctx.info["tilt"]
        _rw = ctx.info["swing_rw_theta"]
        _lw = ctx.info["swing_lw_theta"]
        _tg = ctx.info["swing_target_theta"]
        _theta_err = np.maximum(np.abs(_rw - _tg), np.abs(_lw - _tg))
        _base_z = ctx.info["base_z"]
        _target_z = ctx.info["commands"][:, 3]
        posture_gate, _, _, _ = self._soft_posture_gate(
            _tilt, _theta_err, _base_z, low=1.5, high=0.5,
            swing_center=14.0, swing_width=6.0,
        )

        return np.asarray(raw * posture_gate, dtype=get_global_dtype())


    def _reward_wheel_target(self, ctx: RewardContext) -> np.ndarray:
        """前进+转向一起考虑：左右轮分别设置目标角速度（差速驱动）。

        给摆腿发动留出时间；几轮后目标自然回到原始值。
        """
        cmd_vx = ctx.info["commands"][:, 0]
        cmd_ang = ctx.info["commands"][:, 2]
        rw_vel = ctx.info["rw_vel"]
        lw_vel = ctx.info["lw_vel"]

        # 顶点触地时轮子目标归零，防止栽倒时轮子继续加速
        body_vertex_z = ctx.info.get("body_vertex_z", np.ones_like(cmd_vx) * 0.2)
        vertex_touch = body_vertex_z < 0.035
        safe_mask = ~vertex_touch
        cmd_vx = cmd_vx * safe_mask
        cmd_ang = cmd_ang * safe_mask

        # 差速驱动：左轮 = effective - 0.5*cmd_ang*L；右轮 = effective + 0.5*cmd_ang*L
        L = 0.4155  # 轮距
        target_left = cmd_vx - 0.5 * cmd_ang * L
        target_right = cmd_vx + 0.5 * cmd_ang * L

        # 单个轮子的奖励：omega(rad/s)×wheel_r→线速度(m/s)，与cmd_vx同单位比较
        # 方向对→speed(0~1.0)，方向错且|omega|>0.05→对称惩罚(-1.0量级)，无命令→omega小得分高
        wheel_r = 0.06  # 轮半径 m，rad/s × m = m/s
        def _wheel(omega, target):
            wheel_speed = omega * wheel_r
            t_speed = np.where(np.abs(target) > 0.06, target, 0.0)
            has_cmd = np.abs(t_speed) > 1e-6
            sigma = np.where(has_cmd, np.maximum(np.abs(t_speed) * 0.5, 0.05), 0.25)
            raw = np.exp(-np.square(wheel_speed - t_speed) / (2.0 * np.square(sigma)))
            return raw

        combined = (_wheel(lw_vel, target_left) + _wheel(rw_vel, target_right)) * 0.5
        _tilt = ctx.info["tilt"]
        _theta_err = np.maximum(np.abs(ctx.info["swing_rw_theta"] - ctx.info["swing_target_theta"]),
                                np.abs(ctx.info["swing_lw_theta"] - ctx.info["swing_target_theta"]))
        _base_z = ctx.info["base_z"]
        gating, _, _, _ = self._soft_posture_gate(_tilt, _theta_err, _base_z, low=0.6, high=1.2)
        return np.asarray(combined * gating, dtype=get_global_dtype())

    def _reward_swing_vel(self, ctx: RewardContext) -> np.ndarray:
        """摆角速度惩罚（返回 ≤ 0）—— 从 info 读，不需要假设关节索引"""
        return np.asarray(ctx.info["swing_vel_sq"], dtype=get_global_dtype())

    def _reward_swing_symmetry(self, ctx: RewardContext) -> np.ndarray:
        """左右腿不对称惩罚（1-exp(-(θ_r-θ_l)²/0.1)，对称→0，不对称→1；配合负 scale）。"""
        rw_theta = ctx.info["swing_rw_theta"]
        lw_theta = ctx.info["swing_lw_theta"]
        ang_diff = np.square(rw_theta - lw_theta)
        return np.asarray(1.0 - np.exp(-ang_diff / 0.1), dtype=get_global_dtype())

    def _reward_tracking_lin_vel_pbrs(self, ctx: RewardContext) -> np.ndarray:
        """PBRS: 只奖励进步，不奖励已经好。reward = φ(t) - φ(t-1)。"""
        phi = self._reward_tracking_lin_vel(ctx)
        delta = ~ctx.info["terminated"] * (phi - self._pbrs_lin_vel_prev)
        self._pbrs_lin_vel_prev = phi.copy()
        return np.asarray(delta, dtype=get_global_dtype())

    def _reward_tracking_ang_vel_pbrs(self, ctx: RewardContext) -> np.ndarray:
        """PBRS: 只奖励进步，不奖励已经好。reward = φ(t) - φ(t-1)。"""
        phi = self._reward_tracking_ang_vel(ctx)
        delta = ~ctx.info["terminated"] * (phi - self._pbrs_ang_vel_prev)
        self._pbrs_ang_vel_prev = phi.copy()
        return np.asarray(delta, dtype=get_global_dtype())

    def _reward_wheel_slip(self, ctx: RewardContext) -> np.ndarray:
        """轮子空转/打滑惩罚：
        轮子转了但机体没动（或方向反了）→ 惩罚
        slip = |wheel_vel_avg − vx|
        0.2 m/s 差 → 惩罚 0.63；0.4 m/s 差 → 惩罚 0.98
        """
        wheel_vel = ctx.info["wheel_vel_avg"]
        vx = ctx.info["body_vx"]
        sigma = 0.1   # (m/s)²：0.2 m/s 差时惩罚 ≈ 0.63
        slip_sq = np.square(wheel_vel*0.06 - vx)
        return 1.0 - np.exp(-slip_sq / sigma)

    def _reward_body_accel_tracking(self, ctx: RewardContext) -> np.ndarray:
        """世界坐标系下的加速度跟踪奖励（带方向对齐的摆腿门控）。

        target_accel = k_accel * (cmd_vx - vx_horiz)  （P 控制器：水平面投影速度误差 → 期望加速度）
        actual_accel = (vx_horiz - prev_vx_horiz) / ctrl_dt  （实际水平面投影加速度）
        奖励：exp(-(target - actual)² / σ²)

        与 tracking_lin_vel 的区别：tracking 奖"最终速度对了"，
        本项奖"加速度方向和大小对了"——
        强制 policy 真的加速/减速，而不是靠倾斜身体蹭速度分。

        方向对齐摆腿门控（新增）：
          - 前进加速（cmd_vx - vx_horiz > 0）：腿应后摆（theta > target）→ swing_err 正
          - 后退/减速（cmd_vx - vx_horiz < 0）：腿应前摆（theta < target）→ swing_err 负          - 方向对了时 swing_err 与 speed_err 同号 → aligned_swing > 0 → gate 增长
          - 死区内不需要加速 → gate = 1（满分）
        只有摆腿方向与加速方向一致时才给奖励，防止靠倾斜身体 / 车轮空转蹭分。
        """
        commands = ctx.info["commands"]
        cmd_vx = commands[:, 0]
        vx = ctx.info["body_vx"]
        upvector = ctx.info["upvector"]
        cos_pitch = np.cos(np.arcsin(np.clip(-upvector[:, 0], -1.0, 1.0)))
        vx_horiz = vx * cos_pitch
        ctrl_dt = 0.02
        k_accel = 2.5  # m/s² per (m/s error)：误差 0.5 m/s → 目标 1.5 m/s²

        prev_vx = ctx.info.get("prev_body_vx", vx)
        prev_vx_horiz = prev_vx * cos_pitch
        actual_accel = (vx_horiz - prev_vx_horiz) / ctrl_dt

        # ── 死区：|cmd_vx − vx| < 0.15 m/s 时直接给满分 ──
        speed_err = cmd_vx - vx_horiz
        in_deadzone = np.abs(speed_err) < 0.15
        target_accel = k_accel * speed_err

        sigma = 1.5   # (m/s²)²：误差 ≈ 1.5 m/s² 时奖励降到 37%
        err_sq = np.square(target_accel - actual_accel)
        raw = np.where(in_deadzone, 1.0, np.exp(-err_sq / sigma))

        _tilt = ctx.info["tilt"]
        _rw = ctx.info["swing_rw_theta"]
        _lw = ctx.info["swing_lw_theta"]
        _tg = ctx.info["swing_target_theta"]
        _theta_err = np.maximum(np.abs(_rw - _tg), np.abs(_lw - _tg))
        _base_z = ctx.info["base_z"]

        _target_z = ctx.info["commands"][:, 3]
        gating, _, _, _ = self._soft_posture_gate(
            _tilt, _theta_err, _base_z, low=0.6, high=1.0,
            tilt_center=8.0, tilt_width=4.0,
            swing_center=18.0, swing_width=6.0,
        )
        return np.asarray(raw * gating, dtype=get_global_dtype())

    # ── 主计算：构建 context + dispatch ────────────────────────────
    def _compute_reward(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        upvector: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        base_pos: np.ndarray,
        num_envs: int,
    ) -> np.ndarray:
        rc = self._cfg.reward_config
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=num_envs,
            default_angles=np.asarray(self.default_angles, dtype=self._np_dtype),
            tracking_sigma=rc.tracking_sigma,
            tracking_sigma_lin=rc.tracking_sigma_lin,
            tracking_sigma_ang=rc.tracking_sigma_ang,
            base_height_target=info["commands"][:, 3],
            base_height=base_pos[:, 2],
            gravity=upvector,
        )
        return run_reward_dispatch(
            scales=rc.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=True,
            ctrl_dt=float(self._cfg.ctrl_dt),
        )

    # -----------------------------------------------------------------
    # commands 刷新
    # -----------------------------------------------------------------

    def _update_commands(self, info: dict) -> None:
        """标准项目写法：初始化 + 按 resampling_time 周期性刷新命令。

        - steps: info["steps"] 为 (num_envs,) uint32 数组，每环境独立递增
        - resample: steps % interval_steps == 0 时刷新对应 env 的 cmd
        - 保持 4 维：[vx, vy, vyaw, height]
        - 调用 zero_small_xy_commands + rel_standing_envs 与 reset 时保持一致
        """
        dtype = get_global_dtype()
        n = self._num_envs

        # —— ① 初始化：commands 缺失 / 形状不对时兜底 ——
        commands = np.asarray(info.get("commands"), dtype=dtype)
        if commands is None or commands.shape != (n, 4):
            default_height = float(self._cfg.reward_config.base_height_target)
            commands = np.concatenate(
                [np.zeros((n, 3), dtype=dtype), np.full((n, 1), default_height, dtype=dtype)],
                axis=1,
            )

        # —— ② 每环境步数计数（对齐 go2w：只读，框架每步自动递增）——
        steps = np.asarray(
            info.get("steps", np.zeros((n,), dtype=np.uint32)),
            dtype=np.uint32,
        )

        # —— ③ 周期性重采样（resampling_time > 0 才开启）——
        resampling_time = float(getattr(self._cfg.commands, "resampling_time", 0.0))
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / float(self._cfg.ctrl_dt))), 1)
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                vx_low = float(self._cmd_vel_range[0, 0])
                vx_high = float(self._cmd_vel_range[0, 1])
                yaw_low = float(self._cfg.commands.vel_limit[0][2])
                yaw_high = float(self._cfg.commands.vel_limit[1][2])
                new_cmd = np.asarray(
                    np.random.uniform(
                        low=[vx_low, 0.0, yaw_low],
                        high=[vx_high, 0.0, yaw_high],
                        size=(num_resample, 3),
                    ),
                    dtype=dtype,
                )
                zero_small_xy_commands(new_cmd, threshold=0.05)
                standing_prob = float(getattr(self._cfg.commands, "rel_standing_envs", 0.0))
                if standing_prob > 0.0:
                    standing = np.random.uniform(size=(num_resample,)) < min(standing_prob, 1.0)
                    new_cmd[standing, :3] = 0.0
                if hasattr(self._cfg.commands, "height_limit"):
                    hlim = np.asarray(self._cfg.commands.height_limit, dtype=dtype)
                    height = np.asarray(
                        np.random.uniform(float(hlim[0]), float(hlim[1]), size=(num_resample, 1)),
                        dtype=dtype,
                    )
                else:
                    default_height = float(self._cfg.reward_config.base_height_target)
                    height = np.full((num_resample, 1), default_height, dtype=dtype)
                new_cmd_4d = np.concatenate([new_cmd, height], axis=1)
                commands[resample_mask] = new_cmd_4d

        info["commands"] = commands

    # -----------------------------------------------------------------
    # update_state：状态计算 → 塞 info → 调用 _compute_reward → 构建 obs
    # -----------------------------------------------------------------

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        gyro = self.get_gyro()
        upvector = self.get_upvector()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        base_pos = self.get_base_pos()
        world_linvel = self.get_world_linvel()
        local_linvel = self.get_local_linvel()
        rw, lw = self.get_wheel_pos()

        num_envs = gyro.shape[0]
        commands = state.info["commands"]
        _dt = float(self._cfg.ctrl_dt)

        # body 系 sagittal 平面内的腿摆角（0=竖直向下，正=后摆，负=前摆）
        # 用 arctan2 直接算角度，避免 3D 归一化时 y 分量（左右间距）污染方向
        rw_theta = -np.arctan2(rw[:, 0], -rw[:, 2])
        lw_theta = -np.arctan2(lw[:, 0], -lw[:, 2])

        # 延迟终止（用 box 几何精确计算最低点 z，不依赖 touch 传感器）
        rc = self._cfg.reward_config
        tilt = np.arccos(np.clip(upvector[:, 2], 0.0, 1.0))
        body_vertex_z = self._get_body_min_z(base_pos, upvector)
        vertex_touch = body_vertex_z < 0.035  # box 最低点低于 35cm → 认为主体已碰地
        body_too_low = base_pos[:, 2] < 0.25
        body_tilt_too_high = tilt > rc.tilt_limit
        self._vertex_touch_count = np.where(vertex_touch, self._vertex_touch_count + 1, 0)
        self._body_low_count = np.where(body_too_low, self._body_low_count + 1, 0)
        self._tilt_count = np.where(body_tilt_too_high, self._tilt_count + 1, 0)
        terminated = (
            (self._vertex_touch_count >= rc.n_steps_delay)
            | (self._body_low_count >= rc.n_steps_delay)
            | (self._tilt_count >= rc.n_steps_delay)
        )
        self._episode_steps += 1
        self._total_steps += num_envs
        self._term_window_steps += num_envs
        self._hit_vertex += int(np.sum(vertex_touch))
        self._hit_low += int(np.sum(body_too_low))
        self._hit_tilt += int(np.sum(body_tilt_too_high))
        if np.any(terminated):
            self._term_vertex += int(np.sum(
                (self._vertex_touch_count >= rc.n_steps_delay) & terminated
            ))
            self._term_low += int(np.sum(
                (self._body_low_count >= rc.n_steps_delay) & terminated
            ))
            self._term_tilt += int(np.sum(
                (self._tilt_count >= rc.n_steps_delay) & terminated
            ))
        if self._term_window_steps >= 200 * num_envs:
            self._term_vertex = 0
            self._term_low = 0
            self._term_tilt = 0
            self._term_window_steps = 0
            self._hit_vertex = 0
            self._hit_low = 0
            self._hit_tilt = 0

        # torques / qacc
        torques = self._last_motor_ctrl.copy()
        qacc = (dof_vel - self._last_dof_vel_for_acc) / _dt
        self._last_dof_vel_for_acc[:] = dof_vel

        # 摆角速度：body 系 theta 的时间变化率（真正的物理意义）
        # per-env 计算：first_step 的 env 速度设为 0，其他 env 正常差分
        d_rw = rw_theta - self._last_rw_body
        d_lw = lw_theta - self._last_lw_body
        swing_vel_sq = d_rw ** 2 + d_lw ** 2
        swing_vel_sq = swing_vel_sq / (_dt ** 2)
        if hasattr(self, "_swing_first_step"):
            swing_vel_sq = np.where(self._swing_first_step, 0.0, swing_vel_sq)
            self._swing_first_step = np.zeros(num_envs, dtype=bool)
        self._last_rw_body = rw_theta.copy()
        self._last_lw_body = lw_theta.copy()
        wheel_vel_avg = np.mean(dof_vel[:, WHEEL_JOINT_INDICES], axis=1)
        rw_vel = dof_vel[:, WHEEL_JOINT_INDICES[0]]
        lw_vel = dof_vel[:, WHEEL_JOINT_INDICES[1]]

        # 摆角目标固定为默认角度，不跟速度误差走——
        # 方向/加速的信号由 body_accel_tracking 和 wheel_target 提供，
        # swing_vertical 只负责「腿别偏离默认位置太远」。
        swing_target_theta = np.full_like(rw_theta, np.deg2rad(3.0))
        swing_err_rw = rw_theta - swing_target_theta
        swing_err_lw = lw_theta - swing_target_theta

        # 把奖励函数需要的量塞进 info
        state.info.update({
            "tilt": tilt,
            "base_z": base_pos[:, 2],
            "base_pos_x": base_pos[:, 0],
            "torques": torques,
            "qacc": qacc,
            "swing_rw_theta": rw_theta,
            "swing_lw_theta": lw_theta,
            "swing_vel_sq": swing_vel_sq,
            "wheel_vel_avg": wheel_vel_avg,
            "rw_vel": rw_vel,
            "lw_vel": lw_vel,
            "swing_target_theta": swing_target_theta,
            "swing_err_rw": swing_err_rw,
            "swing_err_lw": swing_err_lw,
            "terminated": terminated,
            "body_vertex_z": body_vertex_z,
            "prev_body_vx": local_linvel[:, 0],
            "body_vx": local_linvel[:, 0],
            "body_vy": local_linvel[:, 1],
            "upvector": upvector,
        })

        # 计算 reward（dispatch）—— ctx.linvel 传世界坐标系速度
        reward = self._compute_reward(
            state.info, world_linvel, gyro, upvector, dof_pos, dof_vel, base_pos, num_envs
        )

        tracking_lin_per_env = np.asarray(
            self._reward_tracking_lin_vel(
                RewardContext(
                    info=state.info, linvel=world_linvel, gyro=gyro,
                    dof_pos=dof_pos, dof_vel=dof_vel, num_envs=num_envs,
                    default_angles=np.asarray(self.default_angles, dtype=self._np_dtype),
                    tracking_sigma=rc.tracking_sigma,
                    tracking_sigma_lin=rc.tracking_sigma_lin,
                    tracking_sigma_ang=rc.tracking_sigma_ang,
                    base_height_target=state.info["commands"][:, 3],
                    base_height=base_pos[:, 2],
                    gravity=upvector,
                )
            ), dtype=self._np_dtype,
        )
        self._episode_tracking_reward += tracking_lin_per_env
        self._episode_vel_sum += np.abs(local_linvel[:, 0])
        self._episode_cmd_vel_sum += np.abs(commands[:, 0])

        self._debug_print(
            state.info, reward, commands, gyro, upvector, dof_vel,
            base_pos, world_linvel, local_linvel, tilt, body_vertex_z, torques, num_envs,
        )

        # 构建 obs
        obs = self._compute_obs(state.info, gyro, upvector, dof_pos, dof_vel, rw, lw, vx=local_linvel[:, 0])
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _debug_print(
        self, info, reward, commands, gyro, upvector, dof_vel,
        base_pos, world_linvel, local_linvel, tilt, body_vertex_z, torques, num_envs,
    ):
        steps = info.get("steps", np.zeros((num_envs,), dtype=np.uint32))
        if int(steps[0]) % 200 != 0:
            return

        tilt_raw = float(np.mean(np.sqrt(np.sum(np.square(upvector[:, :2]), axis=1))))
        base_z_mean = float(np.mean(base_pos[:, 2]))
        cos_pitch_raw = float(np.mean(np.cos(np.arcsin(np.clip(-upvector[:, 0], -1.0, 1.0)))))
        vx_abs = float(np.mean(np.abs(local_linvel[:, 0])))
        vx_mean = float(np.mean(local_linvel[:, 0]))
        vx_cmd_abs = float(np.mean(np.abs(commands[:, 0])))
        vx_cmd_mean = float(np.mean(commands[:, 0]))
        vz_raw = float(np.mean(np.abs(world_linvel[:, 2])))
        angv_raw = float(np.mean(np.sqrt(np.sum(np.square(gyro[:, :2]), axis=1))))
        angv_cmd_abs = float(np.mean(np.abs(commands[:, 2])))
        wheel_v_raw = float(np.mean(info["wheel_vel_avg"]))
        rw_vel_raw = float(np.mean(info["rw_vel"]))
        lw_vel_raw = float(np.mean(info["lw_vel"]))
        swing_vel_raw = float(np.rad2deg(np.mean(np.sqrt(info["swing_vel_sq"]))))
        rw_theta_deg = float(np.rad2deg(np.mean(info["swing_rw_theta"])))
        lw_theta_deg = float(np.rad2deg(np.mean(info["swing_lw_theta"])))
        swing_sym_deg = float(np.rad2deg(np.abs(np.mean(info["swing_rw_theta"] - info["swing_lw_theta"]))))

        fwd_mask = commands[:, 0] > 0.02
        bwd_mask = commands[:, 0] < -0.02

        swing_target = info["swing_target_theta"]
        swing_err_rw = info["swing_err_rw"]
        swing_err_lw = info["swing_err_lw"]

        all_cmd_vx = float(np.mean(commands[:, 0]))
        all_vx = float(np.mean(local_linvel[:, 0]))
        all_wheel_avg = float(np.mean(info["wheel_vel_avg"])) * 0.06
        all_rw_v = float(np.mean(info["rw_vel"])) * 0.06
        all_lw_v = float(np.mean(info["lw_vel"])) * 0.06
        all_cmd_ang = float(np.mean(commands[:, 2]))
        L = 0.4155
        all_tg_rw = all_cmd_vx + 0.5 * all_cmd_ang * L
        all_tg_lw = all_cmd_vx - 0.5 * all_cmd_ang * L
        all_theta_tg = float(np.rad2deg(np.mean(swing_target)))
        all_rw_err = float(np.rad2deg(np.mean(np.abs(swing_err_rw))))
        all_lw_err = float(np.rad2deg(np.mean(np.abs(swing_err_lw))))
        all_rw_theta = float(np.rad2deg(np.mean(info["swing_rw_theta"])))
        all_lw_theta = float(np.rad2deg(np.mean(info["swing_lw_theta"])))

        if fwd_mask.sum() > 0:
            fwd_cmd_vx = float(np.mean(commands[fwd_mask, 0]))
            fwd_cmd_ang = float(np.mean(commands[fwd_mask, 2])) 
            fwd_vx = float(np.mean(local_linvel[fwd_mask, 0]))
            fwd_wheel_avg = float(np.mean(info["wheel_vel_avg"][fwd_mask])) * 0.06
            fwd_rw_v = float(np.mean(info["rw_vel"][fwd_mask])) * 0.06
            fwd_lw_v = float(np.mean(info["lw_vel"][fwd_mask])) * 0.06
            fwd_tg_rw = fwd_cmd_vx + 0.5 * fwd_cmd_ang * L
            fwd_tg_lw = fwd_cmd_vx - 0.5 * fwd_cmd_ang * L
            fwd_theta_tg = float(np.rad2deg(np.mean(swing_target[fwd_mask])))
            fwd_rw_err = float(np.rad2deg(np.mean(np.abs(swing_err_rw[fwd_mask]))))
            fwd_lw_err = float(np.rad2deg(np.mean(np.abs(swing_err_lw[fwd_mask]))))
            fwd_rw_theta = float(np.rad2deg(np.mean(info["swing_rw_theta"][fwd_mask])))
            fwd_lw_theta = float(np.rad2deg(np.mean(info["swing_lw_theta"][fwd_mask])))
        else:
            fwd_cmd_vx = all_cmd_vx
            fwd_vx = all_vx
            fwd_cmd_ang = all_cmd_ang
            fwd_wheel_avg = all_wheel_avg
            fwd_rw_v = all_rw_v
            fwd_lw_v = all_lw_v
            fwd_tg_rw = all_tg_rw
            fwd_tg_lw = all_tg_lw
            fwd_theta_tg = all_theta_tg
            fwd_rw_err = all_rw_err
            fwd_lw_err = all_lw_err
            fwd_rw_theta = all_rw_theta
            fwd_lw_theta = all_lw_theta

        if bwd_mask.sum() > 0:
            bwd_cmd_vx = float(np.mean(commands[bwd_mask, 0]))
            bwd_cmd_ang = float(np.mean(commands[bwd_mask, 2])) 
            bwd_vx = float(np.mean(local_linvel[bwd_mask, 0]))
            bwd_wheel_avg = float(np.mean(info["wheel_vel_avg"][bwd_mask])) * 0.06
            bwd_rw_v = float(np.mean(info["rw_vel"][bwd_mask])) * 0.06
            bwd_lw_v = float(np.mean(info["lw_vel"][bwd_mask])) * 0.06
            bwd_tg_rw = bwd_cmd_vx + 0.5 * bwd_cmd_ang * L           # ← 新增
            bwd_tg_lw = bwd_cmd_vx - 0.5 * bwd_cmd_ang * L           # ← 新增
            bwd_theta_tg = float(np.rad2deg(np.mean(swing_target[bwd_mask])))
            bwd_rw_err = float(np.rad2deg(np.mean(np.abs(swing_err_rw[bwd_mask]))))
            bwd_lw_err = float(np.rad2deg(np.mean(np.abs(swing_err_lw[bwd_mask]))))
            bwd_rw_theta = float(np.rad2deg(np.mean(info["swing_rw_theta"][bwd_mask])))
            bwd_lw_theta = float(np.rad2deg(np.mean(info["swing_lw_theta"][bwd_mask])))
        else:
            bwd_cmd_vx = all_cmd_vx
            bwd_cmd_ang = all_cmd_ang
            bwd_vx = all_vx
            bwd_wheel_avg = all_wheel_avg
            bwd_rw_v = all_rw_v
            bwd_lw_v = all_lw_v
            bwd_tg_rw = all_tg_rw
            bwd_tg_lw = all_tg_lw
            bwd_theta_tg = all_theta_tg
            bwd_rw_err = all_rw_err
            bwd_lw_err = all_lw_err
            bwd_rw_theta = all_rw_theta
            bwd_lw_theta = all_lw_theta

        print(f"[state] total_reward=" f"{float(np.mean(reward)):+.4f}")
        pitch_all = np.rad2deg(np.arcsin(np.clip(-upvector[:, 0], -1.0, 1.0)))
        roll_all  = np.rad2deg(np.arcsin(np.clip( upvector[:, 1], -1.0, 1.0)))
        pitch_mean = float(np.mean(pitch_all))
        roll_mean  = float(np.mean(roll_all))
        pitch_abs  = float(np.mean(np.abs(pitch_all)))
        roll_abs   = float(np.mean(np.abs(roll_all)))
        h_tg_mean = float(np.mean(commands[:, 3]))
        print(
            f"  base_z=" f"{base_z_mean:.3f}" f"(tg:{h_tg_mean:.3f})"
            f"  tilt=" f"{np.rad2deg(tilt.mean()):+.1f}°"
            f"  pitch=" f"{pitch_mean:+.1f}°" f"(|{pitch_abs:.1f}°)"
            f"  roll=" f"{roll_mean:+.1f}°" f"(|{roll_abs:.1f}°)"
            f"  vx=" f"{vx_mean:+.3f}" f"(tg:{vx_cmd_mean:+.3f})"
            f"  |vx|=" f"{vx_abs:.3f}" f"(tg:{vx_cmd_abs:.3f})"
            f"  cos_pitch=" f"{cos_pitch_raw:.3f}"
            )
        policy_rw_raw = float(np.mean(info["current_actions"][:, 2]))
        policy_lw_raw = float(np.mean(info["current_actions"][:, 5]))
        policy_rw_tg_v = policy_rw_raw * self._cfg.control_config.wheel_action_scale * 0.06
        policy_lw_tg_v = policy_lw_raw * self._cfg.control_config.wheel_action_scale * 0.06

        print(f"  vz="f"{vz_raw:.3f}"
              f"  w="f"{angv_raw:.3f}"f"(tg:{angv_cmd_abs:.3f})"
              f"  wheel_avg="f"{wheel_v_raw * 0.06:+.3f}m/s"
              f"  rw_v="f"{rw_vel_raw * 0.06:+.3f}m/s"f"  lw_v="f"{lw_vel_raw * 0.06:+.3f}m/s")
        print(f"  policy_rw="f"{policy_rw_raw:+.3f}"f"(→v="f"{policy_rw_tg_v:+.3f}m/s)"
              f"  policy_lw="f"{policy_lw_raw:+.3f}"f"(→v="f"{policy_lw_tg_v:+.3f}m/s)")
        print(f"  rw_θ="f"{rw_theta_deg:+.1f}°"
              f"  lw_θ="f"{lw_theta_deg:+.1f}°"
              f"  θ_tg="f"{float(np.rad2deg(np.mean(info['swing_target_theta']))):+.1f}°"
              f"  |err_rw|="f"{float(np.rad2deg(np.mean(np.abs(info['swing_err_rw'])))):+.1f}°"
              f"  |err_lw|="f"{float(np.rad2deg(np.mean(np.abs(info['swing_err_lw'])))):+.1f}°"
              f"  |Δθ|="f"{swing_sym_deg:+.1f}°"
              f"  θ_vel="f"{swing_vel_raw:+.1f}°/s"
              f"  step="f"{int(self._episode_steps.mean())}"
              f"  vertex="f"{float(np.mean(body_vertex_z)):+.3f}")
        if fwd_mask.sum() > 0:
            fwd_policy_rw = float(np.mean(info['current_actions'][fwd_mask, 2]))
            fwd_policy_lw = float(np.mean(info['current_actions'][fwd_mask, 5]))
        else:
            fwd_policy_rw = policy_rw_raw
            fwd_policy_lw = policy_lw_raw
        if bwd_mask.sum() > 0:
            bwd_policy_rw = float(np.mean(info['current_actions'][bwd_mask, 2]))
            bwd_policy_lw = float(np.mean(info['current_actions'][bwd_mask, 5]))
        else:
            bwd_policy_rw = policy_rw_raw
            bwd_policy_lw = policy_lw_raw

        print(f"  前进→ cmd_vx="f"{fwd_cmd_vx:+.2f}m/s"
              f"  vx="f"{fwd_vx:+.2f}m/s"
              f"  wheel="f"{fwd_wheel_avg:+.2f}m/s(rw:{fwd_rw_v:+.2f}/lw:{fwd_lw_v:+.2f} tg_rw:{fwd_tg_rw:+.2f}/tg_lw:{fwd_tg_lw:+.2f})"              f"  policy_rw="f"{fwd_policy_rw:+.3f}"
              f"  policy_lw="f"{fwd_policy_lw:+.3f}"
              f"  θ_tg="f"{fwd_theta_tg:+.1f}°"
              f"  rw_θ="f"{fwd_rw_theta:+.1f}°"
              f"  lw_θ="f"{fwd_lw_theta:+.1f}°"
              f"  err_rw="f"{fwd_rw_err:+.1f}°"
              f"  err_lw="f"{fwd_lw_err:+.1f}°")
        print(f"  后退→ cmd_vx="f"{bwd_cmd_vx:+.2f}m/s"
              f"  vx="f"{bwd_vx:+.2f}m/s"
              f"  wheel="f"{bwd_wheel_avg:+.2f}m/s(rw:{bwd_rw_v:+.2f}/lw:{bwd_lw_v:+.2f} tg_rw:{bwd_tg_rw:+.2f}/tg_lw:{bwd_tg_lw:+.2f})"              f"  policy_rw="f"{bwd_policy_rw:+.3f}"
              f"  policy_lw="f"{bwd_policy_lw:+.3f}"
              f"  θ_tg="f"{bwd_theta_tg:+.1f}°"
              f"  rw_θ="f"{bwd_rw_theta:+.1f}°"
              f"  lw_θ="f"{bwd_lw_theta:+.1f}°"
              f"  err_rw="f"{bwd_rw_err:+.1f}°"
              f"  err_lw="f"{bwd_lw_err:+.1f}°")
        cur_theta_avg = (info["swing_rw_theta"] + info["swing_lw_theta"]) * 0.5
        if not hasattr(self, "_last_theta_for_debug"):
            self._last_theta_for_debug = cur_theta_avg.copy()
        dtheta_dt = (cur_theta_avg - self._last_theta_for_debug) / 0.02
        self._last_theta_for_debug = cur_theta_avg.copy()

        speed_err = commands[:, 0] - local_linvel[:, 0]
        avg_swing_err = cur_theta_avg - info["swing_target_theta"]
        swing_range_gate = np.clip(1.0 - np.abs(avg_swing_err) / np.deg2rad(10.0), 0.0, 1.0)
        direction_ok = (speed_err * dtheta_dt) > 0
        dir_ratio = float(np.mean(direction_ok.astype(np.float32)))
        target_dtheta = np.clip(np.abs(speed_err) * np.deg2rad(30.0), np.deg2rad(5.0), np.deg2rad(60.0))
        amount = np.clip(np.abs(dtheta_dt) / target_dtheta, 0.0, 1.0)

        posture_gate = float(np.mean(info.get("linvel_gate", np.ones(num_envs))))
        g_tilt_val = float(np.mean(info.get("linvel_g_tilt", np.ones(num_envs))))
        g_swing_val = float(np.mean(info.get("linvel_g_swing", np.ones(num_envs))))
        g_height_val = float(np.mean(info.get("linvel_g_height", np.ones(num_envs))))
        in_deadzone = np.abs(speed_err) < 0.15
        
        core = np.where(in_deadzone, 0.0, np.where(direction_ok, amount * swing_range_gate, 0.0))
        score = core * posture_gate

        print(
            f"  dθ/dt="f"{float(np.rad2deg(np.mean(dtheta_dt))):+.1f}°/s"
            f"  tg_dθ="f"{float(np.rad2deg(np.mean(target_dtheta))):+.1f}°/s"
            f"  amount="f"{float(np.mean(amount)):.3f}"
            f"  dir_ok="f"{dir_ratio:.2f}"
            f"  swing_gate="f"{float(np.mean(swing_range_gate)):.3f}"
            f"  posture_gate="f"{posture_gate:.3f}"f"(g_t={g_tilt_val:.3f}/g_s={g_swing_val:.3f}/g_h={g_height_val:.3f})"
            f"  |swing_err|="f"{float(np.rad2deg(np.mean(np.abs(avg_swing_err)))):+.1f}°"
            f"  score="f"{float(np.mean(score)):.4f}"
        )

        print(f"  torque="f"{float(np.sqrt(np.mean(np.sum(np.square(torques), axis=1)))):+.3f}"
              f"  rw_torque="f"{float(np.mean(np.abs(torques[:, 2]))):+.3f}N·m"
              f"  lw_torque="f"{float(np.mean(np.abs(torques[:, 5]))):+.3f}N·m"
              f"  r_leg_torque="f"{float(np.mean(np.abs(torques[:, 0]))):+.3f}/"f"{float(np.mean(np.abs(torques[:, 1]))):+.3f}"
              f"  l_leg_torque="f"{float(np.mean(np.abs(torques[:, 3]))):+.3f}/"f"{float(np.mean(np.abs(torques[:, 4]))):+.3f}"
              f"  term_rate="f"{(self._term_vertex + self._term_low + self._term_tilt) / max(self._term_window_steps, 1) * 1000:.1f}/1000step"
              f"  hit(touch/low/tilt)/1000step="f"{self._hit_vertex / max(self._term_window_steps, 1) * 1000:.0f}/{self._hit_low / max(self._term_window_steps, 1) * 1000:.0f}/{self._hit_tilt / max(self._term_window_steps, 1) * 1000:.0f}"
              f"  body_vertex_z mean={float(np.mean(body_vertex_z)):.4f}m  min={float(np.min(body_vertex_z)):.4f}m  max={float(np.max(body_vertex_z)):.4f}m  base_z_mean={float(np.mean(base_pos[:, 2])):.4f}m"
              f"  total_steps="f"{self._total_steps}"
              f"  cmd_vx_range=["f"{float(self._cmd_vel_range[0, 0]):+.2f}, "f"{float(self._cmd_vel_range[0, 1]):+.2f}]"
              )


registry.register_env("Cod2026JoystickFlat", Cod2026JoystickEnv, sim_backend="motrix")