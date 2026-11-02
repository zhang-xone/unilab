from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import mujoco
import time

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_quat_apply, np_yaw_to_quat, np_yaw_from_quat, np_wrap_to_pi
from unilab.dr import DomainRandomizationCapabilities, ResetPlan, IntervalRandomizationPlan, ResetRandomizationPayload
from unilab.dr.dr_utils import build_interval_push_plan, validate_interval_push_support, zero_actions

from unilab.envs.locomotion.common.commands import (
    Commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.curriculum import CurriculumManager
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common import rewards as common_rewards
from unilab.envs.locomotion.common.rewards import (
    RewardContext,
    run_reward_dispatch,
)
from unilab.envs.locomotion.common.height_scan import (
    HeightScanConfig,
    base_height_from_scan,
    combined_base_height_from_scan,
    height_scan_obs,
    init_height_scan_sensor,
    terrain_out_of_bounds,
)
from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)
from unilab.terrains import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    box_pyramid_stairs,
    pyramid_stairs,
    pyramid_stairs_inv,
    box_pyramid_stairs_inv,
    box_wave,
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    random_rough,
    wave_terrain,
)

from unilab.envs.locomotion.balance_infantry.base import (
    JOINT_LIMITS,
    NUM_BALANCE_ACTIONS,
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
    WHEEL_JOINT_INDICES,
    WHEEL_RADIUS,
    WHEELBASE,
    BalanceInfantryBaseCfg,
    BalanceInfantryBaseEnv,
    WheelTorqueEnvelope,
    forward_kinematics_fivebar,
    vmc_feedforward_torques_no_ramp,
    vmc_mapping_fivebar,
    compute_vmc_torques_fivebar,
    compute_balance_motor_ctrl,
)


# =====================================================================
# 配置类
# =====================================================================


@dataclass
class BalanceInfantryJoystickSensor:
    gyro: str = "gyro"
    accel: str = "accel"
    local_linvel: str = "local_linvel"
    upvector: str = "upvector"
    base_pos: str = "base_pos"
    right_wheel: str = "right_wheel_pos"
    left_wheel: str = "left_wheel_pos"
    left_wheel_contact_force: str = "left_wheel_contact_force"
    right_wheel_contact_force: str = "right_wheel_contact_force"
    base_body_contact_force: str = "base_body_contact_force"
    base_center_contact_force: str = "base_center_contact_force"
    left_front_leg_contact_force: str = "left_front_leg_contact_force"
    left_front_child1_leg_contact_force: str = "left_front_child1_leg_contact_force"
    left_front_leg1_contact_force: str = "left_front_leg1_contact_force"
    left_rear_child1_leg2_contact_force: str = "left_rear_child1_leg2_contact_force"
    left_rear_leg_contact_force: str = "left_rear_leg_contact_force"
    left_rear_leg2_contact_force: str = "left_rear_leg2_contact_force"
    left_rear_child1_leg1_contact_force: str = "left_rear_child1_leg1_contact_force"
    right_front_leg_contact_force: str = "right_front_leg_contact_force"
    right_front_child1_leg_contact_force: str = "right_front_child1_leg_contact_force"
    right_front_leg1_contact_force: str = "right_front_leg1_contact_force"
    right_rear_child1_leg2_contact_force: str = "right_rear_child1_leg2_contact_force"
    right_rear_leg_contact_force: str = "right_rear_leg_contact_force"
    right_rear_leg2_contact_force: str = "right_rear_leg2_contact_force"
    right_rear_child1_leg1_contact_force: str = "right_rear_child1_leg1_contact_force"


@dataclass
class BalanceRewardConfig:
    """Balance 奖励配置 —— 参考 go2w/go2/g1 的标准设计：scales + fns(ctx) + dispatch。

    设计原则：
      - 量级差 < 1000 倍（避免 base_height=-3000 这种独大）
      - 高度 / 姿态用 square(z_err) 线性惩罚（标准形式，不是高斯）
      - 速度追踪用标准高斯 exp(-err²/sigma²)
      - 无离散脉冲惩罚（termination）
      - alive 给小正分替代"别死"的激励
    """

    clip_range: float = 0.0
    clip_ranges: dict[str, float] = field(default_factory=dict)

    scales: dict[str, float] = field(
        default_factory=lambda: {
            # ── 主任务（正奖励，量级 1~5） ──────────────────────
            "tracking_lin_vel": 0.0,
            "tracking_ang_vel": 0.0,
            # ── 姿态 / 高度（惩罚，量级 10~200） ───────────────────
            "roll": 0.0,
            "pitch": 0.0,
            "base_height": 0.0,
            "lin_vel_z": 0.0,
            "ang_vel_xy": 0.0,

            # ── Balance 特有：腿 / 对称（惩罚） ───────────────────
            "swing_vertical": 0.0,
            "swing_symmetry": 0.0,
            "swing_vel": 0.0,

            # ── 平滑 / 能耗（小惩罚，量级 1e-5~1e-3） ─────────────
            "action_rate": 0.0,
            "dof_torques_l2": 0.0,

            # ── 经验性小惩罚 ─────────────────────────────────────

            # ── 站立姿态保持（低速/姿态恶化时启用） ────────────────
            "stand": 0.0,
            # ── 碰撞惩罚（机体接触地面） ────────────────────────
            "collision": 0.0,
        }
    )
    base_height_target: float = 0.31
    base_height_sigma: float = 0.05      # 高度奖励的 σ
    leg_length_sigma: float = 0.02      # 腿长奖励的 σ
    tilt_precise_sigma: float = 0.07    # tilt 精确奖励的 σ（4°）
    tracking_sigma: float = 0.25
    tracking_sigma_lin: float = 0.2
    tracking_sigma_ang: float = 0.1
    tracking_sigma_yaw: float = 0.55
    tracking_sigma_dynamic: bool = True   # True: σ 随 cmd² 动态放大; False: 固定 σ²
    wheel_power_limit: float | None = None  # 轮子功率上限(W)，超限按差值扣分；None 则用 torque 惩罚
    wheel_torque_penalty_start: float = 2.5    # 轮子力矩惩罚起始值（N·m），低于此值不惩罚
    wheel_torque_penalty_max: float = 3.5      # 轮子力矩惩罚最大值（N·m），ratio=1
    wheel_air_power_max: float = 100.0         # 空中轮子功率归一化上限（W），penalty=(P/Pmax)²
    # ── 分层渐进式姿态终止 ──
    tilt_limit_tight: float = 20.0 * np.pi / 180.0
    tilt_limit_loose: float = 60.0 * np.pi / 180.0  # 后期宽松倾角
    swing_angle_limit: float = 45.0 * np.pi / 180.0  # 摆角 θ 过大终止
    swing_angle_limit_loose: float = 90.0 * np.pi / 180.0  # 后期宽松摆角
    min_height_tight: float = 0.18          # 高度终止线（m）
    min_height_loose: float = 0.14          # 后期宽松高度线（m）
    # ── 课程完成后的终止收敛：完成后只剩 tilt（翻车）终止 ──
    # 届时关闭 vertex_touch / swing / height 三种姿态终止，只保留 tilt > tilt_limit_tight
    # 与地形出界（防止掉出地形后无限下坠）。tilt_limit_tight 本身不变。
    term_loosen_enabled: bool = False
    term_loosen_from_step: int = 0           # 生效的控制步（每 iter = num_steps_per_env 步）；0=立即
    air_spawn_max_height: float = 0.40       # 空中出生最大高度（m）
    air_spawn_probability: float = 0.3       # 空中出生概率，其余从地面出生
    ground_spawn_max_height: float = 0.10    # 地面出生最大高度偏移（m）
    n_steps_delay: int = 50
    powerless_duration: float = 1.0  # reset 后无力时长（秒），期间扭矩=0/奖励=0/不终止
    powerless_curriculum_threshold: float = 0.125  # curriculum progress 超过此值才启用无力期（2400步≈100iter）
    powerless_probability: float = 0.2  # 每个 reset 随机启用无力期的概率
    # orientation_stand 的启用阈值
    stand_cmd_linvel_threshold: float = 0.1  # |cmd_vx| < 此值 → 认为应该站立
    stand_cmd_angvel_threshold: float = 0.1  # |cmd_yaw| < 此值 → 认为应该站立
    stand_tilt_threshold: float = 7.5 * np.pi / 180.0  # tilt > 此值 → 姿态已恶化
    stand_vel_weight: float = 1.0  # 站立惩罚中水平速度项权重
    stand_angvel_weight: float = 1.0  # 站立惩罚中角速度项权重
    stand_linvelz_weight: float = 1.0  # 站立惩罚中垂直速度项权重
    # 原地旋转模式：|cmdvyaw| > 2.0 时自动进入（vx=0，开启 stand 惩罚限制水平移动）
    # ── 三区域 spawn 概率（坡/粗糙/平地，三者之和不应超过 1.0）──
    slope_spawn_prob: float = 0.4   # 斜坡圈 spawn 概率
    rough_spawn_prob: float = 0.15  # 远处粗糙地形 spawn 概率（hfield 边缘）
    # 平地概率 = 1 - slope_spawn_prob - rough_spawn_prob
    body_corner_clearance_threshold: float = 0.08  # 角点离地 < 此值（m）开始惩罚
    


@dataclass(kw_only=True)
class BalanceRoughTerrainCfg(TerrainGeneratorCfg):
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 10
    num_cols: int = 10
    border_width: float = 1.0
    add_lights: bool = True
    horizontal_scale: float = 0.2

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "flat": flat(proportion=0.1),
            "random_rough": random_rough(
                proportion=0.1,
                noise_range=(0.02, 0.10),
                noise_step=0.005,
                border_width=0.2,
                downsampled_scale=0.2,
            ),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.15,
                slope_range=(0.0, 0.40),
                platform_width=2.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.15,
                slope_range=(0.0, 0.40),
                platform_width=2.0,
                border_width=0.2,
            ),
            "pyramid_stairs_inv": box_pyramid_stairs_inv(
                proportion=0.2,
                step_height_range=(0.02, 0.22),
                step_width=0.6,
                platform_width=2.0,
                border_width=0.2,
            ),
            "pyramid_stairs": box_pyramid_stairs(
                proportion=0.2,
                step_height_range=(0.02, 0.22),
                step_width=0.6,
                platform_width=2.0,
                border_width=0.2,
            ),
            }
    )

    def __post_init__(self):
        self.num_cols = len(self.sub_terrains)


@dataclass
class RoughTerminationConfig:
    terrain_out_of_bounds: bool = True
    terrain_distance_buffer: float = 3.0


@registry.envcfg("BalanceInfantryJoystickFlat")
@dataclass
class BalanceInfantryJoystickCfg(BalanceInfantryBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "balance_infantry" / "balance_infantry.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "balance_infantry" / "balance_infantry_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=BalanceRoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    max_episode_seconds: float = 20.0
    sensor: BalanceInfantryJoystickSensor = field(default_factory=BalanceInfantryJoystickSensor)
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
            push_body_name="base_link",
            curriculum_enabled=True,
            curriculum_warmup_steps=2_000,
            curriculum_full_steps=500_000,
            curriculum_initial_scale=0.05,
            curriculum_shape="smooth",
            randomize_body_mass=True,
            body_mass_multiplier_range=[0.9, 1.1],
            randomize_joint_friction=True,
            randomize_dof_armature=True,
        )
    )
    reward_config: BalanceRewardConfig = field(default_factory=BalanceRewardConfig)
    terrain_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    # Reset 时机器人放置位置的比例：periphery_ratio 为放置在外围的比例，
    # 1 - periphery_ratio 为放置在中心的比例。默认 0.5 表示各一半。
    spawn_periphery_ratio: float = 0.5
    # 调试开关：开启后每 200 步打印详细状态，并存储中间变量
    debug_enabled: bool = False


# =====================================================================
# 领域随机化 Provider
# =====================================================================


def build_balance_backend_reset_randomization(
    env: Any, num_reset: int
) -> ResetRandomizationPayload | None:
    """Build reset DR payload for Balance.

    kp/kd are intentionally excluded -- they are sampled and applied
    inside the owner pre-step motor control via info_updates.
    """
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None:
        return None

    payload = ResetRandomizationPayload()
    if getattr(domain_rand, "randomize_base_mass", False):
        low, high = domain_rand.added_mass_range
        payload.base_mass_delta = np.random.uniform(low, high, size=(num_reset,))

    if getattr(domain_rand, "randomize_body_mass", False):
        base_body_mass = env._backend.get_body_mass()
        body_mass_template = np.asarray(base_body_mass, dtype=np.float64)
        low, high = domain_rand.body_mass_multiplier_range
        multipliers = np.ones((num_reset, body_mass_template.size), dtype=np.float64)
        leg_body_ids = env._backend.get_body_ids([
            "left_front_link", "left_front_child1_link", "left_front_child2_link", "left_front_child3_link",
            "left_rear_link", "left_rear_child1_link", "left_wheel_link",
            "right_front_link", "right_front_child1_link", "right_front_child2_link", "right_front_child3_link",
            "right_rear_link", "right_rear_child1_link", "right_wheel_link",
        ])
        multipliers[:, leg_body_ids] = np.random.uniform(low, high, size=(num_reset, len(leg_body_ids)))
        body_mass = np.broadcast_to(body_mass_template, multipliers.shape).copy()
        body_mass *= multipliers
        payload.body_mass = body_mass

    if getattr(domain_rand, "random_com", False):
        base_com_offset = np.zeros((num_reset, 3), dtype=np.float64)
        low, high = domain_rand.com_offset_x
        base_com_offset[:, 0] = np.random.uniform(low, high, size=(num_reset,))
        com_offset_y = getattr(domain_rand, "com_offset_y", None)
        if com_offset_y is not None:
            low, high = com_offset_y
            base_com_offset[:, 1] = np.random.uniform(low, high, size=(num_reset,))
        com_offset_z = getattr(domain_rand, "com_offset_z", None)
        if com_offset_z is not None:
            low, high = com_offset_z
            base_com_offset[:, 2] = np.random.uniform(low, high, size=(num_reset,))
        payload.base_com_offset = base_com_offset

    randomize_ground = bool(getattr(domain_rand, "randomize_ground_friction", False))
    randomize_wheel_ground = bool(getattr(domain_rand, "randomize_wheel_ground_friction", False))
    if randomize_ground or randomize_wheel_ground:
        base_geom_friction = env._backend.get_geom_friction()
        terrain_cfg = getattr(env._cfg.scene, "terrain", None)
        terrain_geom_name = terrain_cfg.geom_name if terrain_cfg is not None else "floor"
        ground_geom_id = env._backend.get_geom_id(terrain_geom_name)
        if base_geom_friction is None:
            raise ValueError("ground friction randomization requires a cached geom-friction table")
        if ground_geom_id is None:
            raise ValueError("ground friction randomization requires a ground geom id")
        geom_friction_template = np.asarray(base_geom_friction, dtype=np.float64)
        geom_friction = np.broadcast_to(
            geom_friction_template, (num_reset, *geom_friction_template.shape)
        ).copy()
        if randomize_wheel_ground:
            low, high = domain_rand.wheel_ground_friction_range
            wheel_ground_friction = np.random.uniform(low=low, high=high, size=(num_reset,))
            for wheel_geom_name in ("left_wheel_geom", "right_wheel_geom"):
                try:
                    wheel_geom_id = env._backend.get_geom_id(wheel_geom_name)
                except ValueError as exc:
                    raise ValueError(
                        f"wheel-ground friction randomization requires geom '{wheel_geom_name}'"
                    ) from exc
                geom_friction[:, wheel_geom_id, 0] = wheel_ground_friction
            geom_friction[:, ground_geom_id, 0] = wheel_ground_friction
        else:
            low, high = domain_rand.ground_friction_multiplier_range
            ground_friction_scale = np.random.uniform(low=low, high=high, size=(num_reset,))
            geom_friction[:, ground_geom_id, 0] = geom_friction_template[ground_geom_id, 0] * ground_friction_scale
            geom_friction[:, ground_geom_id, 1] = geom_friction_template[ground_geom_id, 1] * ground_friction_scale
        payload.geom_friction = geom_friction

    if getattr(domain_rand, "randomize_geom_solref", False):
        base_geom_solref = env._backend.get_geom_solref()
        terrain_geom_name = env._cfg.scene.terrain.geom_name if env._cfg.scene.terrain is not None else "floor"
        ground_geom_id = env._backend.get_geom_id(terrain_geom_name)
        if base_geom_solref is None:
            raise ValueError("geom solref randomization requires a cached geom-solref table")
        if ground_geom_id is None:
            raise ValueError("geom solref randomization requires a ground geom id")
        geom_solref_template = np.asarray(base_geom_solref, dtype=np.float64)
        low_tc, high_tc = domain_rand.geom_solref_timeconst_range
        low_dr, high_dr = domain_rand.geom_solref_dampratio_range
        geom_solref = np.broadcast_to(
            geom_solref_template, (num_reset, *geom_solref_template.shape)
        ).copy()
        geom_solref[:, ground_geom_id, 0] = np.random.uniform(
            low=low_tc, high=high_tc, size=(num_reset,)
        )
        geom_solref[:, ground_geom_id, 1] = np.random.uniform(
            low=low_dr, high=high_dr, size=(num_reset,)
        )
        payload.geom_solref = geom_solref

    if getattr(domain_rand, "randomize_inertia", False):
        base_inertia = np.asarray(env._backend._model.body_inertia, dtype=np.float64)
        low, high = domain_rand.inertia_multiplier_range
        mass_scale = np.random.uniform(low, high, size=(num_reset, base_inertia.shape[0]))
        body_inertia = np.broadcast_to(base_inertia, (num_reset, *base_inertia.shape)).copy()
        body_inertia *= mass_scale[:, :, np.newaxis]
        payload.body_inertia = body_inertia

        # 与复旦一致：同一个 body 的 mass 和 inertia 用同一个 scale。
        # base 附加质量先加、再整体乘 scale；随后清掉 base_mass_delta，
        # 避免 backend 在 body_mass 上再叠加一次。
        if payload.body_mass is not None:
            body_mass = np.asarray(payload.body_mass, dtype=np.float64).copy()
        else:
            body_mass = np.broadcast_to(
                np.asarray(env._backend.get_body_mass(), dtype=np.float64),
                (num_reset, mass_scale.shape[1]),
            ).copy()
        if payload.base_mass_delta is not None:
            base_body_id = env._backend.get_body_ids([env._cfg.asset.base_name])[0]
            body_mass[:, base_body_id] += np.asarray(payload.base_mass_delta, dtype=np.float64)
            payload.base_mass_delta = None
        body_mass *= mass_scale
        payload.body_mass = body_mass

    if getattr(domain_rand, "randomize_dof_armature", False):
        base_armature = env._backend.get_dof_armature().astype(np.float64)
        dof_armature = np.broadcast_to(base_armature, (num_reset, base_armature.size)).copy()

        leg_main_joints = ["left_front_joint", "left_rear_joint", "right_front_joint", "right_rear_joint"]
        wheel_joints = ["left_wheel_joint", "right_wheel_joint"]
        passive_joints = [
            "left_front_child1_joint", "left_front_child2_joint", "left_front_child3_joint",
            "left_rear_child1_joint",
            "right_front_child1_joint", "right_front_child2_joint", "right_front_child3_joint",
            "right_rear_child1_joint",
        ]

        # dof_armature_multiplier_range 作为所有关节组的默认值；
        # 显式配置 leg_main/wheel/passive_* 时再覆盖。
        armature_default = getattr(domain_rand, "dof_armature_multiplier_range", [0.8, 1.2])
        for joint_names, multiplier_range in [
            (leg_main_joints, getattr(domain_rand, "leg_main_armature_multiplier_range", armature_default)),
            (wheel_joints, getattr(domain_rand, "wheel_armature_multiplier_range", armature_default)),
            (passive_joints, getattr(domain_rand, "passive_armature_multiplier_range", armature_default)),
        ]:
            low, high = multiplier_range
            dof_ids = env._backend.get_joint_dof_indices(joint_names)
            dof_armature[:, dof_ids] *= np.random.uniform(low, high, size=(num_reset, len(dof_ids)))

        payload.dof_armature = dof_armature

    if getattr(domain_rand, "randomize_joint_friction", False):
        base_damping = env._backend.get_dof_damping().astype(np.float64)
        base_frictionloss = env._backend.get_dof_frictionloss().astype(np.float64)
        dof_damping = np.broadcast_to(base_damping, (num_reset, base_damping.size)).copy()
        dof_frictionloss = np.broadcast_to(base_frictionloss, (num_reset, base_frictionloss.size)).copy()

        leg_main_joints = ["left_front_joint", "left_rear_joint", "right_front_joint", "right_rear_joint"]
        wheel_joints = ["left_wheel_joint", "right_wheel_joint"]
        passive_joints = [
            "left_front_child1_joint", "left_front_child2_joint", "left_front_child3_joint",
            "left_rear_child1_joint",
            "right_front_child1_joint", "right_front_child2_joint", "right_front_child3_joint",
            "right_rear_child1_joint",
        ]

        damp_mult = getattr(domain_rand, "damping_multiplier_range", [0.1, 10.0])
        fric_mult = getattr(domain_rand, "frictionloss_multiplier_range", [0.1, 10.0])
        for joint_names in (leg_main_joints, wheel_joints, passive_joints):
            dof_ids = env._backend.get_joint_dof_indices(joint_names)
            dof_damping[:, dof_ids] *= np.random.uniform(damp_mult[0], damp_mult[1],
                                                          size=(num_reset, len(dof_ids)))
            dof_frictionloss[:, dof_ids] *= np.random.uniform(fric_mult[0], fric_mult[1],
                                                               size=(num_reset, len(dof_ids)))

        payload.dof_damping = dof_damping
        payload.dof_frictionloss = dof_frictionloss

    return None if payload.is_empty() else payload


# =====================================================================
# 地形相关 cmdvx 工具函数
# =====================================================================


def get_terrain_cmdvx_limits(terrain_types: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """根据地形类型返回 cmdvx 的上下限。

    地形 → cmdvx 范围:
      - 台阶 (pyramid_stairs / pyramid_stairs_inv) 与高台 (plateau): [-1.5, 1.5]
      - 平地 (flat): [-2.8, 2.8]
      - 坡 / 波浪 (hf_pyramid_slope / hf_pyramid_slope_inv / wave_terrain): [-2.8, 2.8]
      - 粗糙 (random_rough): [-1.5, 1.5]
      - 其他: 默认 [-3.0, 3.0]（= ``commands.vel_limit``，即不收紧）

    这些值是 **上限**，只会收紧课程范围、不会放宽；经过本函数后实际用于
    采样的边界由 :func:`effective_cmd_vx_bounds` 计算，``[info] vx_range``
    打印的就是它（地形夹紧后的真实范围），``vx_cfg`` 才是课程名义范围。
    """
    n = len(terrain_types)
    dtype = get_global_dtype()
    vx_low = np.full(n, -3.0, dtype=dtype)
    vx_high = np.full(n, 3.0, dtype=dtype)

    for i, t in enumerate(terrain_types):
        if t in ("pyramid_stairs", "pyramid_stairs_inv", "plateau"):
            vx_low[i] = -1.7
            vx_high[i] = 1.7
        elif t in ("flat"):
            vx_low[i] = -2.5
            vx_high[i] = 2.5
        elif t in ("random_rough"):
            vx_low[i] = -1.7
            vx_high[i] = 1.7
        elif t in ("hf_pyramid_slope", "hf_pyramid_slope_inv", "wave_terrain"):
            vx_low[i] = -2.5
            vx_high[i] = 2.5
        # 其他地形保持默认 [-3.0, 3.0]

    return vx_low, vx_high


def effective_cmd_vx_bounds(
    cur_low: np.ndarray,
    cur_high: np.ndarray,
    terrain_types: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """把 per-env 的课程范围与地形限制求交，得到真正用于采样的 vx 边界。

    地形限制只会收紧课程范围（``max(low, t_low)`` / ``min(high, t_high)``），
    所以课程范围可以比实际能采到的命令更大。诊断打印必须用本函数的输出，
    否则看到的是"名义范围"而不是真实范围。
    """
    dtype = get_global_dtype()
    lo = np.array(cur_low, dtype=dtype, copy=True)
    hi = np.array(cur_high, dtype=dtype, copy=True)
    if terrain_types is not None:
        t_low, t_high = get_terrain_cmdvx_limits(terrain_types)
        np.maximum(lo, t_low, out=lo)
        np.minimum(hi, t_high, out=hi)
    return lo, hi


# 命令模式占比 (前进, 转向, 原地自转)。
# 注意实测"自转"占比 = rotate×P(|yaw|>3) + turn×P(|yaw|>3)：|yaw|>3 的转向样本
# 也会被置成 vx=0，所以自转的真实占比高于 rotate 本身。
CMD_MODE_PROBS_DEFAULT = (0.65, 0.20, 0.15)
CMD_MODE_PROBS_STAIRS = (0.75, 0.175, 0.075)


def get_terrain_height_params(
    terrain_types: list[str],
    terrain_difficulty: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(terrain_types)
    dtype = get_global_dtype()
    use_uniform = np.ones(n, dtype=bool)
    uni_low = np.full(n, 0.16, dtype=dtype)
    uni_high = np.full(n, 0.42, dtype=dtype)
    beta_a = np.zeros(n, dtype=dtype)
    beta_b = np.zeros(n, dtype=dtype)

    for i, t in enumerate(terrain_types):
        if t == "pyramid_stairs_inv":
            diff = float(terrain_difficulty[i]) if terrain_difficulty is not None else 0.0
            use_uniform[i] = True
            uni_low[i] = 0.18 + diff * 0.20
            uni_high[i] = 0.38
        elif t == "plateau":
            use_uniform[i] = False
            beta_a[i] = 3.0
            beta_b[i] = 1.0

    return use_uniform, uni_low, uni_high, np.stack([beta_a, beta_b], axis=-1)


def get_terrain_plateau_mask(terrain_types: list[str]) -> np.ndarray:
    """返回高台地形 (plateau) 的布尔掩码。"""
    return np.array([t == "plateau" for t in terrain_types], dtype=bool)


def get_terrain_yaw_init_range(terrain_types: list[str]) -> np.ndarray:
    """返回初始朝向的 yaw 范围（弧度）。

    台阶地形: ±30°（面朝台阶方向，60°范围内随机）
    坡/平地:  ±180°（完全随机）
    """
    n = len(terrain_types)
    dtype = get_global_dtype()
    yaw_half_range = np.full(n, np.pi, dtype=dtype)

    for i, t in enumerate(terrain_types):
        if t in ("pyramid_stairs", "pyramid_stairs_inv"):
            yaw_half_range[i] = np.deg2rad(30.0)

    return yaw_half_range


def get_terrain_cmd_yaw_limits(terrain_types: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """返回各环境 yaw 命令的上下限。

    台阶地形: [-3.0, 3.0]（允许转向但范围适中）
    坡/平地: 使用全局 curriculum 的 yaw 范围（完全随机）
    """
    n = len(terrain_types)
    dtype = get_global_dtype()
    yaw_low = np.full(n, -5.0, dtype=dtype)
    yaw_high = np.full(n, 5.0, dtype=dtype)

    for i, t in enumerate(terrain_types):
        if t in ("pyramid_stairs", "pyramid_stairs_inv", "plateau"):
            yaw_low[i] = -2.5
            yaw_high[i] = 2.5
        # 坡和平地用全局范围，不做额外限制

    return yaw_low, yaw_high


def get_terrain_vx_forward_bias(terrain_types: list[str]) -> np.ndarray:
    """返回各环境前进命令的概率偏向。

    台阶地形: 80% 前进（vx>0），20% 后退（vx<0）
    坡/平地: 50% 前进 / 50% 后退（均匀随机）

    Returns:
        forward_prob: shape (n,) 的前进概率
    """
    n = len(terrain_types)
    dtype = get_global_dtype()
    forward_prob = np.full(n, 0.5, dtype=dtype)

    for i, t in enumerate(terrain_types):
        if t in ("pyramid_stairs", "pyramid_stairs_inv", "plateau"):
            forward_prob[i] = 0.5

    return forward_prob


class BalanceInfantryJoystickDomainRandomizationProvider(LocomotionDRProvider):

    def _get_curriculum_scale(self, env: Any) -> float:
        curriculum = getattr(env, "_curriculum", None)
        return curriculum.scale if curriculum is not None else 1.0

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        payload = build_balance_backend_reset_randomization(env, num_reset=1)
        if payload is not None:
            unsupported = capabilities.get_unsupported_reset_terms(payload.requested_terms())
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support "
                    f"Balance reset randomization terms: {names}"
                )
        validate_interval_push_support(env, capabilities)

    # =================================================================
    # 区间随机化：定时推搡机器人
    # =================================================================

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        domain_rand = getattr(env.cfg, "domain_rand", None)
        if domain_rand is None or not getattr(domain_rand, "push_robots", False):
            return None

        # 初始化推搡状态缓存
        if not hasattr(self, "_push_remaining_steps"):
            self._push_remaining_steps = 0
            self._push_base_force = None
            self._push_leg_body_ids = None
            self._push_leg_force = None

        # 推搡持续中：继续施加已有力
        if self._push_remaining_steps > 0:
            self._push_remaining_steps -= 1
            return IntervalRandomizationPlan(
                push_perturbation_limit=self._push_base_force,
                body_ids=self._push_leg_body_ids,
                body_force=self._push_leg_force,
            )

        # 根据 curriculum 进度计算推搡间隔
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
        # base 推搡力
        base_force_base = domain_rand.max_force
        base_force = [float(f) * scale for f in base_force_base]

        # 腿推搡力（作用在 rear_child1_link 上）
        LEG_BODY_NAMES = ["right_rear_child1_link", "left_rear_child1_link"]
        leg_force_cfg = getattr(domain_rand, "leg_max_force", None)
        if leg_force_cfg is not None:
            LEG_MAX_FORCE_BASE = [float(x) for x in leg_force_cfg]
        else:
            LEG_MAX_FORCE_BASE = [20.0, 20.0, 3.0]
        LEG_MAX_FORCE = [f * scale for f in LEG_MAX_FORCE_BASE]

        leg_body_ids = backend.get_body_ids(LEG_BODY_NAMES).astype(np.int32)

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

        # 推搡持续时长 3~8 步
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

    # =================================================================
    # Reset 随机化：构建 reset 时的状态分布
    # =================================================================

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        scale = self._get_curriculum_scale(env)

        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))

        # -----------------------------------------------------------
        # 1. 初始位置随机化
        # -----------------------------------------------------------

        _xy_j = float(env.cfg.spawn_xy_jitter)
        if _xy_j > 0.0:
            qpos[:, 0:2] += np.random.uniform(-_xy_j, _xy_j, (num_reset, 2))  # xy 抖动

        # 高台地形：固定在平台正中心，清零 xy 偏移
        _get_terrain = getattr(env._spawn, "get_terrain_type_for", None)
        plateau_mask = np.zeros(num_reset, dtype=bool)
        if _get_terrain is not None:
            plateau_mask = np.array(
                [t == "plateau" for t in _get_terrain(env_ids)],
                dtype=bool,
            )
        qpos[plateau_mask, 0:2] = 0.0
        extra_z = np.zeros(num_reset, dtype=get_global_dtype())
        air_spawn_prob = env.cfg.reward_config.air_spawn_probability
        air_mask = np.random.uniform(size=(num_reset,)) < air_spawn_prob
        n_air = int(np.sum(air_mask))
        if n_air > 0:
            extra_z[air_mask] = np.random.uniform(
                0.1, env.cfg.reward_config.air_spawn_max_height, size=(n_air,)
            )
        ground_mask = ~air_mask
        n_ground = int(np.sum(ground_mask))
        if n_ground > 0:
            ground_max = max(
                0.0, float(getattr(env.cfg.reward_config, "ground_spawn_max_height", 0.1))
            )
            extra_z[ground_mask] = np.random.uniform(0.0, ground_max, size=n_ground)
        qpos[:, 2] += extra_z
        qpos[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos[:, 0:3])  # 地形边界裁剪

        # -----------------------------------------------------------
        # 2. 初始朝向随机化（地形相关）
        #    4 垂直方向 (0°, 90°, 180°, 270°) 随机选一个 + 高斯噪声
        # -----------------------------------------------------------
        dirs = np.array([0.0, np.pi], dtype=get_global_dtype())
        chosen = dirs[np.random.randint(0, 2, size=(num_reset,))]
        yaw = chosen + np.random.normal(0.0, np.pi / 9, size=(num_reset,)).astype(get_global_dtype())
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))

        # -----------------------------------------------------------
        # 3. 初始姿态随机化（pitch / roll）
        # -----------------------------------------------------------
        # 权重归一化：4 维随机 → 迭代 3 次归一化，使总和为 2
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

        pitch_base = 0.0
        roll_base = 0.0
        pitch = np.random.uniform(-pitch_base * w_pitch * scale, +pitch_base * w_pitch * scale, (num_reset,))
        roll = np.random.uniform(-roll_base * w_roll * scale, +roll_base * w_roll * scale, (num_reset,))
        cpt, spt = np.cos(pitch / 2), np.sin(pitch / 2)
        q_pitch = np.stack([cpt, np.zeros(num_reset), spt, np.zeros(num_reset)], axis=1)
        cpr, spr = np.cos(roll / 2), np.sin(roll / 2)
        q_roll = np.stack([cpr, spr, np.zeros(num_reset), np.zeros(num_reset)], axis=1)
        qpos[:, 3:7] = np_quat_mul(np_quat_mul(q_roll, q_pitch), qpos[:, 3:7])

        # -----------------------------------------------------------
        # 4. 摆腿关节速度随机化
        # -----------------------------------------------------------

        swing_vel_base = 2.0
        leg_qvel_offsets = np.array([10, 17], dtype=np.int32)  # 右腿 / 左腿 swing 关节 qvel 索引
        swing_vel = np.random.uniform(-4.0, 4.0, (num_reset, len(leg_qvel_offsets)))
        qvel[:, 10] = qvel[:, 10] + swing_vel[:, 0]
        qvel[:, 17] = qvel[:, 17] + swing_vel[:, 1]

        # -----------------------------------------------------------
        # 5. 机体质心线速度 / 角速度随机化
        # -----------------------------------------------------------

        linvel_base_x = 4.2  # 前进方向最大线速度
        linvel_base_y = 1.0  # 侧向最大线速度
        angvel_base = 1.0    # 偏航最大角速度
        body_vx = np.asarray(
            np.random.uniform(0.0, linvel_base_x * w_lin * scale, size=(num_reset,)),
            dtype=get_global_dtype(),
        )
        body_vy = np.asarray(
            np.random.uniform(-linvel_base_y * w_lin * scale, +linvel_base_y * w_lin * scale, size=(num_reset,)),
            dtype=get_global_dtype(),
        )
        # 高台地形：无水平速度，从静止开始
        body_vx[plateau_mask] = 0.0
        body_vy[plateau_mask] = 0.0
        # 地面出生：xy 线速度减小到 1/4
        body_vx[ground_mask] *= 0.25
        body_vy[ground_mask] *= 0.25
        yaw = np_yaw_from_quat(qpos[:, 3:7])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        qvel[:, 0] = body_vx * cos_yaw - body_vy * sin_yaw  # 世界系 vx
        qvel[:, 1] = body_vx * sin_yaw + body_vy * cos_yaw  # 世界系 vy
        qvel[:, 2] = 0.0
        qvel[:, 3:5] = np.asarray(
            np.random.uniform(-angvel_base * 0.1 * w_ang[:, np.newaxis] * scale, +angvel_base * 0.1 * w_ang[:, np.newaxis] * scale, size=(num_reset, 2)),
            dtype=get_global_dtype(),
        )
        qvel[:, 5] = np.asarray(
            np.random.uniform(-angvel_base * w_ang * scale, +angvel_base * w_ang * scale, size=(num_reset,)),
            dtype=get_global_dtype(),
        )

        # -----------------------------------------------------------
        # 6. 轮子速度随机化（根据 body 线速度/角速度反算 + 噪声）
        # -----------------------------------------------------------

        wheel_radius = 0.06
        wheelbase = 0.380
        wyaw = qvel[:, 5].copy()
        target_rw = (body_vx + 0.5 * wyaw * wheelbase) / wheel_radius  # 右轮目标角速度
        target_lw = (body_vx - 0.5 * wyaw * wheelbase) / wheel_radius  # 左轮目标角速度
        noise_ratio = 0.1
        noise_floor = 5.0
        noise_std = noise_ratio * np.maximum(np.maximum(np.abs(target_rw), np.abs(target_lw)), noise_floor) * scale
        rw_vel = target_rw + np.asarray(
            np.random.uniform(-noise_std, noise_std, size=(num_reset,)),
            dtype=get_global_dtype(),
        )
        lw_vel = target_lw + np.asarray(
            np.random.uniform(-noise_std, noise_std, size=(num_reset,)),
            dtype=get_global_dtype(),
        )
        qvel[:, 12] = rw_vel
        qvel[:, 19] = lw_vel

        # -----------------------------------------------------------
        # 7. 指令采样
        # -----------------------------------------------------------

        commands = self._sample_commands(env, num_reset, env_ids)

        # -----------------------------------------------------------
        # 8. 域随机化参数
        # -----------------------------------------------------------

        init_actions = np.zeros((num_reset, NUM_BALANCE_ACTIONS), dtype=get_global_dtype())
        rc = env._cfg.reward_config

        dr = getattr(env.cfg, "domain_rand", None)

        # 8.1 VMC / wheel / leg PD 参数随机化（独立采样）
        vmc_ff_mult = np.ones((num_reset, 1), dtype=get_global_dtype())
        wheel_kd_mult = np.ones((num_reset, NUM_WHEEL_ACTIONS), dtype=get_global_dtype())
        leg_kp_mult = np.ones((num_reset, NUM_LEG_ACTIONS), dtype=get_global_dtype())
        leg_kd_mult = np.ones((num_reset, NUM_LEG_ACTIONS), dtype=get_global_dtype())
        if dr is not None:
            if getattr(dr, "randomize_vmc_ff", False):
                low, high = dr.vmc_ff_multiplier_range
                vmc_ff_mult = np.random.uniform(low, high, size=(num_reset, 1)).astype(get_global_dtype())
            if getattr(dr, "randomize_wheel_kd", False):
                low, high = dr.wheel_kd_multiplier_range
                wheel_kd_mult = np.random.uniform(low, high, size=(num_reset, NUM_WHEEL_ACTIONS)).astype(get_global_dtype())
            if getattr(dr, "randomize_kp", False):
                low, high = dr.kp_multiplier_range
                leg_kp_mult = np.random.uniform(low, high, size=(num_reset, NUM_LEG_ACTIONS)).astype(get_global_dtype())
            if getattr(dr, "randomize_kd", False):
                low, high = dr.kd_multiplier_range
                leg_kd_mult = np.random.uniform(low, high, size=(num_reset, NUM_LEG_ACTIONS)).astype(get_global_dtype())

        # 8.2 力矩缩放随机化（per joint）
        torque_scale = np.ones((num_reset, NUM_BALANCE_ACTIONS), dtype=get_global_dtype())
        if dr is not None and getattr(dr, "randomize_motor_torque", False):
            low, high = dr.torque_scale_range
            torque_scale = np.random.uniform(low, high, size=(num_reset, NUM_BALANCE_ACTIONS)).astype(get_global_dtype())

        # 8.3 执行器噪声随机化
        actuator_noise_std = np.zeros(num_reset, dtype=get_global_dtype())
        if dr is not None and getattr(dr, "actuator_noise", False):
            low, high = dr.actuator_noise_std_range
            actuator_noise_std = np.random.uniform(low, high, size=(num_reset,)).astype(get_global_dtype())

        # 8.4 动作平滑随机化
        cc = env._cfg.control_config
        action_smoothing = np.ones(num_reset, dtype=get_global_dtype())
        if cc.action_smoothing < 1.0:
            action_smoothing[:] = cc.action_smoothing

        # -----------------------------------------------------------
        # 9. 组装 ResetPlan
        # -----------------------------------------------------------

        info_updates: dict[str, Any] = {
            "commands": commands,
            "current_actions": init_actions,
            "last_actions": init_actions.copy(),
            "vmc_ff_multiplier": vmc_ff_mult,
            "wheel_kd_multiplier": wheel_kd_mult,
            "leg_kp_multiplier": leg_kp_mult,
            "leg_kd_multiplier": leg_kd_mult,
            "action_smoothing": action_smoothing,
            "torque_scale": torque_scale,
            "actuator_noise_std": actuator_noise_std,
        }

        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])

        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_balance_backend_reset_randomization(env, num_reset),
        )

    def _compute_reset_obs(self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel):
        env._compute_state()

        leg_len_l = env._vmc_L0[:, 0]
        leg_len_r = env._vmc_L0[:, 1]
        env._prev_leg_len[env_ids, 0] = leg_len_r[env_ids]
        env._prev_leg_len[env_ids, 1] = leg_len_l[env_ids]
        gravity_body = env._compute_gravity_body(env.get_base_quat()[env_ids])

        info_updates["swing_lw_theta"] = env._theta_vmc_l[env_ids]
        info_updates["swing_rw_theta"] = env._theta_vmc_r[env_ids]
        info_updates["_vmc_L0"] = env._vmc_L0[env_ids]
        info_updates["_vmc_theta_dot"] = env._vmc_theta_dot[env_ids]
        info_updates["_vmc_L0_dot"] = env._vmc_L0_dot[env_ids]
        info_updates["_action_t2"] = env._action_t2[env_ids]
        info_updates["_action_t3"] = env._action_t3[env_ids]
        info_updates["qacc"] = np.zeros((len(env_ids), NUM_BALANCE_ACTIONS), dtype=env._np_dtype)
        info_updates["torques"] = np.zeros((len(env_ids), NUM_BALANCE_ACTIONS), dtype=env._np_dtype)
        info_updates["_base_mass"] = env._cached_base_mass[env_ids]
        info_updates["_base_com"] = env._cached_base_com[env_ids]
        info_updates["_ground_friction"] = env._cached_ground_friction[env_ids]
        info_updates["_vmc_ff"] = env._vmc_ff[env_ids]
        dr = getattr(env._cfg, "domain_rand", None)
        if dr is not None and getattr(dr, "randomize_default_dof_pos", False):
            low, high = dr.randomize_default_dof_pos_range
            env._default_dof_pos[env_ids] = env._base_default_dof_pos[env_ids] + np.random.uniform(
                low, high, size=(len(env_ids), 4)
            ).astype(env._np_dtype)
        else:
            env._default_dof_pos[env_ids] = env._base_default_dof_pos[env_ids]

        return env._compute_obs(info_updates, gyro, gravity_body, dof_pos, dof_vel, local_linvel=linvel, env_ids=env_ids,
                              world_linvel=np.zeros((len(env_ids), 3), dtype=env._np_dtype))

    def _sample_commands(self, env, num_reset, env_ids=None):
        dtype = get_global_dtype()
        cmd_env_ids = (
            np.arange(num_reset, dtype=np.int64)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.int64)[:num_reset]
        )
        # ── 地形相关 cmdvx 限制 ──
        terrain_types = None
        if env_ids is not None and hasattr(env, '_spawn'):
            _get_terrain = getattr(env._spawn, "get_terrain_type_for", None)
            if _get_terrain is not None:
                terrain_types = _get_terrain(env_ids)
        if terrain_types is not None:
            t_yaw_low, t_yaw_high = get_terrain_cmd_yaw_limits(terrain_types)
            forward_bias = get_terrain_vx_forward_bias(terrain_types)
        else:
            forward_bias = None

        # 课程范围 ∩ 地形限制 = 真正用于采样的边界（并记录下来供 [info] 打印）
        vx_low_arr, vx_high_arr = effective_cmd_vx_bounds(
            env._curriculum.cmd_vel_range_env[cmd_env_ids, 0],
            env._curriculum.cmd_vel_range_env[cmd_env_ids, 1],
            terrain_types,
        )
        env._eff_vx_bounds[cmd_env_ids, 0] = vx_low_arr
        env._eff_vx_bounds[cmd_env_ids, 1] = vx_high_arr

        # ── 模式化命令采样: 前进 / 转向 / 原地自转 ──
        #  前进: 高 vx + 窄 yaw → 大多数样本
        #  转向: 低 vx + 宽 yaw → 高速转弯
        #  自转: vx=0  + 宽 yaw → 原地高速自转
        if terrain_types is not None:
            stairs_mask = np.array(
                [t in ("pyramid_stairs", "pyramid_stairs_inv", "plateau") for t in terrain_types],
                dtype=bool,
            )
        else:
            stairs_mask = np.zeros(num_reset, dtype=bool)
        fwd_prob = np.where(stairs_mask, CMD_MODE_PROBS_STAIRS[0], CMD_MODE_PROBS_DEFAULT[0]).astype(dtype)
        turn_prob = np.where(stairs_mask, CMD_MODE_PROBS_STAIRS[1], CMD_MODE_PROBS_DEFAULT[1]).astype(dtype)
        mode_r = np.random.uniform(size=num_reset).astype(dtype)
        is_forward = mode_r < fwd_prob
        is_turn = (mode_r >= fwd_prob) & (mode_r < fwd_prob + turn_prob)
        is_rotate = mode_r >= fwd_prob + turn_prob

        vx_cmd = np.random.uniform(vx_low_arr, vx_high_arr, size=(num_reset,)).astype(dtype)
        if np.any(is_rotate):
            vx_cmd[is_rotate] = 0.0

        yaw_cmd = np.random.normal(0.0, 2.4, size=(num_reset,)).astype(dtype)
        if np.any(is_forward):
            yaw_cmd[is_forward] = np.random.normal(
                0.0, 0.8, size=(int(is_forward.sum()),),
            ).astype(dtype)
        if np.any(is_rotate):
            yaw_cmd[is_rotate] = np.random.choice(
                [-1.0, 1.0], size=(int(is_rotate.sum()),),
            ).astype(dtype) * np.abs(np.random.normal(
                4.0, 1.0, size=(int(is_rotate.sum()),),
            )).astype(dtype)

        commands = np.asarray(
            np.stack([vx_cmd, np.zeros(num_reset, dtype=dtype), yaw_cmd], axis=1),
            dtype=dtype,
        )
        if terrain_types is not None:
            commands[:, 0] = np.clip(commands[:, 0], vx_low_arr, vx_high_arr)
            # ── 台阶地形: 前进偏好 (80% forward) ──
            if np.any(stairs_mask):
                stairs_idx = np.where(stairs_mask)[0]
                fwd_bias = float(forward_bias[stairs_mask][0])
                go_forward = np.random.uniform(size=(int(np.count_nonzero(stairs_mask)),)) < fwd_bias
                for k, idx in enumerate(stairs_idx):
                    if go_forward[k]:
                        commands[idx, 0] = np.random.uniform(0.0, vx_high_arr[idx])
                    elif commands[idx, 0] > 0.0:
                        commands[idx, 0] = -commands[idx, 0]

        zero_small_xy_commands(commands, threshold=0.05)

        high_yaw = np.abs(commands[:, 2]) > 3.0
        if np.any(high_yaw):
            commands[high_yaw, 0] = 0.0
            if env_ids is not None:
                env._spin_mask[env_ids[high_yaw]] = True

        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing, :3] = 0.0
        if getattr(env.cfg.commands, "heading_command", False):
            commands[:, 2] = 0.0

        if hasattr(env.cfg.commands, "height_limit"):
            hlim = np.asarray(env.cfg.commands.height_limit, dtype=dtype)
            hlow, hhigh = float(hlim[0]), float(hlim[1])
            if terrain_types is not None:
                terrain_difficulty = None
                if env_ids is not None and hasattr(env, '_spawn'):
                    _get_diff = getattr(env._spawn, "get_terrain_difficulty_for", None)
                    if _get_diff is not None:
                        terrain_difficulty = _get_diff(env_ids)
                use_uni, h_uni_low, h_uni_high, h_beta = get_terrain_height_params(terrain_types, terrain_difficulty)
                uni_mask = use_uni.astype(bool)
                beta_mask = ~uni_mask
                height = np.empty((num_reset, 1), dtype=dtype)
                if uni_mask.any():
                    height[uni_mask] = np.random.uniform(
                        h_uni_low[uni_mask, None], h_uni_high[uni_mask, None],
                        size=(int(uni_mask.sum()), 1),
                    ).astype(dtype)
                if beta_mask.any():
                    a = h_beta[beta_mask, 0:1]
                    b = h_beta[beta_mask, 1:2]
                    raw = np.random.beta(a, b, size=(int(beta_mask.sum()), 1)).astype(dtype)
                    height[beta_mask] = hlow + raw * (hhigh - hlow)
            else:
                height = np.asarray(
                    np.random.uniform(hlow, hhigh, size=(num_reset, 1)),
                    dtype=dtype,
                )
        else:
            default_height = float(env.cfg.reward_config.base_height_target)
            height = np.full((num_reset, 1), default_height, dtype=dtype)
        commands = np.concatenate([commands, height], axis=1)

        # ── 第 5 维：预留的跳跃命令槽位（跳跃功能已移除，恒为 0，保留观测维度） ──
        jump = np.zeros((num_reset, 1), dtype=dtype)
        commands = np.concatenate([commands, jump], axis=1)

        return commands
# =====================================================================
# 主环境类
# =====================================================================


@registry.env("BalanceInfantryJoystickFlat", sim_backend="mujoco")
@registry.env("BalanceInfantryJoystickFlat", sim_backend="motrix")
class BalanceInfantryJoystickEnv(BalanceInfantryBaseEnv):
    _cfg: BalanceInfantryJoystickCfg

    def __init__(self, cfg: BalanceInfantryJoystickCfg | None = None,
                 num_envs: int = 1, backend_type: str = "mujoco"):
        cfg = cfg or BalanceInfantryJoystickCfg()
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            motrix_max_iterations=cfg.motrix_max_iterations,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()

        self._height_scan_dim = len(cfg.terrain_scan.measured_points_x) * len(
            cfg.terrain_scan.measured_points_y
        )
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

        self._terrain_surface_sampler = getattr(backend, "terrain_surface_sampler", None)
        terrain_origins = getattr(backend, "terrain_origins", None)
        self._scene_terrain_origins = terrain_origins if terrain_origins is not None else None
        terrain_generator = cfg.scene.terrain.generator if cfg.scene.terrain is not None else None
        if self._scene_terrain_origins is not None and terrain_generator is not None:
            self._spawn = TerrainSpawnManager(
                num_envs,
                self._scene_terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=cfg.terrain_curriculum,
                terrain_surface_sampler=self._terrain_surface_sampler,
                terrain_type_names=list(terrain_generator.sub_terrains.keys()),
            )

        try:
            self._right_wheel_body_id = self._backend.get_body_ids(["right_wheel_link"])[0]
            self._left_wheel_body_id = self._backend.get_body_ids(["left_wheel_link"])[0]
            self._base_body_id = self._backend.get_body_ids([cfg.asset.base_name])[0]
        except Exception:
            self._right_wheel_body_id = None
            self._left_wheel_body_id = None
            self._base_body_id = 0

        if cfg.scene.terrain is not None:
            self._ground_geom_id = self._backend.get_geom_id(cfg.scene.terrain.geom_name)
        else:
            self._ground_geom_id = self._backend.get_geom_id("floor")
        self._cached_base_mass = np.zeros(num_envs, dtype=self._np_dtype)
        self._cached_base_com = np.zeros((num_envs, 3), dtype=self._np_dtype)
        self._cached_ground_friction = np.zeros((num_envs, 2), dtype=self._np_dtype)

        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)

        wheel_cfg = np.asarray(self._cfg.control_config.wheel_kd, dtype=self._np_dtype)
        self._wheel_kd = np.broadcast_to(wheel_cfg, (num_envs, NUM_WHEEL_ACTIONS)).copy()
        self._base_wheel_kd = self._wheel_kd.copy()
        self._last_motor_ctrl = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)

        ctrl_cfg = self._cfg.control_config
        self._leg_kp = np.tile(np.array(ctrl_cfg.leg_kp, dtype=self._np_dtype), (num_envs, 1))
        self._leg_kd = np.tile(np.array(ctrl_cfg.leg_kd, dtype=self._np_dtype), (num_envs, 1))
        self._base_leg_kp = self._leg_kp.copy()
        self._base_leg_kd = self._leg_kd.copy()
        # 轮子扭矩-转速（功率）包络；默认 disabled，行为与旧版一致
        self._wheel_envelope = WheelTorqueEnvelope(ctrl_cfg.wheel_torque_speed)
        self._wheel_idx = np.asarray(WHEEL_JOINT_INDICES, dtype=np.intp)

        self._vmc_ff = np.full(num_envs, ctrl_cfg.vmc.feedforward_force, dtype=self._np_dtype)
        self._base_vmc_ff = self._vmc_ff.copy()

        self._base_default_dof_pos = np.broadcast_to(
            self.get_dof_pos()[0, :4].copy(), (num_envs, 4)
        )
        self._default_dof_pos = self._base_default_dof_pos.copy()

        self._backend.set_pre_step_control(self._pre_step_motor_control)

        self._curriculum = CurriculumManager(cfg, num_envs)
        # 实际生效的 per-env cmdvx 边界（课程范围 ∩ 地形限制），供 [info] 打印。
        # 初始化为课程名义范围；首次 reset 采样后即被真实边界覆盖。
        self._eff_vx_bounds = np.array(
            self._curriculum.cmd_vel_range_env, dtype=self._np_dtype, copy=True
        )
        self._init_domain_randomization(BalanceInfantryJoystickDomainRandomizationProvider())
        self._init_reward_functions()

        # 延迟终止 / 差分加速度所需缓冲
        self._vertex_touch_count = np.zeros(num_envs, dtype=np.int32)
        self._tilt_count = np.zeros(num_envs, dtype=np.int32)
        self._height_low_count = np.zeros(num_envs, dtype=np.int32)
        self._swing_count = np.zeros(num_envs, dtype=np.int32)
        
        self._last_dof_vel_for_acc = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._last_rw_body = None   
        self._last_lw_body = None
        self._episode_steps = np.zeros(num_envs, dtype=np.int32)
        self._powerless_active = np.zeros(num_envs, dtype=bool)
        self._term_vertex = 0
        self._term_tilt = 0
        self._term_height_low = 0
        self._term_swing = 0
        self._term_loosened = False
        self._total_steps = 0
        self._term_window_steps = 0
        self._hit_vertex = 0
        self._hit_tilt = 0
        self._hit_height_low = 0
        self._hit_swing = 0

        self._episode_vel_sum = np.zeros(num_envs, dtype=self._np_dtype)
        self._episode_cmd_vel_sum = np.zeros(num_envs, dtype=self._np_dtype)
        self._episode_vel_sum_fwd = np.zeros(num_envs, dtype=self._np_dtype)
        self._episode_cmd_vel_sum_fwd = np.zeros(num_envs, dtype=self._np_dtype)
        self._episode_vel_sum_bwd = np.zeros(num_envs, dtype=self._np_dtype)
        self._episode_cmd_vel_sum_bwd = np.zeros(num_envs, dtype=self._np_dtype)
        self._episode_tilt_sum = np.zeros(num_envs, dtype=self._np_dtype)
        self._episode_swing_err_sum = np.zeros(num_envs, dtype=self._np_dtype)
        self._episode_height_err_sum = np.zeros(num_envs, dtype=self._np_dtype)
        self._episode_tracking_reward_sum = np.zeros(num_envs, dtype=self._np_dtype)
        self._spin_mask = np.zeros(num_envs, dtype=bool)
        self._leg_action_clip = 100.0  # 动作不限制（等价于不夹），只由执行器力矩 forcerange 限制
        self._prev_body_vx = np.zeros(num_envs, dtype=self._np_dtype)
        self._target_yaw = np.zeros(num_envs, dtype=self._np_dtype)
        # ── 手动观测归一化（29/32 维） ──
        # 组成: gyro(3) pitch(1) roll(1) cmd(4) last_act(6)
        #       dof_pos_err(4) dof_vel(6) vmc_θ(2) vmc_L0(2)
        #       [+ local_linvel(3) if obs_vel]
        self._obs_mean = np.array([
            0.0, 0.0, 0.0,     # 0-2:  gyro
            0.0,                # 3:    pitch
            0.0,                # 4:    roll
            0.0, 0.0, 0.29, 0.05,  # 5-8:  cmd(vx,vyaw,height,jump)  jump 槽位恒 0
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 9-14: last_actions
            0.0, 0.0, 0.0, 0.0, # 15-18: dof_pos_err
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 19-24: dof_vel
            0.0, 0.0,           # 25-26: vmc_theta
            0.28, 0.28,         # 27-28: vmc_L0
        ], dtype=self._np_dtype)
        self._obs_std = np.array([
            0.34, 0.62, 2.0,      # 0-2:  gyro (z 2.0)
            0.15,               # 3:    pitch
            0.10,               # 4:    roll
            1.5, 2.0, 0.08, 0.22,  # 5-8:  cmd
            0.6, 0.6, 0.6, 0.6, 0.6, 0.6,  # 9-14: last_actions
            0.3, 0.3, 0.3, 0.3, # 15-18: dof_pos_err
            1.5, 1.5, 1.5, 1.5, 22.0, 22.0,  # 19-24: dof_vel (leg 1.5, wheel 22)
            0.4, 0.4,           # 25-26: vmc_theta
            0.05, 0.05,         # 27-28: vmc_L0
        ], dtype=self._np_dtype)
        if self._cfg.obs_vel:
            self._obs_mean = np.append(self._obs_mean, np.array([0.0, 0.0, 0.0], dtype=self._np_dtype))
            self._obs_std = np.append(self._obs_std, np.array([1.5, 0.5, 0.5], dtype=self._np_dtype))
        self._prev_leg_len = np.zeros((num_envs, 2), dtype=self._np_dtype)  # 上一步腿长 [右, 左]，用于计算腿长速度

        # ── 接触滤波 ──
        self._contact_filt = np.zeros((num_envs, 2), dtype=bool)
        self._last_contacts = np.zeros((num_envs, 2), dtype=bool)
        self._base_height_phi_hold = np.zeros(num_envs, dtype=self._np_dtype)

        # ── VMC 状态缓冲 (五连杆) ──
        self._vmc_phi1 = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._vmc_phi4 = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._vmc_phi2 = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._vmc_phi3 = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._vmc_sigma1 = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._vmc_L0 = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._vmc_phi0 = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._vmc_L0_dot = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._vmc_phi0_dot = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._vmc_last_L0 = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._vmc_last_phi0 = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._vmc_last_gravity = np.zeros((num_envs, 3), dtype=self._np_dtype)
        self._vmc_theta_dot = np.zeros((num_envs, 2), dtype=self._np_dtype)
        max_delay = max(self._cfg.control_config.action_delay_range)
        obs_delay_min, obs_delay_max = self._cfg.control_config.obs_delay_range
        if obs_delay_min == 0 and obs_delay_max == 0:
            obs_delay_min, obs_delay_max = self._cfg.control_config.action_delay_range
        self._max_delay = max(max_delay, obs_delay_max)
        self._vmc_targets_fifo = np.zeros((num_envs, max_delay + 1, 2, 3), dtype=self._np_dtype)
        self._action_delay_steps = np.zeros(num_envs, dtype=np.int32)
        self._obs_fifo = None
        self._obs_delay_steps = np.zeros(num_envs, dtype=np.int32)
        self._action_delay_range = (self._cfg.control_config.action_delay_range[0],
                                     self._cfg.control_config.action_delay_range[1])
        self._obs_delay_range = (obs_delay_min, obs_delay_max)
        self._vmc_targets = np.zeros((num_envs, 2, 3), dtype=self._np_dtype)  # 当前控制步目标
        self._vmc_F = np.zeros((num_envs, 2), dtype=self._np_dtype)        # VMC 虚拟推力
        self._vmc_T = np.zeros((num_envs, 2), dtype=self._np_dtype)        # VMC 虚拟力矩
        self._vmc_just_reset = np.ones(num_envs, dtype=bool)

        self._prev_filtered_actions = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._action_smoothing = np.full(num_envs, float(self._cfg.control_config.action_smoothing), dtype=self._np_dtype)
        self._torque_scale = np.ones((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._actuator_noise_std = np.zeros(num_envs, dtype=self._np_dtype)
        self._action_delay_steps = np.zeros(num_envs, dtype=np.int32)
        self._prev_targets_buffer = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._action_t2 = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._action_t3 = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        # 真实动作历史（供 action_smooth 二阶差分；与 critic 的 target 历史 _action_t2/t3 分开）
        self._prev_actions = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._prev2_actions = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._current_targets = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._cmd_vel_ref = float(self._cfg.commands.vel_limit[1][0])
        self._cached_kin_torques = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._cached_kin_qacc = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._latest_obs_scaled = None

    def set_terrain_iteration(self, iteration: int) -> None:
        self._spawn.set_iteration(iteration)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        obs_dim = 29 + (3 if self._cfg.obs_vel else 0)
        return {"obs": obs_dim, "critic": 68 + self._height_scan_dim}

    # -----------------------------------------------------------------
    # reset：清零缓存
    # -----------------------------------------------------------------

    def reset(self, env_indices: np.ndarray):
        env_ids = np.asarray(env_indices, dtype=np.int32)

        self._curriculum.on_reset(
            env_ids,
            self._episode_vel_sum_fwd,
            self._episode_cmd_vel_sum_fwd,
            self._episode_vel_sum_bwd,
            self._episode_cmd_vel_sum_bwd,
            self._episode_steps,
            self._episode_tilt_sum,
            self._episode_swing_err_sum,
            self._episode_height_err_sum,
            self._episode_tracking_reward_sum,
        )

        # 先清零 spin_mask，_sample_commands 里会重新设置
        self._spin_mask[env_ids] = False
        self._prev_leg_len[env_ids] = 0.0
        self._contact_filt[env_ids] = False
        self._base_height_phi_hold[env_ids] = 0.0
        self._prev_filtered_actions[env_ids] = 0.0

        obs, info = super().reset(env_ids)

        self._cached_base_mass[env_ids] = self._backend.get_body_mass()[self._base_body_id]
        self._cached_base_com[env_ids] = self._backend.get_body_ipos()[self._base_body_id]
        if self._ground_geom_id is not None:
            geom_friction = self._backend.get_geom_friction()
            self._cached_ground_friction[env_ids] = geom_friction[self._ground_geom_id, :2]

        dr = getattr(self._cfg, "domain_rand", None)
        if dr is not None:
            vmc_ff = np.asarray(info.get("vmc_ff_multiplier", np.ones((len(env_ids), 1))), dtype=self._np_dtype).flatten()
            wheel_kd = np.asarray(info.get("wheel_kd_multiplier", np.ones((len(env_ids), NUM_WHEEL_ACTIONS))), dtype=self._np_dtype)
            leg_kp = np.asarray(info.get("leg_kp_multiplier", np.ones((len(env_ids), NUM_LEG_ACTIONS))), dtype=self._np_dtype)
            leg_kd = np.asarray(info.get("leg_kd_multiplier", np.ones((len(env_ids), NUM_LEG_ACTIONS))), dtype=self._np_dtype)
            self._vmc_ff[env_ids] = self._base_vmc_ff[env_ids] * vmc_ff
            self._wheel_kd[env_ids] = self._base_wheel_kd[env_ids] * wheel_kd
            self._leg_kp[env_ids] = self._base_leg_kp[env_ids] * leg_kp
            self._leg_kd[env_ids] = self._base_leg_kd[env_ids] * leg_kd

        self._action_smoothing[env_ids] = np.asarray(
            info.get("action_smoothing", np.full(len(env_ids), self._cfg.control_config.action_smoothing)),
            dtype=self._np_dtype,
        )
        self._torque_scale[env_ids] = np.asarray(
            info.get("torque_scale", np.ones((len(env_ids), NUM_BALANCE_ACTIONS))),
            dtype=self._np_dtype,
        )
        self._actuator_noise_std[env_ids] = np.asarray(
            info.get("actuator_noise_std", np.zeros(len(env_ids))),
            dtype=self._np_dtype,
        ).flatten()

        self._vertex_touch_count[env_ids] = 0
        self._tilt_count[env_ids] = 0
        self._height_low_count[env_ids] = 0
        self._swing_count[env_ids] = 0
        self._last_dof_vel_for_acc[env_ids] = 0.0
        self._last_motor_ctrl[env_ids] = 0.0
        cur_n = getattr(self, "_last_rw_body", None)
        total = cur_n.shape[0] if isinstance(cur_n, np.ndarray) else 0
        if total == 0 or total < int(env_ids.max()) + 1:
            n = len(self._vertex_touch_count)
            self._last_rw_body = np.zeros(n, dtype=get_global_dtype())
            self._last_lw_body = np.zeros(n, dtype=get_global_dtype())
            self._swing_first_step = np.ones(n, dtype=bool)
        self._swing_first_step[env_ids] = True
        self._episode_steps[env_ids] = 0
        self._powerless_active[env_ids] = np.random.random(len(env_ids)) < self._cfg.reward_config.powerless_probability
        self._vmc_targets_fifo[env_ids] = 0.0
        self._vmc_targets[env_ids] = 0.0
        self._current_targets[env_ids] = self.default_angles
        self._prev_targets_buffer[env_ids] = self.default_angles
        self._action_t2[env_ids] = self.default_angles
        self._action_t3[env_ids] = self.default_angles
        self._prev_actions[env_ids] = 0.0
        self._prev2_actions[env_ids] = 0.0

        self._vmc_last_L0[env_ids] = 0.0
        self._vmc_last_phi0[env_ids] = 0.0
        self._vmc_last_gravity[env_ids] = 0.0
        self._vmc_just_reset[env_ids] = True
        self._vmc_L0[env_ids] = 0.0
        self._vmc_phi0[env_ids] = 0.0
        self._vmc_L0_dot[env_ids] = 0.0
        self._vmc_phi0_dot[env_ids] = 0.0
        self._vmc_F[env_ids] = 0.0
        self._vmc_T[env_ids] = 0.0

        self._episode_vel_sum[env_ids] = 0.0
        self._episode_cmd_vel_sum[env_ids] = 0.0
        self._episode_vel_sum_fwd[env_ids] = 0.0
        self._episode_cmd_vel_sum_fwd[env_ids] = 0.0
        self._episode_vel_sum_bwd[env_ids] = 0.0
        self._episode_cmd_vel_sum_bwd[env_ids] = 0.0
        self._episode_tilt_sum[env_ids] = 0.0
        self._episode_swing_err_sum[env_ids] = 0.0
        self._episode_height_err_sum[env_ids] = 0.0
        self._episode_tracking_reward_sum[env_ids] = 0.0
        self._target_yaw[env_ids] = 0.0

        return obs, info

    # -----------------------------------------------------------------
    # action → PD torque
    # -----------------------------------------------------------------

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        leg_clip = self._leg_action_clip
        wheel_clip = self._cfg.control_config.clip_actions
        clipped = actions.copy()
        clipped[:, [0, 1, 2, 3]] = np.clip(actions[:, [0, 1, 2, 3]], -leg_clip, leg_clip)
        clipped[:, [4, 5]] = np.clip(actions[:, [4, 5]], -wheel_clip, wheel_clip)

        alpha = self._action_smoothing  # per-env 随机化 α (训练时 DR, 部署时固定)
        mask = alpha < 1.0
        if np.any(mask):
            smoothed = alpha[mask, np.newaxis] * clipped[mask] + (1.0 - alpha[mask, np.newaxis]) * self._prev_filtered_actions[mask]
            self._prev_filtered_actions[mask] = smoothed.copy()
            clipped[mask] = smoothed

        clipped_actions = np.asarray(clipped, dtype=self._np_dtype)
        # 动作历史后移：_prev_actions ← a_{t-1}，_prev2_actions ← a_{t-2}
        # 注意：_action_t2/_action_t3 存的是关节 target，不能用于动作差分
        self._prev2_actions[:] = self._prev_actions
        self._prev_actions[:] = state.info.get(
            "current_actions", np.zeros_like(self._prev_actions)
        )
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped_actions))

        exec_actions = clipped_actions

        self._action_t3[:] = self._action_t2
        self._action_t2[:] = self._prev_targets_buffer
        self._prev_targets_buffer[:] = self._current_targets

        ctrl_cfg = self._cfg.control_config
        default_dof = self._default_dof_pos  # (num_envs, 4): LF, RF, LR, RR

        # 腿关节：参考 CoD per-joint 物理限位缩放
        # action 0 → LF (joint 0), limits[0]=[-1.57, 2.40]
        lf_lower, lf_upper = JOINT_LIMITS[0]
        lf_default = default_dof[:, 0]
        lf_scale = ctrl_cfg.action_scale_joint
        lf_target = exec_actions[:, 0] * lf_scale + lf_default
        np.clip(lf_target, lf_lower, lf_upper, out=lf_target)

        # action 1 → RF (joint 1), limits[1]=[-1.57, 2.40]
        rf_lower, rf_upper = JOINT_LIMITS[1]
        rf_default = default_dof[:, 1]
        rf_scale = ctrl_cfg.action_scale_joint
        rf_target = exec_actions[:, 1] * rf_scale + rf_default
        np.clip(rf_target, rf_lower, rf_upper, out=rf_target)

        # action 2 → LR (joint 2), limits[2]=[-1.57, 1.57]
        lr_lower, lr_upper = JOINT_LIMITS[2]
        lr_default = default_dof[:, 2]
        lr_scale = ctrl_cfg.action_scale_joint
        lr_target = exec_actions[:, 2] * lr_scale + lr_default
        np.clip(lr_target, lr_lower, lr_upper, out=lr_target)

        # action 3 → RR (joint 3), limits[3]=[-1.57, 1.57]
        rr_lower, rr_upper = JOINT_LIMITS[3]
        rr_default = default_dof[:, 3]
        rr_scale = ctrl_cfg.action_scale_joint
        rr_target = exec_actions[:, 3] * rr_scale + rr_default
        np.clip(rr_target, rr_lower, rr_upper, out=rr_target)

        # 轮子：速度控制（可选按 max_wheel_speed 夹紧目标速度）
        lw_target = exec_actions[:, 4] * ctrl_cfg.action_scale_wheel
        rw_target = exec_actions[:, 5] * ctrl_cfg.action_scale_wheel
        wheel_targets = self._wheel_envelope.clamp_target(
            np.column_stack([lw_target, rw_target])
        )
        lw_target = wheel_targets[:, 0]
        rw_target = wheel_targets[:, 1]

        # DEBUG: 打印模型原始输出和目标
        if self._cfg.debug_enabled:
            if not hasattr(self, '_debug_apply_count'):
                object.__setattr__(self, '_debug_apply_count', 0)
            if not hasattr(self, '_debug_print_flag'):
                object.__setattr__(self, '_debug_print_flag', False)
        if self._cfg.debug_enabled and self._debug_print_flag:
            debug_obs = self._debug_last_obs
            obs_names = ["gyro_x","gyro_y","gyro_z","pitch","roll","cmd_vel","cmd_yaw","cmd_h","cmd_0",
                         "act0","act1","act2","act3","act4","act5",
                         "dof_err_LF","dof_err_RF","dof_err_LR","dof_err_RR",
                         "dvel_LF","dvel_RF","dvel_LR","dvel_RR","dvel_LW","dvel_RW",
                         "theta_l","theta_r","L0_l","L0_r"]
            pairs = " ".join(
                f"{obs_names[i] if i < len(obs_names) else f'obs{i}'}={debug_obs[i]:+.4f}"
                for i in range(len(debug_obs))
            )
            print(f"[DEBUG #{self._debug_obs_count}] obs_raw={pairs}")
            print(f"[DEBUG #{self._debug_obs_count}] model_action={exec_actions[0]} last_torque={self._last_motor_ctrl[0]}")
            object.__setattr__(self, '_debug_print_flag', False)

        self._vmc_targets[:, 0, 0] = lf_target
        self._vmc_targets[:, 0, 1] = lr_target
        self._vmc_targets[:, 0, 2] = lw_target
        self._vmc_targets[:, 1, 0] = rf_target
        self._vmc_targets[:, 1, 1] = rr_target
        self._vmc_targets[:, 1, 2] = rw_target

        targets = np.zeros((clipped_actions.shape[0], NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        targets[:, 0] = lf_target
        targets[:, 1] = rf_target
        targets[:, 2] = lr_target
        targets[:, 3] = rr_target
        targets[:, 4] = lw_target
        targets[:, 5] = rw_target

        state.info["current_actions"] = clipped_actions
        self._current_targets[:] = targets

        return targets

    def _update_vmc_state(self) -> None:
        """更新 VMC 虚拟状态: 五连杆正运动学计算 L₀, φ₀, φ₂, φ₃, σ₁ 及其速度。

        电机 → 五连杆映射:
            左腿: φ₁ = dof_pos[2] (LR, 后大腿) + 150°,  φ₄ = dof_pos[0] (LF, 前大腿) + 30°
            右腿: φ₁ = dof_pos[3] (RR, 后大腿) + 150°,  φ₄ = dof_pos[1] (RF, 前大腿) + 30°
        """
        vmc_cfg = self._cfg.control_config.vmc
        joint_pos = self.get_dof_pos()
        sim_dt = float(self._cfg.sim_dt)

        calibrated = joint_pos[:, :4] - self._default_dof_pos[:, :4]  # (N,4): LF, RF, LR, RR
        phi1 = np.stack([
            calibrated[:, 2] + 5*np.pi/6,       # 左腿: LR (后大腿) + 150°
            calibrated[:, 3] + 5*np.pi/6,       # 右腿: RR (后大腿) + 150°
        ], axis=1)
        phi4 = np.stack([
            calibrated[:, 0] + np.pi/6,          # 左腿: LF (前大腿) + 30°
            calibrated[:, 1] + np.pi/6,          # 右腿: RF (前大腿) + 30°
        ], axis=1)

        L0, phi0, phi2, phi3, sigma1 = forward_kinematics_fivebar(
            phi1, phi4,
            vmc_cfg.L1, vmc_cfg.L2, vmc_cfg.L3, vmc_cfg.L4, vmc_cfg.L5,
        )

        self._vmc_phi1 = phi1
        self._vmc_phi4 = phi4
        self._vmc_L0 = L0
        self._vmc_phi0 = phi0
        self._vmc_phi2 = phi2
        self._vmc_phi3 = phi3
        self._vmc_sigma1 = sigma1

        if np.any(self._vmc_just_reset):
            self._vmc_last_L0[self._vmc_just_reset] = L0[self._vmc_just_reset]
            self._vmc_last_phi0[self._vmc_just_reset] = phi0[self._vmc_just_reset]
            self._vmc_L0_dot[self._vmc_just_reset] = 0.0
            self._vmc_phi0_dot[self._vmc_just_reset] = 0.0
            self._vmc_just_reset[:] = False
        else:
            self._vmc_L0_dot = (L0 - self._vmc_last_L0) / sim_dt
            self._vmc_phi0_dot = (phi0 - self._vmc_last_phi0) / sim_dt

        self._vmc_last_L0 = L0.copy()
        self._vmc_last_phi0 = phi0.copy()

    def _compute_state(self) -> None:
        """每物理步：更新 VMC 原始量 → 派生状态量 (θ, θ̇, world-frame θ)。"""
        just_reset = self._vmc_just_reset.copy()
        self._update_vmc_state()
        quat = self.get_base_quat()
        gravity_body = self._compute_gravity_body(quat)

        if np.any(just_reset):
            self._vmc_last_gravity[just_reset] = gravity_body[just_reset]
            pitch_dot = np.zeros(len(gravity_body), dtype=self._np_dtype)
        else:
            sim_dt = float(self._cfg.sim_dt)
            pitch_dot = (gravity_body[:, 0] - self._vmc_last_gravity[:, 0]) / sim_dt

        self._vmc_last_gravity = gravity_body.copy()
        pitch = np.arcsin(np.clip(gravity_body[:, 0], -1.0, 1.0))

        self._vmc_theta_dot = -self._vmc_phi0_dot - pitch_dot[:, np.newaxis]
        self._theta_vmc_l = (np.pi / 2 - self._vmc_phi0[:, 0] + pitch + np.pi) % (2.0 * np.pi) - np.pi
        self._theta_vmc_r = (np.pi / 2 - self._vmc_phi0[:, 1] + pitch + np.pi) % (2.0 * np.pi) - np.pi
    def _compute_vmc_torques(self, backend: Any) -> np.ndarray:
        """纯 VMC 重力前馈: mg/cos(θ) → 雅可比映射 → 关节力矩。

        Returns:
            vmc_torques: shape (num_envs, 6), 叠加到 PD 力矩上的 VMC 前馈力矩
        """
        vmc_cfg = self._cfg.control_config.vmc

        theta_world = np.column_stack([self._theta_vmc_l, self._theta_vmc_r])
        theta_abs = np.abs(theta_world)
        ramp = np.clip((np.pi / 3 - theta_abs) / (np.pi / 6), 0.0, 1.0)
        feedforward = self._vmc_ff.reshape(-1, 1) / np.cos(np.clip(theta_world, -1.5, 1.5)) * ramp

        tau1, tau4 = compute_vmc_torques_fivebar(
            self._vmc_phi1, self._vmc_phi4,
            self._vmc_phi0, self._vmc_phi2, self._vmc_phi3, self._vmc_sigma1,
            self._vmc_L0,
            self._vmc_L0_dot,
            theta_world, self._vmc_theta_dot,
            theta_world, self._vmc_L0,
            vmc_cfg.L1, vmc_cfg.L4,
            np.zeros_like(theta_world), np.zeros_like(theta_world),
            np.zeros_like(theta_world), np.zeros_like(theta_world),
            feedforward,
        )

        # 存储 VMC 虚拟力/力矩，供调试打印
        self._vmc_F = feedforward
        self._vmc_T = np.zeros_like(feedforward)

        vmc_torques = np.zeros((self._num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        vmc_torques[:, 0] = tau4[:, 0]   # τ₄ → LF (左前)
        vmc_torques[:, 2] = tau1[:, 0]   # τ₁ → LR (左后)
        vmc_torques[:, 3] = tau1[:, 1]   # τ₁ → RR (右后)
        vmc_torques[:, 1] = tau4[:, 1]   # τ₄ → RF (右前)

        return vmc_torques

    def _compute_vmc_feedforward_torques(self) -> np.ndarray:
        """无力期专用：只有一个不淡出的重力前馈（Tp = 0，轮驱为 0）。"""
        vmc_cfg = self._cfg.control_config.vmc
        theta_world = np.column_stack([self._theta_vmc_l, self._theta_vmc_r])
        return vmc_feedforward_torques_no_ramp(
            self._vmc_ff, theta_world,
            self._vmc_phi1, self._vmc_phi4, self._vmc_phi0,
            self._vmc_phi2, self._vmc_phi3, self._vmc_sigma1, self._vmc_L0,
            vmc_cfg.L1, vmc_cfg.L4, self._num_envs, self._np_dtype,
        )

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        _ps = time.perf_counter()

        self._compute_state()

        joint_pos = self.get_dof_pos()
        joint_vel = self.get_dof_vel()

        self._vmc_targets_fifo[:, 1:, :, :] = self._vmc_targets_fifo[:, :-1, :, :]
        self._vmc_targets_fifo[:, 0, :, :] = self._vmc_targets

        delay_idx = np.random.randint(
            self._action_delay_range[0], self._action_delay_range[1] + 1, self._num_envs,
        ).astype(np.int32)
        lf_ref = self._vmc_targets_fifo[np.arange(self._num_envs), delay_idx, 0, 0]
        lr_ref = self._vmc_targets_fifo[np.arange(self._num_envs), delay_idx, 0, 1]
        lw_ref = self._vmc_targets_fifo[np.arange(self._num_envs), delay_idx, 0, 2]
        rf_ref = self._vmc_targets_fifo[np.arange(self._num_envs), delay_idx, 1, 0]
        rr_ref = self._vmc_targets_fifo[np.arange(self._num_envs), delay_idx, 1, 1]
        rw_ref = self._vmc_targets_fifo[np.arange(self._num_envs), delay_idx, 1, 2]

        joint_targets = np.column_stack([lf_ref, rf_ref, lr_ref, rr_ref, lw_ref, rw_ref])

        motor_ctrl = compute_balance_motor_ctrl(
            joint_targets, joint_pos, joint_vel,
            self._leg_kp, self._leg_kd, self._wheel_kd,
            self._ctrl_lower, self._ctrl_upper,
            np.zeros_like(joint_pos),
        )

        vmc_torques = self._compute_vmc_torques(backend)
        motor_ctrl += vmc_torques

        # 轮子执行器功率包络：τ 受 τ_max(ω) 限制（legs/VMC 不受影响）
        if self._wheel_envelope.enabled:
            motor_ctrl[:, self._wheel_idx] = self._wheel_envelope.limit(
                motor_ctrl[:, self._wheel_idx], joint_vel[:, self._wheel_idx]
            )

        powerless_steps = int(self._curriculum.effective_powerless_duration / self._cfg.ctrl_dt)
        mask = (self._episode_steps < powerless_steps) & self._powerless_active
        if np.any(mask):
            # 无力期：关掉 PD 与轮驱，只保留**不淡出**的重力前馈，模拟机械力把腿蹬出
            motor_ctrl[mask] = self._compute_vmc_feedforward_torques()[mask]

        motor_ctrl *= self._torque_scale

        noise_add = np.random.randn(*motor_ctrl.shape).astype(self._np_dtype) * self._actuator_noise_std[:, np.newaxis]
        motor_ctrl += noise_add
        np.clip(motor_ctrl, self._ctrl_lower, self._ctrl_upper, out=motor_ctrl)

        self._last_motor_ctrl = motor_ctrl.copy()

        if self._cfg.debug_enabled:
            self._debug_joint_pos = joint_pos
            self._debug_joint_vel = joint_vel
            self._debug_motor_ctrl = motor_ctrl
        if self._cfg.debug_enabled:
            if not hasattr(self, '_pre_step_time_ms'):
                object.__setattr__(self, '_pre_step_time_ms', 0.0)
            object.__setattr__(self, '_pre_step_time_ms',
                               self._pre_step_time_ms + (time.perf_counter() - _ps) * 1000)
        return motor_ctrl

    # -----------------------------------------------------------------
    # 传感器辅助
    # -----------------------------------------------------------------

    def get_accel(self) -> np.ndarray:
        return np.asarray(self._backend.get_sensor_data(self._cfg.sensor.accel), dtype=get_global_dtype())

    def get_upvector(self) -> np.ndarray:
        return np.asarray(self._backend.get_sensor_data(self._cfg.sensor.upvector), dtype=get_global_dtype())

    @staticmethod
    def _compute_gravity_body(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        gx = 2 * (x * z - w * y)
        gy = 2 * (y * z + w * x)
        gz = 1 - 2 * (x * x + y * y)
        return np.stack([gx, gy, gz], axis=1)

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

    # -----------------------------------------------------------------
    # observation
    # -----------------------------------------------------------------

    def _compute_obs(self, info, gyro, gravity_body, dof_pos, dof_vel, local_linvel=None, env_ids=None, world_linvel=None):
        _tco0 = time.perf_counter()
        n = gyro.shape[0]
        dtype = get_global_dtype()
        noise_cfg = self._cfg.noise_config
        last_actions = np.asarray(info.get("last_actions", np.zeros((n, NUM_BALANCE_ACTIONS), dtype=dtype)), dtype=dtype)
        commands = np.asarray(info.get("commands", np.zeros((n, 5), dtype=dtype)), dtype=dtype)

        pitch = np.arcsin(np.clip(gravity_body[:, 0], -1.0, 1.0)).reshape(-1, 1)
        roll = np.arcsin(np.clip(-gravity_body[:, 1], -1.0, 1.0)).reshape(-1, 1)

        if local_linvel is None:
            local_linvel = np.zeros((n, 3), dtype=dtype)

        if world_linvel is None:
            world_linvel = np.zeros((n, 3), dtype=dtype)

        yaw_error = np.asarray(info.get("yaw_error", np.zeros(n, dtype=dtype)), dtype=dtype)

        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        noisy_pitch, noisy_roll = self._noisy_attitude(gravity_body)

        default_dof_pos = self._default_dof_pos[env_ids] if env_ids is not None else self._default_dof_pos

        vmc_cols = self._vmc_obs_columns(info, noisy=True)
        vmc_theta_l, vmc_theta_r, vmc_L0_l, vmc_L0_r = (
            vmc_cols[0], vmc_cols[1], vmc_cols[4], vmc_cols[5],
        )

        obs_cols = [
            noisy_gyro, noisy_pitch, noisy_roll, commands[:, [0, 2, 3, 4]], last_actions,
            self._obs_noise(
                dof_pos[:, :4] - default_dof_pos[:, :4],
                noise_cfg.scale_joint_angle,
            ),
            noisy_dof_vel,
            vmc_theta_l, vmc_theta_r, vmc_L0_l, vmc_L0_r,
        ]
        if self._cfg.obs_vel:
            obs_cols.append(self._obs_noise(local_linvel, noise_cfg.scale_linvel))
        obs_flat = np.asarray(np.concatenate(obs_cols, axis=1), dtype=dtype)

        if self._cfg.debug_enabled and env_ids is None:
            if not hasattr(self, '_debug_obs_count'):
                object.__setattr__(self, '_debug_obs_count', 0)
            self._debug_obs_count += 1
            if self._debug_obs_count % 50 == 1:  # 每50个控制步=1s打印一次 obs
                object.__setattr__(self, '_debug_last_obs', obs_flat[0].copy())
                object.__setattr__(self, '_debug_print_flag', True)

        if self._cfg.obs_manual_norm:
            obs_scaled = np.clip(obs_flat, -100.0, 100.0).astype(dtype, copy=False)
            obs_scaled = (obs_scaled - self._obs_mean) / (self._obs_std + 1e-4)
        else:
            obs_scaled = np.clip(obs_flat, -100.0, 100.0).astype(dtype, copy=False)

        _tco1 = time.perf_counter()
        wheel_pos_l_fmod = np.fmod(dof_pos[:, 4] + np.pi, 2 * np.pi) - np.pi
        wheel_pos_r_fmod = np.fmod(dof_pos[:, 5] + np.pi, 2 * np.pi) - np.pi
        critic_flat = np.asarray(np.concatenate([
            world_linvel,
            gyro, pitch, roll, commands[:, [0, 2, 3, 4]], last_actions,
            wheel_pos_l_fmod.reshape(-1, 1),
            wheel_pos_r_fmod.reshape(-1, 1),
            local_linvel,
        ] + self._vmc_obs_columns(info, noisy=False), axis=1), dtype=dtype)

        idx = env_ids if env_ids is not None else slice(None)
        action_t2 = self._action_t2[idx]
        action_t3 = self._action_t3[idx]
        dof_acc = self._cached_kin_qacc[idx]
        torques = self._cached_kin_torques[idx]
        base_mass = self._cached_base_mass[idx].reshape(-1, 1)
        base_com = self._cached_base_com[idx]
        ground_friction = self._cached_ground_friction[idx]
        feedforward_force = self._vmc_ff[idx].reshape(-1, 1)

        dof_pos_wrapped = dof_pos.copy()
        dof_pos_wrapped[:, 4] = wheel_pos_l_fmod
        dof_pos_wrapped[:, 5] = wheel_pos_r_fmod

        _tscan0 = time.perf_counter()
        _scan_data = self._box_aware_height_scan(self._cfg.terrain_scan, n)
        _tscan1 = time.perf_counter()
        if self.step_counter < 3:
            print(f"  [OBS] height_scan_obs={(_tscan1-_tscan0)*1e3:.1f}ms shape={_scan_data.shape}")
        _tobs0 = time.perf_counter()
        critic = np.concatenate([
            critic_flat,
            action_t2,
            action_t3,
            dof_acc,
            torques,
            base_mass,
            base_com,
            feedforward_force,
            dof_pos_wrapped,
            ground_friction[:, 0].reshape(-1, 1),
            ground_friction[:, 1].reshape(-1, 1),
            _scan_data,
        ], axis=1, dtype=dtype)
        _tobs1 = time.perf_counter()
        if self.step_counter < 3:
            print(f"  [OBS] critic_concat={(_tobs1-_tobs0)*1e3:.1f}ms")
        if self._obs_fifo is None:
            obs_dim = obs_scaled.shape[1]
            self._obs_fifo = np.zeros((self._num_envs, self._max_delay + 1, obs_dim), dtype=dtype)
        if self._latest_obs_scaled is None:
            self._latest_obs_scaled = np.zeros_like(self._obs_fifo[:, 0, :])
        if env_ids is not None:
            self._latest_obs_scaled[env_ids] = obs_scaled
            self._obs_fifo[env_ids] = obs_scaled[:, np.newaxis, :]
            delay_idx = np.random.randint(
                self._obs_delay_range[0], self._obs_delay_range[1] + 1, len(env_ids),
            ).astype(np.int32)
            delay_idx = np.clip(delay_idx, 0, self._max_delay)
            obs_delayed = self._obs_fifo[env_ids, self._max_delay - delay_idx, :]
        else:
            self._obs_fifo[:, :-1, :] = self._obs_fifo[:, 1:, :]
            self._latest_obs_scaled = obs_scaled
            self._obs_fifo[:, -1, :] = obs_scaled
            delay_idx = np.random.randint(
                self._obs_delay_range[0], self._obs_delay_range[1] + 1, n,
            ).astype(np.int32)
            delay_idx = np.clip(delay_idx, 0, self._max_delay)
            obs_delayed = self._obs_fifo[np.arange(n), self._max_delay - delay_idx, :]
        _tco2 = time.perf_counter()
        if self.step_counter < 3:
            print(f"  [OBS] actor_obs={(_tco1-_tco0)*1e3:.1f}ms critic_obs={(_tco2-_tco1)*1e3:.1f}ms total_obs={(_tco2-_tco0)*1e3:.1f}ms")
        return {"obs": obs_delayed, "critic": critic}

    def _box_aware_height_scan(self, scan_cfg, num_obs):
        heights = height_scan_obs(self, scan_cfg, num_obs)
        sampler = getattr(self, "_terrain_surface_sampler", None)
        if sampler is None:
            return heights
        boxes = getattr(sampler, "_boxes", None)
        if not boxes:
            return heights

        dtype = get_global_dtype()
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=dtype)
        if base_pos.shape[0] != num_obs:
            return heights

        scan_radius_x = float(max(abs(p) for p in scan_cfg.measured_points_x))
        scan_radius_y = float(max(abs(p) for p in scan_cfg.measured_points_y))

        near_mask = np.zeros(num_obs, dtype=bool)
        for box in boxes:
            bx, by, bsx, bsy = box.pos[0], box.pos[1], box.size[0], box.size[1]
            near_mask |= (
                (np.abs(base_pos[:, 0] - bx) <= bsx + scan_radius_x) &
                (np.abs(base_pos[:, 1] - by) <= bsy + scan_radius_y)
            )
        if not np.any(near_mask):
            return heights

        base_quat = np.asarray(self._backend.get_base_quat(), dtype=dtype)
        if base_quat.shape[0] != num_obs:
            return heights

        offsets = self._height_scan_offsets
        if offsets is None:
            return heights
        P = offsets.shape[0]
        if P != heights.shape[1]:
            return heights

        near_idx = np.where(near_mask)[0]
        ox, oy = offsets[:, 0], offsets[:, 1]
        bp_near = base_pos[near_idx]
        qw = base_quat[near_idx, 0]
        qx = base_quat[near_idx, 1]
        qy = base_quat[near_idx, 2]
        qz = base_quat[near_idx, 3]
        cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
        sin_yaw = 2.0 * (qw * qz + qx * qy)
        nn = len(near_idx)

        world_x = bp_near[:, 0:1] + cos_yaw[:, None] * ox[None, :] - sin_yaw[:, None] * oy[None, :]
        world_y = bp_near[:, 1:2] + sin_yaw[:, None] * ox[None, :] + cos_yaw[:, None] * oy[None, :]

        hfield_only = heights.copy()
        heights_near = heights[near_idx].copy()
        tmp_x = np.empty(world_x.shape, dtype=dtype)
        tmp_y = np.empty(world_y.shape, dtype=dtype)
        for box in boxes:
            np.subtract(world_x, box.pos[0], out=tmp_x)
            np.abs(tmp_x, out=tmp_x)
            inside_x = tmp_x <= box.size[0]
            np.subtract(world_y, box.pos[1], out=tmp_y)
            np.abs(tmp_y, out=tmp_y)
            inside = inside_x & (tmp_y <= box.size[1])
            if not np.any(inside):
                continue
            box_top = box.pos[2] + box.size[2]
            corrected = np.clip(
                bp_near[:, 2:3] - scan_cfg.vertical_offset - box_top,
                -1.0, 1.0,
            ) * scan_cfg.scale
            heights_near[inside] = np.broadcast_to(corrected, heights_near.shape)[inside]

        heights = heights.copy()
        heights[near_idx] = heights_near

        if self.step_counter < 3:
            n_overridden = int(np.sum(np.abs(heights - hfield_only) > 1e-6))
            print(f"  [BOX] boxes={len(boxes)} near_envs={nn} overridden_points={n_overridden}")

        return np.asarray(heights, dtype=dtype)

    def _vmc_obs_columns(self, info: dict, noisy: bool = True) -> list[np.ndarray]:
        """VMC 观测列: 角度 (θ, θ̇) + 腿长 (L₀, L̇₀)。

        Args:
            info: state.info dict，用于 reset 时传入初始值。
            noisy: True 加观测噪声（actor），False 返回原始值（critic）。
        """
        noise_cfg = self._cfg.noise_config
        theta_l = info.get("swing_lw_theta")
        theta_r = info.get("swing_rw_theta")
        vmc_theta_dot = info.get("_vmc_theta_dot")
        vmc_L0 = info.get("_vmc_L0")
        vmc_L0_dot = info.get("_vmc_L0_dot")
        if theta_l is None:
            theta_l = self._theta_vmc_l
        if theta_r is None:
            theta_r = self._theta_vmc_r
        if vmc_theta_dot is None:
            vmc_theta_dot = self._vmc_theta_dot
        if vmc_L0 is None:
            vmc_L0 = self._vmc_L0
        if vmc_L0_dot is None:
            vmc_L0_dot = self._vmc_L0_dot
        if noisy:
            theta_l = self._obs_noise(theta_l.reshape(-1, 1), noise_cfg.scale_vmc_phi0)
            theta_r = self._obs_noise(theta_r.reshape(-1, 1), noise_cfg.scale_vmc_phi0)
            vmc_theta_dot = self._obs_noise(vmc_theta_dot, noise_cfg.scale_vmc_phi0_dot)
            vmc_L0 = self._obs_noise(vmc_L0, noise_cfg.scale_vmc_L0)
            vmc_L0_dot = self._obs_noise(vmc_L0_dot, noise_cfg.scale_vmc_L0_dot)
        else:
            theta_l = theta_l.reshape(-1, 1)
            theta_r = theta_r.reshape(-1, 1)
        return [
            theta_l,                                  # θ 左腿摆角
            theta_r,                                  # θ 右腿摆角
            vmc_theta_dot[:, 0].reshape(-1, 1),       # θ̇ (sitad) 左腿
            vmc_theta_dot[:, 1].reshape(-1, 1),       # θ̇ (sitad) 右腿
            vmc_L0[:, 0].reshape(-1, 1),              # L₀ 左腿
            vmc_L0[:, 1].reshape(-1, 1),              # L₀ 右腿
            vmc_L0_dot[:, 0].reshape(-1, 1),          # L̇₀ (L0d) 左腿
            vmc_L0_dot[:, 1].reshape(-1, 1),          # L̇₀ (L0d) 右腿
        ]

    # =================================================================
    # 奖励系统（go2w 规范：scales + fns(ctx) + dispatch）
    # =================================================================

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            # ── 标准共享（与 go2w/go2/g1 一致） ───────────────────
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": common_rewards.ang_vel_xy,
            "orientation": common_rewards.orientation,
            "tilt_precise": self._reward_tilt_precise,
            "action_rate": common_rewards.action_rate,
            "action_smooth": self._reward_action_smooth,
            "dof_vel": self._reward_dof_vel,
            "dof_acc": common_rewards.dof_acc,
            "dof_torques_l2": common_rewards.dof_torques_l2,
            # ── Balance 特有：
            # ——速度追踪奖励 ——————————————————
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "tracking_lin_vel_precise": self._reward_tracking_lin_vel_precise,
            "tracking_ang_vel_precise": self._reward_tracking_ang_vel_precise,
            "tracking_lin_vel_quad": self._reward_tracking_lin_vel_quad,
            "tracking_ang_vel_quad": self._reward_tracking_ang_vel_quad,
            "tracking_yaw": self._reward_tracking_yaw,
            # ——高度奖励 ——————————————————
            "base_height": self._reward_base_height,
            "base_height_precise": self._reward_base_height_precise,
            "leg_length": self._reward_leg_length,
            "leg_length_rate": self._reward_leg_length_rate,
            "body_corner_clearance": self._reward_body_corner_clearance,
            # ——腿摆角 / 对称惩罚 ───────────────────────
            "swing_vertical": self._reward_swing_vertical,
            "swing_symmetry": self._reward_swing_symmetry,
            # ── 轮子空转/打滑惩罚 ───────────────────────────
            "wheel_slip": self._reward_wheel_slip,
            # ── 碰撞惩罚（机体接触地面） ────────
            "collision": self._reward_collision,
            # ── 终止条件按步惩罚 ────────────────
            "vertex_penalty": self._reward_vertex_penalty,
            # ── 站立惩罚：无命令时惩罚速度和摆角变化 ──────
            "stand": self._reward_stand,
            # ── 轮子接触惩罚 ──────────────────
            "wheel_contact": self._reward_wheel_contact,
            "wheel_power": self._reward_wheel_power,
        }

    # ── A. 姿态 / 高度 ──────────────────────────────────────────────

    def _reward_wheel_contact(self, ctx: RewardContext) -> np.ndarray:
        contact_filt = np.asarray(ctx.info.get("contact_filt", np.ones((ctx.num_envs, 2), dtype=bool)), dtype=bool)
        num_wheels_off = (2 - np.sum(contact_filt, axis=1)).astype(get_global_dtype())
        return np.asarray(num_wheels_off, dtype=get_global_dtype())

    def _reward_collision(self, ctx: RewardContext) -> np.ndarray:
        """主动碰撞惩罚：统计有多少个 penalised 刚体接触地面（轮子除外）。

        接触力 > 1.0N 视为发生碰撞，每个碰撞刚体贡献 1.0 分原始惩罚。
        经 reward dispatch 乘 scale(-1.0) 和 ctrl_dt(0.008s) 后，
        每步每碰撞体实际惩罚 ≈ -0.008，累积效应随时间线性增长。

        当前 penalised 刚体（15 个）：
          - base_box                          (机体)
          - left_front_leg_geom               (前腿左)
          - left_front_child1_leg_geom        (前腿左 child1)
          - left_front_leg_geom1              (前腿左 child2 连接段)
          - left_rear_child1_leg_geom2        (后腿左 child1 下半段)
          - left_rear_leg_geom1               (后腿左)
          - left_rear_leg_geom2               (后腿左)
          - left_rear_child1_leg_geom1        (后腿左 child1)
          - right_front_leg_geom              (前腿右)
          - right_front_child1_leg_geom       (前腿右 child1)
          - right_front_leg_geom1             (前腿右 child2 连接段)
          - right_rear_child1_leg_geom2       (后腿右 child1 下半段)
          - right_rear_leg_geom1              (后腿右)
          - right_rear_leg_geom2              (后腿右)
          - right_rear_child1_leg_geom1       (后腿右 child1)

        设计意图：碰撞惩罚不追求"一步罚死"，而是通过持续累积
        让策略形成"远离地面"的行为习惯，配合 orientation 的姿态惩罚
        形成"姿态 + 接触"双重防护。
        """
        stacked = np.asarray(ctx.info["collision_stacked"], dtype=get_global_dtype())
        return (stacked > 1.0).astype(get_global_dtype()).sum(axis=-1)
    def _reward_vertex_penalty(self, ctx: RewardContext) -> np.ndarray:
        """机体接触惩罚：步数越多扣越多。"""
        return self._vertex_touch_count.astype(get_global_dtype())

    def _reward_lin_vel_z(self, ctx: RewardContext) -> np.ndarray:
        v_z = ctx.linvel[:, 2]
        down_weight = np.where(v_z < 0, 5.0, 1.0)
        return np.square(v_z) * down_weight

    def _reward_base_height(self, ctx: RewardContext) -> np.ndarray:
        rc = self._cfg.reward_config
        sigma = float(getattr(rc, "base_height_sigma", 0.10))
        actual_height = np.asarray(ctx.info["base_height_above_ground"], dtype=get_global_dtype())
        target_height = np.asarray(ctx.info["commands"][:, 3], dtype=get_global_dtype())
        contact_filt = np.asarray(ctx.info.get("contact_filt", np.ones((ctx.num_envs, 2), dtype=bool)), dtype=bool)
        on_ground = np.any(contact_filt, axis=1)

        normal_err = actual_height - target_height
        normal_phi = np.square(normal_err) / (sigma * sigma)

        self._base_height_phi_hold = np.where(on_ground, normal_phi, self._base_height_phi_hold)
        phi = np.where(on_ground, normal_phi, self._base_height_phi_hold)
        return phi

    def _reward_base_height_precise(self, ctx: RewardContext) -> np.ndarray:
        rc = self._cfg.reward_config
        sigma = float(getattr(rc, "base_height_sigma", 0.10)) / 3.0
        actual_height = np.asarray(ctx.info["base_height_above_ground"], dtype=get_global_dtype())
        target_height = np.asarray(ctx.info["commands"][:, 3], dtype=get_global_dtype())
        err = actual_height - target_height
        return np.exp(-np.square(err) / (sigma * sigma))

    def _reward_body_corner_clearance(self, ctx: RewardContext) -> np.ndarray:
        min_z = np.asarray(ctx.info["body_corner_min_z"], dtype=get_global_dtype())
        threshold = float(getattr(self._cfg.reward_config, "body_corner_clearance_threshold", 0.08))
        below = np.maximum(0.0, 1.0 - min_z / threshold)
        return np.square(below)

    def _reward_leg_length(self, ctx: RewardContext) -> np.ndarray:
        """Roll 补偿腿长奖励：用目标高度 + roll 角分配左右腿目标长度。

        公式：
          target_left  = h_tgt + R_l * tan(roll)
          target_right = h_tgt - R_l * tan(roll)

        其中 R_l = 0.21 取轮子实际 y 偏移（非髋间距），
        roll = arcsin(-gravity_body[:, 1])（右侧低为负，R_l*tan(roll) 为负）。

        物理含义：右低时右腿蹬长、左腿收短，使机体在斜面上保持水平。
        """
        rc = self._cfg.reward_config
        h_tgt = np.asarray(ctx.info["commands"][:, 3], dtype=get_global_dtype())
        leg_l = np.asarray(ctx.info["leg_l"], dtype=get_global_dtype())
        leg_r = np.asarray(ctx.info["leg_r"], dtype=get_global_dtype())
        gravity_body = np.asarray(ctx.info["gravity_body"], dtype=get_global_dtype())

        roll = np.arcsin(np.clip(-gravity_body[:, 1], -1.0, 1.0))
        R_l = 0.21

        leg_tgt = h_tgt - 0.023
        target_left = leg_tgt + R_l * np.tan(roll)
        target_right = leg_tgt - R_l * np.tan(roll)

        err_l = leg_l - target_left
        err_r = leg_r - target_right

        sigma = float(getattr(rc, "leg_length_sigma", 0.02))
        sigma_sq = sigma * sigma

        raw = np.exp(-(np.square(err_l) + np.square(err_r)) / (2.0 * sigma_sq))
        return np.asarray(raw, dtype=get_global_dtype())

    def _reward_leg_length_rate(self, ctx: RewardContext) -> np.ndarray:
        """腿长变化率惩罚：惩罚左右腿长速度的平方和，促进平滑的腿长控制。

        计算腿长速度 (m/s)，与 dof_vel 等速率惩罚保持量纲一致。
        """
        leg_l = np.asarray(ctx.info["leg_l"], dtype=get_global_dtype())
        leg_r = np.asarray(ctx.info["leg_r"], dtype=get_global_dtype())
        prev = np.asarray(ctx.info["_prev_leg_len"], dtype=get_global_dtype())
        prev_r = prev[:, 0]
        prev_l = prev[:, 1]
        dt = float(self._cfg.ctrl_dt)
        vel_r = (leg_r - prev_r) / dt
        vel_l = (leg_l - prev_l) / dt
        return np.asarray(np.square(vel_r) + np.square(vel_l), dtype=get_global_dtype())

    def _reward_tracking_lin_vel(self, ctx: RewardContext) -> np.ndarray:
        """世界坐标系下的速度追踪：将机体坐标系指令旋转到世界坐标系，再与世界坐标系速度比较。

        好处：转向时速度追踪更稳定，不会因 pitch 变化导致机体坐标系速度分量突变。
        静态 σ：tracking_sigma_dynamic=False 时直接用 sigma_base（不再对 |cmd|≤1 减半）；
        True 时才按 cmd² 放大：cmd=1m/s → σ=0.25，cmd=2m/s → σ=1.0。

        接触门控已按 Fudan 对齐关闭：台阶上腿段蹭地形时仍保留速度压力；
        防"用腿撑地取巧"改由 collision 惩罚负责，而不是把 tracking 归零。
        """
        commands = ctx.info["commands"]
        vx = ctx.info["body_vx"]
        gravity_body = ctx.info["gravity_body"]
        cos_pitch = np.cos(np.arcsin(np.clip(gravity_body[:, 0], -1.0, 1.0)))
        vx_horiz = vx * cos_pitch
        lin_vel_error = np.square(commands[:, 0] - vx_horiz)
        sigma_base = ctx.tracking_sigma_lin if ctx.tracking_sigma_lin is not None else ctx.tracking_sigma
        sigma_dyn = np.maximum(sigma_base * sigma_base, sigma_base * np.square(commands[:, 0]))
        sigma = sigma_dyn if ctx.tracking_sigma_dynamic else sigma_base
        raw = np.exp(-lin_vel_error / sigma)
        return np.asarray(raw, dtype=get_global_dtype())

    def _reward_tracking_ang_vel(self, ctx: RewardContext) -> np.ndarray:
        """和 common_rewards.tracking_ang_vel 一样的高斯形式，只是在末尾加姿态门控。

        作用：避免 policy 为了追角速度指令（转向）而甩倾斜/乱晃腿——
        姿态好才给全额（×1.2），姿态差打折扣（×0.6），和直线速度的门控保持一致。
        静态 σ：tracking_sigma_dynamic=False 时直接用 sigma_base；
        True 时才按 cmd² 放大：cmd=1rad/s → σ=0.25，cmd=2rad/s → σ=1.0。
        """
        commands = ctx.info["commands"]
        ang_vel_error = np.square(commands[:, 2] - ctx.gyro[:, 2])
        sigma_base = ctx.tracking_sigma_ang if ctx.tracking_sigma_ang is not None else ctx.tracking_sigma
        sigma_dyn = np.maximum(sigma_base * sigma_base, sigma_base * np.square(commands[:, 2]))
        sigma = sigma_dyn if ctx.tracking_sigma_dynamic else sigma_base
        raw = np.exp(-ang_vel_error / sigma)
        return np.asarray(raw, dtype=get_global_dtype())

    def _reward_tracking_lin_vel_precise(self, ctx: RewardContext) -> np.ndarray:
        """精度加强版线速度追踪：σ = 原 σ / 10，与 jumpvf 同形（Fudan ``tracking_lin_vel_enhance``）。

            raw = (exp(-err² / (σ/10)) - 1) * 2      # ∈[-2, 0]，完美追踪为 0

        恒 ≤ 0 的纯惩罚：误差小 → 接近 0，误差大 → 趋向 -2。
        （此前实现是 ``exp(...) - 0.5``，居中型 ∈[-0.5, +0.5]，与 jumpvf 不等价；
        梯度只有 jumpvf 的 1/2，惩罚幅度只有 1/4。）

        不做触地门控（与 Fudan 一致）：台阶上腿段蹭到地形时正是需要“保持速度”
        压力的时刻，归零会让该项恰好失效；防“用腿撑地取巧”改由 collision 惩罚负责。
        """
        commands = ctx.info["commands"]
        vx = ctx.info["body_vx"]
        gravity_body = ctx.info["gravity_body"]
        cos_pitch = np.cos(np.arcsin(np.clip(gravity_body[:, 0], -1.0, 1.0)))
        vx_horiz = vx * cos_pitch
        lin_vel_error = np.square(commands[:, 0] - vx_horiz)
        sigma_base = (ctx.tracking_sigma_lin if ctx.tracking_sigma_lin is not None else ctx.tracking_sigma) / 10.0
        sigma_dyn = np.maximum(sigma_base * sigma_base, sigma_base * np.square(commands[:, 0]))
        sigma = sigma_dyn if ctx.tracking_sigma_dynamic else sigma_base
        raw = (np.exp(-lin_vel_error / sigma) - 1.0) * 2.0
        return np.asarray(raw, dtype=get_global_dtype())

    def _reward_tracking_lin_vel_quad(self, ctx: RewardContext) -> np.ndarray:
        """二次惩罚项：对大误差施加无上限的二次惩罚，弥补 exp 奖励在大误差时梯度消失的缺陷。

        返回 raw = -(error_x² + error_y²)，由 scale 控制惩罚强度。
        推荐 scale 为负值（如 -0.5），误差 1m/s 时惩罚 ~0.5，误差 2m/s 时惩罚 ~2.0。

        与 tracking 主项一致，接触门控已关闭（碰地不豁免速度误差）。
        """
        commands = ctx.info["commands"]
        vx = ctx.info["body_vx"]
        gravity_body = ctx.info["gravity_body"]
        cos_pitch = np.cos(np.arcsin(np.clip(gravity_body[:, 0], -1.0, 1.0)))
        vx_horiz = vx * cos_pitch
        lin_vel_error = np.square(commands[:, 0] - vx_horiz)
        return np.asarray(lin_vel_error, dtype=get_global_dtype())

    def _reward_tracking_ang_vel_precise(self, ctx: RewardContext) -> np.ndarray:
        """精度加强版角速度追踪：σ = 原 σ / 10，与 jumpvf 同形。

            raw = exp(-err² / (σ/10))      # ∈[0, 1]，完美追踪为 1

        （此前实现是 ``exp(...) - 0.5``，居中型 ∈[-0.5, +0.5]。）
        """
        commands = ctx.info["commands"]
        ang_vel_error = np.square(commands[:, 2] - ctx.gyro[:, 2])
        sigma_base = (ctx.tracking_sigma_ang if ctx.tracking_sigma_ang is not None else ctx.tracking_sigma) / 10.0
        sigma_dyn = np.maximum(sigma_base * sigma_base, sigma_base * np.square(commands[:, 2]))
        sigma = sigma_dyn if ctx.tracking_sigma_dynamic else sigma_base
        raw = np.exp(-ang_vel_error / sigma)
        return np.asarray(raw, dtype=get_global_dtype())

    def _reward_tracking_ang_vel_quad(self, ctx: RewardContext) -> np.ndarray:
        """二次惩罚项：对大角速度误差施加无上限惩罚，弥补 exp 奖励在大误差时梯度消失的缺陷。

        返回 raw = error²，由 scale 控制惩罚强度。
        推荐 scale 为负值（如 -0.3），误差 1 rad/s 时惩罚 ≈0.3，误差 2 rad/s 时惩罚 ≈1.2。
        """
        commands = ctx.info["commands"]
        ang_vel_error = np.square(commands[:, 2] - ctx.gyro[:, 2])
        return np.asarray(ang_vel_error, dtype=get_global_dtype())

    def _reward_tracking_yaw(self, ctx: RewardContext) -> np.ndarray:
        yaw_error = np.asarray(ctx.info.get("yaw_error", np.zeros(ctx.num_envs, dtype=get_global_dtype())), dtype=get_global_dtype())
        sigma = ctx.tracking_sigma_yaw if ctx.tracking_sigma_yaw is not None else (ctx.tracking_sigma_ang if ctx.tracking_sigma_ang is not None else ctx.tracking_sigma)
        raw = np.exp(-np.square(yaw_error) / sigma)
        return np.asarray(raw, dtype=get_global_dtype())

    def _reward_swing_vertical(self, ctx: RewardContext) -> np.ndarray:
        """腿杆偏离动态目标摆角的惩罚。"""
        err_rw = ctx.info["swing_err_rw"]
        err_lw = ctx.info["swing_err_lw"]
        raw = np.abs(err_rw) + np.abs(err_lw)
        return np.asarray(raw, dtype=get_global_dtype())


    def _reward_swing_vel(self, ctx: RewardContext) -> np.ndarray:
        """摆角速度惩罚（返回 ≤ 0）—— 从 info 读，不需要假设关节索引"""
        return np.asarray(ctx.info["swing_vel_sq"], dtype=get_global_dtype())

    def _reward_swing_symmetry(self, ctx: RewardContext) -> np.ndarray:
        """左右腿不对称惩罚：(θ_r - θ_l)²，对照复旦 nominal_state。"""
        rw_theta = ctx.info["swing_rw_theta"]
        lw_theta = ctx.info["swing_lw_theta"]
        ang_diff = np.square(rw_theta - lw_theta)
        return ang_diff.astype(get_global_dtype())

    def _reward_upright_swing_bonus(self, ctx: RewardContext) -> np.ndarray:
        """离散奖励：tilt < 8° 且 swing 误差 < 8° → +1，否则 0。"""
        tilt = np.asarray(ctx.info["tilt"], dtype=get_global_dtype())
        err_rw = np.abs(np.asarray(ctx.info["swing_err_rw"], dtype=get_global_dtype()))
        err_lw = np.abs(np.asarray(ctx.info["swing_err_lw"], dtype=get_global_dtype()))
        swing_err = np.maximum(err_rw, err_lw)

        tilt_th = np.deg2rad(7.5)
        swing_th = np.deg2rad(20)

        return np.asarray((tilt < tilt_th) & (swing_err < swing_th), dtype=get_global_dtype())

    def _reward_stand(self, ctx: RewardContext) -> np.ndarray:
        """站立惩罚：无速度命令时惩罚机体速度变化。

        当 |cmd_vx| 和 |cmd_yaw| 都小于阈值时，认为应该站立不动，
        此时惩罚：
          1. 机体水平速度（vx² + vy²）
          2. 机体角速度（roll² + pitch²）
          3. 机体垂直速度（vz²）

        用 sigmoid 软切换平滑过渡，避免硬切换导致梯度不连续：
          gate = σ((threshold - |cmd|) / width)
        |cmd| << threshold → gate ≈ 1（站立，全额惩罚）
        |cmd| >> threshold → gate ≈ 0（运动，不惩罚）
        """
        rc = self._cfg.reward_config
        commands = ctx.info["commands"]
        cmd_vx = commands[:, 0]
        cmd_yaw = commands[:, 2]

        vx_threshold = float(rc.stand_cmd_linvel_threshold)
        yaw_threshold = float(rc.stand_cmd_angvel_threshold)
        width = 0.03

        gate_vx = 1.0 / (1.0 + np.exp(np.clip((np.abs(cmd_vx) - vx_threshold) / width, -50, 50)))
        gate_yaw = 1.0 / (1.0 + np.exp(np.clip((np.abs(cmd_yaw) - yaw_threshold) / width, -50, 50)))
        gate = gate_vx * gate_yaw

        # 原地旋转模式：强制开启 stand 门控，惩罚水平移动
        spin_mask = np.asarray(ctx.info.get("spin_mask", np.zeros_like(cmd_vx, dtype=bool)), dtype=bool)
        gate = np.where(spin_mask, 1.0, gate)

        base_contact_force = ctx.info.get("base_body_contact_force", np.zeros((ctx.num_envs, 1)))
        touch_mask = base_contact_force.flatten() > 1.0
        gate = np.where(touch_mask, 1.0, gate)

        # 落地前跳过：reset 后前 30 步（0.24s）机体还在下落，不计入 stand
        landing_delay = 40
        landed = self._episode_steps >= landing_delay
        gate = np.where(landed, gate, 0.0)

        # 原地自转时用世界坐标系速度，确保位置不漂移
        world_vx = ctx.info.get("world_vx", ctx.linvel[:, 0])
        world_vy = ctx.info.get("world_vy", ctx.linvel[:, 1])
        vx_penalty = np.where(spin_mask, world_vx, ctx.linvel[:, 0])
        vy_penalty = np.where(spin_mask, world_vy, ctx.linvel[:, 1])
        vel_sq = np.square(vx_penalty) + np.square(vy_penalty)
        ang_vel_xy = np.square(ctx.gyro[:, 0]) + np.square(ctx.gyro[:, 1])
        lin_vel_z = np.square(ctx.linvel[:, 2])

        w_vel = float(rc.stand_vel_weight)
        w_angvel = float(rc.stand_angvel_weight)
        w_linvelz = float(rc.stand_linvelz_weight)
        raw = w_vel * vel_sq + w_angvel * ang_vel_xy + w_linvelz * lin_vel_z
        return np.asarray(gate * raw, dtype=get_global_dtype())

    def _reward_dof_vel(self, ctx: RewardContext) -> np.ndarray:
        """关节速度平方惩罚，只罚腿关节（0,1,2,3），不罚轮子（4,5）。"""
        assert ctx.dof_vel is not None
        return np.asarray(np.sum(np.square(ctx.dof_vel[:, :4]), axis=1), dtype=get_global_dtype())

    def _reward_action_smooth(self, ctx: RewardContext) -> np.ndarray:
        """动作二阶差分惩罚，只罚腿动作（0,1,2,3），不罚轮子（4,5）。"""
        current = np.asarray(ctx.info["current_actions"], dtype=get_global_dtype())
        last = np.asarray(ctx.info["last_actions"], dtype=get_global_dtype())
        second_last = np.asarray(
            ctx.info.get("_prev2_actions", np.zeros_like(current)),
            dtype=get_global_dtype(),
        )
        # 一阶用 last_actions (a_{t-1})，二阶用 _prev2_actions (a_{t-2})，同为真实动作
        diff = current[:, :4] - 2 * last[:, :4] + second_last[:, :4]
        return np.asarray(np.sum(np.square(diff), axis=1), dtype=get_global_dtype())

    def _reward_tilt_precise(self, ctx: RewardContext) -> np.ndarray:
        rc = self._cfg.reward_config
        threshold = float(getattr(rc, "tilt_precise_sigma", 0.07))
        tilt = np.asarray(ctx.info["tilt"], dtype=get_global_dtype())
        return np.clip(1.0 - tilt / threshold, 0.0, 1.0)

    def _reward_wheel_power(self, ctx: RewardContext) -> np.ndarray:
        """轮子惩罚：贴地=(|τ|/3.5)²，空中=(|τ·ω|/P_max)² 按功率惩罚。"""
        torques = ctx.info.get("torques") if ctx.info is not None else None
        if torques is None:
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())

        contact_filt = ctx.info.get("contact_filt")
        lw_vel = ctx.info.get("lw_vel")
        rw_vel = ctx.info.get("rw_vel")
        power_max = float(getattr(self._cfg.reward_config, 'wheel_air_power_max', 100.0))

        tau_left = np.abs(torques[:, 4])
        tau_right = np.abs(torques[:, 5])
        left_penalty = (tau_left / 3.5) ** 2
        right_penalty = (tau_right / 3.5) ** 2

        if contact_filt is not None and lw_vel is not None and rw_vel is not None:
            contact_filt = np.asarray(contact_filt, dtype=bool)
            left_air = ~contact_filt[:, 0]
            right_air = ~contact_filt[:, 1]
            left_power = tau_left * np.abs(np.asarray(lw_vel, dtype=get_global_dtype()))
            right_power = tau_right * np.abs(np.asarray(rw_vel, dtype=get_global_dtype()))
            left_penalty = np.where(left_air, (left_power / power_max) ** 2, left_penalty)
            right_penalty = np.where(right_air, (right_power / power_max) ** 2, right_penalty)

        penalty = left_penalty + right_penalty
        return penalty.astype(get_global_dtype())

    def _reward_wheel_slip(self, ctx: RewardContext) -> np.ndarray:
        """轮子空转/打滑惩罚：按左右轮分别计算，考虑差速转向。

        左轮期望线速度 = vx - 0.5*ω*L
        右轮期望线速度 = vx + 0.5*ω*L
        """
        rw_vel = ctx.info["rw_vel"]
        lw_vel = ctx.info["lw_vel"]
        vx = ctx.info["body_vx"]
        wyaw = ctx.gyro[:, 2]
        L = 0.380

        expected_left = vx - 0.5 * wyaw * L
        expected_right = vx + 0.5 * wyaw * L

        slip_left = np.square(lw_vel * 0.06 - expected_left)
        slip_right = np.square(rw_vel * 0.06 - expected_right)
        return (slip_left + slip_right) * 0.5

    # ── 主计算：构建 context + dispatch ────────────────────────────
    def _compute_reward(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity_body: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        base_pos: np.ndarray,
        num_envs: int,
    ) -> np.ndarray:
        rc = self._cfg.reward_config
        dtype = get_global_dtype()

        powerless_steps = int(self._curriculum.effective_powerless_duration / self._cfg.ctrl_dt)
        powerless_mask = (self._episode_steps < powerless_steps) & self._powerless_active

        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=num_envs,
            default_angles=np.asarray(self.default_angles, dtype=self._np_dtype),
            tracking_sigma=rc.tracking_sigma,
            tracking_sigma_lin=self._curriculum.tracking_sigma_lin,
            tracking_sigma_ang=self._curriculum.tracking_sigma_ang,
            tracking_sigma_yaw=self._curriculum.tracking_sigma_yaw,
            tracking_sigma_dynamic=bool(getattr(rc, 'tracking_sigma_dynamic', True)),
            wheel_torque_start=float(getattr(rc, 'wheel_torque_penalty_start', 2.5)),
            wheel_power_limit=getattr(rc, "wheel_power_limit", None),
            wheel_torque_max=float(getattr(rc, 'wheel_torque_penalty_max', 3.5)),
            base_height_target=info["commands"][:, 3],
            base_height=info.get("base_height_above_ground", np.zeros(num_envs, dtype=dtype)),
            gravity=gravity_body,
        )
        reward = run_reward_dispatch(
            scales=rc.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=True,
            ctrl_dt=float(self._cfg.ctrl_dt),
            clip_range=float(getattr(rc, "clip_range", 0.0)),
            clip_ranges=getattr(rc, "clip_ranges", None) or None,
        )
        reward[powerless_mask] = 0.0
        return reward

    # -----------------------------------------------------------------
    # commands 刷新
    # -----------------------------------------------------------------

    def _update_commands(self, info: dict) -> None:
        """标准项目写法：初始化 + 按 resampling_time 周期性刷新命令。

        - steps: info["steps"] 为 (num_envs,) uint32 数组，每环境独立递增
        - resample: steps % interval_steps == 0 时刷新对应 env 的 cmd
        - 保持 5 维：[vx, vy, vyaw, height, jump]，jump 槽位恒 0（跳跃功能已移除）
        - 调用 zero_small_xy_commands + rel_standing_envs 与 reset 时保持一致
        """
        dtype = get_global_dtype()
        n = self._num_envs

        # —— ① 初始化：commands 缺失 / 形状不对时兜底 ——
        commands = np.asarray(info.get("commands"), dtype=dtype)
        if commands is None or commands.shape != (n, 5):
            default_height = float(self._cfg.reward_config.base_height_target)
            commands = np.concatenate(
                [np.zeros((n, 3), dtype=dtype), np.full((n, 1), default_height, dtype=dtype), np.zeros((n, 1), dtype=dtype)],
                axis=1,
            )

        # —— ② 每环境步数计数（对齐 go2w：只读，框架每步自动递增）——
        steps = np.asarray(
            info.get("steps", np.zeros((n,), dtype=np.uint32)),
            dtype=np.uint32,
        )

        # —— ③ 周期性重采样（resampling_time > 0 才开启）——
        resampling_time = float(getattr(self._cfg.commands, "resampling_time", 0.0))
        resample_mask = np.zeros(n, dtype=bool)
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / float(self._cfg.ctrl_dt))), 1)
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                resample_indices = np.where(resample_mask)[0]
                vx_low_arr = np.asarray(
                    self._curriculum.cmd_vel_range_env[resample_indices, 0], dtype=dtype
                )
                vx_high_arr = np.asarray(
                    self._curriculum.cmd_vel_range_env[resample_indices, 1], dtype=dtype
                )

                # ── 地形相关 cmdvx 限制 ──
                terrain_types = None
                if hasattr(self, '_spawn'):
                    _get_terrain = getattr(self._spawn, "get_terrain_type_for", None)
                    if _get_terrain is not None:
                        terrain_types = _get_terrain(resample_indices)
                if terrain_types is not None:
                    t_vx_low, t_vx_high = get_terrain_cmdvx_limits(terrain_types)
                    t_yaw_low, t_yaw_high = get_terrain_cmd_yaw_limits(terrain_types)
                    forward_bias = get_terrain_vx_forward_bias(terrain_types)
                    vx_low_arr = np.maximum(vx_low_arr, t_vx_low)
                    vx_high_arr = np.minimum(vx_high_arr, t_vx_high)
                else:
                    forward_bias = None
                self._eff_vx_bounds[resample_indices, 0] = vx_low_arr
                self._eff_vx_bounds[resample_indices, 1] = vx_high_arr

                # ── 模式化命令采样: 前进 / 转向 / 原地自转 ──
                if terrain_types is not None:
                    stairs_mask = np.array(
                        [t in ("pyramid_stairs", "pyramid_stairs_inv", "plateau") for t in terrain_types],
                        dtype=bool,
                    )
                else:
                    stairs_mask = np.zeros(num_resample, dtype=bool)
                fwd_prob = np.where(
                    stairs_mask, CMD_MODE_PROBS_STAIRS[0], CMD_MODE_PROBS_DEFAULT[0]
                ).astype(dtype)
                turn_prob = np.where(
                    stairs_mask, CMD_MODE_PROBS_STAIRS[1], CMD_MODE_PROBS_DEFAULT[1]
                ).astype(dtype)
                mode_r = np.random.uniform(size=num_resample).astype(dtype)
                is_forward = mode_r < fwd_prob
                is_turn = (mode_r >= fwd_prob) & (mode_r < fwd_prob + turn_prob)
                is_rotate = mode_r >= fwd_prob + turn_prob

                vx_cmd = np.random.uniform(vx_low_arr, vx_high_arr, size=(num_resample,)).astype(dtype)
                if np.any(is_rotate):
                    vx_cmd[is_rotate] = 0.0

                yaw_cmd = np.random.normal(0.0, 2.4, size=(num_resample,)).astype(dtype)
                if np.any(is_forward):
                    yaw_cmd[is_forward] = np.random.normal(
                        0.0, 0.8, size=(int(is_forward.sum()),),
                    ).astype(dtype)
                if np.any(is_rotate):
                    yaw_cmd[is_rotate] = np.random.choice(
                        [-1.0, 1.0], size=(int(is_rotate.sum()),),
                    ).astype(dtype) * np.abs(np.random.normal(
                        4.0, 1.0, size=(int(is_rotate.sum()),),
                    )).astype(dtype)
                new_cmd = np.asarray(
                    np.stack([vx_cmd, np.zeros(num_resample, dtype=dtype), yaw_cmd], axis=1),
                    dtype=dtype,
                )
                if terrain_types is not None:
                    new_cmd[:, 0] = np.clip(new_cmd[:, 0], vx_low_arr, vx_high_arr)
                    # ── 台阶地形: 前进偏好 (80% forward) ──
                    stairs_mask = np.array(
                        [t in ("pyramid_stairs", "pyramid_stairs_inv", "plateau") for t in terrain_types],
                        dtype=bool,
                    )
                    if np.any(stairs_mask):
                        stairs_idx = np.where(stairs_mask)[0]
                        fwd_prob = float(forward_bias[stairs_mask][0])
                        go_forward = np.random.uniform(size=(int(np.count_nonzero(stairs_mask)),)) < fwd_prob
                        for k, idx in enumerate(stairs_idx):
                            if go_forward[k]:
                                new_cmd[idx, 0] = np.random.uniform(0.0, vx_high_arr[idx])
                            elif new_cmd[idx, 0] > 0.0:
                                new_cmd[idx, 0] = -new_cmd[idx, 0]

                zero_small_xy_commands(new_cmd, threshold=0.05)
                self._spin_mask[resample_indices] = False
                high_yaw = np.abs(new_cmd[:, 2]) > 3.0
                if np.any(high_yaw):
                    new_cmd[high_yaw, 0] = 0.0
                    self._spin_mask[resample_indices[high_yaw]] = True
                standing_prob = float(getattr(self._cfg.commands, "rel_standing_envs", 0.0))
                if standing_prob > 0.0:
                    standing = np.random.uniform(size=(num_resample,)) < min(standing_prob, 1.0)
                    new_cmd[standing, :3] = 0.0
                if hasattr(self._cfg.commands, "height_limit"):
                    hlim = np.asarray(self._cfg.commands.height_limit, dtype=dtype)
                    hlow, hhigh = float(hlim[0]), float(hlim[1])
                    if terrain_types is not None:
                        terrain_difficulty = None
                        if hasattr(self, '_spawn'):
                            _get_diff = getattr(self._spawn, "get_terrain_difficulty_for", None)
                            if _get_diff is not None:
                                terrain_difficulty = _get_diff(resample_indices)
                        use_uni, h_uni_low, h_uni_high, h_beta = get_terrain_height_params(terrain_types, terrain_difficulty)
                        uni_mask = use_uni.astype(bool)
                        beta_mask = ~uni_mask
                        height = np.empty((num_resample, 1), dtype=dtype)
                        if uni_mask.any():
                            height[uni_mask] = np.random.uniform(
                                h_uni_low[uni_mask, None], h_uni_high[uni_mask, None],
                                size=(int(uni_mask.sum()), 1),
                            ).astype(dtype)
                        if beta_mask.any():
                            a = h_beta[beta_mask, 0:1]
                            b = h_beta[beta_mask, 1:2]
                            raw = np.random.beta(a, b, size=(int(beta_mask.sum()), 1)).astype(dtype)
                            height[beta_mask] = hlow + raw * (hhigh - hlow)
                    else:
                        height = np.asarray(
                            np.random.uniform(hlow, hhigh, size=(num_resample, 1)),
                            dtype=dtype,
                        )
                else:
                    default_height = float(self._cfg.reward_config.base_height_target)
                    height = np.full((num_resample, 1), default_height, dtype=dtype)
                # 第 5 维是预留的跳跃命令槽位，跳跃功能已移除，恒为 0（保留观测维度）
                jump = np.zeros((num_resample, 1), dtype=dtype)
                commands[resample_mask] = np.concatenate([new_cmd, height, jump], axis=1)

        info["commands"] = commands
        info["_cmd_resample_mask"] = resample_mask

    def _update_contact_filter(self, sensors: dict) -> None:
        """轮子接触滤波：OR 两帧防抖，结果写入 ``_contact_filt`` 供奖励/观测使用。"""
        wheel_contact = np.column_stack([
            sensors["left_wheel_contact_force"].flatten(),
            sensors["right_wheel_contact_force"].flatten(),
        ])
        contact = wheel_contact > 1.0
        self._contact_filt = np.logical_or(contact, self._last_contacts)
        self._last_contacts = contact.copy()


    # -----------------------------------------------------------------
    # update_state：状态计算 → 塞 info → 调用 _compute_reward → 构建 obs
    # -----------------------------------------------------------------

    def update_state(self, state: NpEnvState) -> NpEnvState:
        _t0 = time.perf_counter()
        self._update_commands(state.info)
        self._curriculum.step(self.step_counter)

        _t1 = time.perf_counter()
        sensors = self._read_sensors()
        _t2 = time.perf_counter()
        nan_mask = np.zeros(self._num_envs, dtype=bool)
        for k, v in sensors.items():
            env_nan = np.any(np.isnan(v), axis=1)
            if np.any(env_nan):
                nan_envs = np.where(env_nan)[0]
                print(f"[NaN DETECT] sensor '{k}' has NaN at envs={nan_envs[:5].tolist()} step={self.step_counter}")
                nan_mask |= env_nan
        if np.any(nan_mask):
            for k, v in sensors.items():
                sensors[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        commands = state.info["commands"]
        _dt = float(self._cfg.ctrl_dt)

        self._update_contact_filter(sensors)

        self._update_yaw(state.info, commands, _dt)

        rw_theta = self._theta_vmc_r.reshape(-1)
        lw_theta = self._theta_vmc_l.reshape(-1)
        _t3 = time.perf_counter()
        tilt, base_body_contact_force, terminated = self._detect_termination(
            sensors["gravity_body"], sensors["base_pos"], sensors["base_body_contact_force"], commands, sensors["upvector"], sensors["local_linvel"],
        )
        terminated = terminated | nan_mask
        _t4 = time.perf_counter()
        kin = self._compute_kinematics(sensors["dof_vel"], rw_theta, lw_theta, _dt)
        _t5 = time.perf_counter()

        _t6 = time.perf_counter()
        self._populate_info(state.info, sensors, tilt, base_body_contact_force, terminated, rw_theta, lw_theta, kin)
        _t7 = time.perf_counter()

        self._prev_leg_len[:, 0] = self._vmc_L0[:, 1]
        self._prev_leg_len[:, 1] = self._vmc_L0[:, 0]

        rc = self._cfg.reward_config

        _t8 = time.perf_counter()
        reward = self._compute_reward(
            state.info, sensors["world_linvel"], sensors["gyro"], sensors["gravity_body"],
            sensors["dof_pos"], sensors["dof_vel"], sensors["base_pos"], self._num_envs,
        )
        _t9 = time.perf_counter()

        self._prev_body_vx = sensors["local_linvel"][:, 0].copy()
        self._accumulate_curriculum_stats(state.info, sensors["local_linvel"], commands, sensors["gyro"])

        if self._cfg.debug_enabled:
            self._debug_print(state.info, reward, commands, sensors, tilt, base_body_contact_force)

        _t10 = time.perf_counter()
        obs = self._compute_obs(
            state.info, sensors["gyro"], sensors["gravity_body"], sensors["dof_pos"],
            sensors["dof_vel"],
            local_linvel=sensors["local_linvel"],
            world_linvel=sensors["world_linvel"],
        )
        _t11 = time.perf_counter()
        for k, v in obs.items():
            if np.any(np.isnan(v)):
                nan_envs = np.where(np.isnan(v).any(axis=1))[0]
                print(f"[NaN DETECT] obs['{k}'] has NaN at envs={nan_envs[:5].tolist()} step={self.step_counter}")
        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        done = state.terminated | state.truncated
        if np.any(done):
            done_indices = np.where(done)[0]
            stats = self._spawn.update_on_done(
                done_indices,
                self._backend.get_base_pos()[done_indices],
                episode_tracking_reward_sum=self._episode_tracking_reward_sum,
                episode_steps=self._episode_steps,
            )
            if stats:
                if "log" not in state.info:
                    state.info["log"] = {}
                for k, v in stats.items():
                    key = f"terrain_curriculum/{k}"
                    if k in ("num_promoted", "num_demoted", "num_skipped", "mean_walked"):
                        state.info["log"][key] = float(state.info["log"].get(key, 0.0) + v)
                    else:
                        state.info["log"][key] = float(v)
        _t_total = time.perf_counter() - _t0
        self._update_state_timings = {
            'sensor_read': (_t2 - _t1) * 1000,
            'pre_step_total': getattr(self, '_pre_step_time_ms', 0.0),
            'detect_term': (_t4 - _t3) * 1000,
            'kinematics': (_t5 - _t4) * 1000,
            'populate_info': (_t7 - _t6) * 1000,
            'compute_reward': (_t9 - _t8) * 1000,
            'debug_print': (_t10 - _t9) * 1000,
            'compute_obs': (_t11 - _t10) * 1000,
            'total': _t_total * 1000,
        }
        if hasattr(self, '_pre_step_time_ms'):
            object.__setattr__(self, '_pre_step_time_ms', 0.0)
        if self.step_counter < 3:
            print(f"[TIMING] step={self.step_counter} read_sensors={(_t2-_t1)*1e3:.1f}ms "
                  f"detect_term={(_t4-_t3)*1e3:.1f}ms compute_kin={(_t5-_t4)*1e3:.1f}ms "
                  f"populate_info={(_t7-_t6)*1e3:.1f}ms compute_reward={(_t9-_t8)*1e3:.1f}ms "
                  f"compute_obs={(_t11-_t10)*1e3:.1f}ms total={_t_total*1e3:.1f}ms")
        return state

    # -----------------------------------------------------------------
    # update_state 子步骤
    # -----------------------------------------------------------------

    def _read_sensors(self) -> dict[str, np.ndarray]:
        quat = self.get_base_quat()
        return {
            "gyro": self.get_gyro(),
            "upvector": self.get_upvector(),
            "gravity_body": self._compute_gravity_body(quat),
            "dof_pos": self.get_dof_pos(),
            "dof_vel": self.get_dof_vel(),
            "base_pos": self.get_base_pos(),
            "world_linvel": self.get_world_linvel(),
            "local_linvel": self.get_local_linvel(),
            "left_wheel_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.left_wheel_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "right_wheel_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.right_wheel_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "base_body_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.base_body_contact_force), dtype=get_global_dtype(),
                ).reshape(-1, 3),
                axis=1,
            ).reshape(-1, 1)
            + np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.base_center_contact_force), dtype=get_global_dtype(),
                ).reshape(-1, 3),
                axis=1,
            ).reshape(-1, 1),
            "raw_base_body_contact_force": np.asarray(
                self._backend.get_sensor_data(self._cfg.sensor.base_body_contact_force), dtype=get_global_dtype(),
            ),
            "left_front_leg_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.left_front_leg_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "left_front_child1_leg_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.left_front_child1_leg_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "left_front_leg1_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.left_front_leg1_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "left_rear_child1_leg2_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.left_rear_child1_leg2_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "left_rear_leg_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.left_rear_leg_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "left_rear_leg2_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.left_rear_leg2_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "left_rear_child1_leg1_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.left_rear_child1_leg1_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "right_front_leg_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.right_front_leg_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "right_front_child1_leg_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.right_front_child1_leg_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "right_front_leg1_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.right_front_leg1_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "right_rear_child1_leg2_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.right_rear_child1_leg2_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "right_rear_leg_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.right_rear_leg_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "right_rear_leg2_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.right_rear_leg2_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
            "right_rear_child1_leg1_contact_force": np.linalg.norm(
                np.asarray(
                    self._backend.get_sensor_data(self._cfg.sensor.right_rear_child1_leg1_contact_force), dtype=get_global_dtype(),
                )[:, -3:],
                axis=1,
            ).reshape(-1, 1),
        }

    def _update_yaw(self, info: dict, commands: np.ndarray, _dt: float) -> None:
        base_quat = self.get_base_quat()
        base_yaw = np_yaw_from_quat(base_quat)

        first_step_mask = self._episode_steps == 0
        if np.any(first_step_mask):
            self._target_yaw[first_step_mask] = base_yaw[first_step_mask]

        resample_mask = info.get("_cmd_resample_mask", np.zeros_like(first_step_mask, dtype=bool))
        if np.any(resample_mask):
            self._target_yaw[resample_mask] = base_yaw[resample_mask]

        self._target_yaw += commands[:, 2] * _dt
        self._target_yaw = np_wrap_to_pi(self._target_yaw)

        yaw_error = np_wrap_to_pi(self._target_yaw - base_yaw)
        info["yaw_error"] = yaw_error

    @staticmethod
    def _compute_swing_angles(rw: np.ndarray, lw: np.ndarray, hip_z: float = 0.141) -> tuple[np.ndarray, np.ndarray]:
        """保留兼容接口，实际不再使用传感器数据。"""
        return rw, lw

    def _term_loosen_active(self, rc: Any) -> bool:
        """课程完成后是否收敛终止：True 时只剩 tilt（翻车）+ 地形出界终止。

        ``term_loosen_from_step`` 用控制步计数（``self.step_counter``，每 iter =
        ``num_steps_per_env`` 步；0 = 立即生效）。首次生效打印一次，便于在日志里定位切换点。
        """
        if not getattr(rc, "term_loosen_enabled", False):
            return False
        active = int(self.step_counter) >= int(getattr(rc, "term_loosen_from_step", 0))
        if active and not self._term_loosened:
            self._term_loosened = True
            print(
                f"[TERM] 课程完成收敛 step={int(self.step_counter)} → 只保留 tilt 终止："
                f"tilt>{float(rc.tilt_limit_tight):.3f}rad"
                f"（vertex_touch / swing / min_height 三种姿态终止已关闭，地形出界仍生效）",
                flush=True,
            )
        return active

    def _detect_termination(self, gravity_body: np.ndarray, base_pos: np.ndarray, base_body_contact_force: np.ndarray, commands: np.ndarray, upvector: np.ndarray, local_linvel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rc = self._cfg.reward_config
        tilt = np.arccos(np.clip(gravity_body[:, 2], 0.0, 1.0))

        powerless_steps = int(self._curriculum.effective_powerless_duration / self._cfg.ctrl_dt)
        powerless_mask = (self._episode_steps < powerless_steps) & self._powerless_active

        base_h_scan = combined_base_height_from_scan(self)
        body_corner_min_z = base_h_scan - np.abs(gravity_body[:, 0]) * 0.28 - np.abs(gravity_body[:, 1]) * 0.14
        vertex_touch = (body_corner_min_z < 0.03) & ~powerless_mask

        angle_limit = rc.tilt_limit_tight
        body_tilt_too_high = (tilt > angle_limit) & ~powerless_mask

        base_height = base_h_scan + 0.141 * upvector[:, 2]
        linvel_norm = np.linalg.norm(local_linvel, axis=1)
        stationary = linvel_norm < 0.1
        height_collapse = (base_height < rc.min_height_tight) & stationary & ~powerless_mask

        max_swing = np.maximum(np.abs(self._theta_vmc_l), np.abs(self._theta_vmc_r))
        swing_too_high = (max_swing > rc.swing_angle_limit) & ~powerless_mask

        # 课程完成收敛：只剩 tilt 终止；命中计数照常记录，用于 [term] 诊断
        loosen_mask = np.bool_(self._term_loosen_active(rc))
        self._vertex_touch_count = np.where(vertex_touch & ~loosen_mask, self._vertex_touch_count + 1, 0)
        self._tilt_count = np.where(body_tilt_too_high, self._tilt_count + 1, 0)
        self._height_low_count = np.where(height_collapse & ~loosen_mask, self._height_low_count + 1, 0)
        self._swing_count = np.where(swing_too_high & ~loosen_mask, self._swing_count + 1, 0)

        terminated = (
            (self._vertex_touch_count >= rc.n_steps_delay)
            | (self._tilt_count >= rc.n_steps_delay)
            | (self._height_low_count >= rc.n_steps_delay)
            | (self._swing_count >= rc.n_steps_delay)
        ) & ~powerless_mask

        if self._cfg.termination_config.terrain_out_of_bounds:
            terrain_cfg = self._cfg.scene.terrain.generator if self._cfg.scene.terrain is not None else None
            oob = terrain_out_of_bounds(
                self,
                terrain_cfg,
                float(self._cfg.termination_config.terrain_distance_buffer),
            )
            terminated = terminated | (oob & ~powerless_mask)

        n_envs = self._num_envs
        self._episode_steps += 1
        self._total_steps += n_envs
        self._term_window_steps += n_envs
        self._hit_vertex += int(np.sum(vertex_touch))
        self._hit_tilt += int(np.sum(body_tilt_too_high))
        self._hit_height_low += int(np.sum(height_collapse))
        self._hit_swing += int(np.sum(swing_too_high))

        if np.any(terminated):
            self._term_vertex += int(np.sum((self._vertex_touch_count >= rc.n_steps_delay) & terminated))
            self._term_tilt += int(np.sum((self._tilt_count >= rc.n_steps_delay) & terminated))
            self._term_height_low += int(np.sum((self._height_low_count >= rc.n_steps_delay) & terminated))
            self._term_swing += int(np.sum((self._swing_count >= rc.n_steps_delay) & terminated))

        if self._term_window_steps >= 200 * n_envs:
            self._term_vertex = self._term_tilt = self._term_height_low = self._term_swing = 0
            self._term_window_steps = 0
            self._hit_vertex = self._hit_tilt = self._hit_height_low = self._hit_swing = 0

        return tilt, base_body_contact_force, terminated

    def _compute_kinematics(self, dof_vel: np.ndarray, rw_theta: np.ndarray, lw_theta: np.ndarray, _dt: float) -> dict[str, np.ndarray]:
        torques = self._last_motor_ctrl.copy()
        qacc = (dof_vel - self._last_dof_vel_for_acc) / _dt
        self._last_dof_vel_for_acc[:] = dof_vel

        # 改后
        swing_vel_sq = self._vmc_phi0_dot[:, 0] ** 2 + self._vmc_phi0_dot[:, 1] ** 2

        if hasattr(self, "_swing_first_step"):
            swing_vel_sq = np.where(self._swing_first_step, 0.0, swing_vel_sq)
            self._swing_first_step = np.zeros_like(self._swing_first_step, dtype=bool)

        self._last_rw_body = rw_theta.copy()
        self._last_lw_body = lw_theta.copy()

        self._cached_kin_torques = torques
        self._cached_kin_qacc = qacc

        return {
            "torques": torques,
            "qacc": qacc,
            "swing_vel_sq": swing_vel_sq,
            "wheel_vel_avg": np.mean(dof_vel[:, WHEEL_JOINT_INDICES], axis=1),
            "lw_vel": dof_vel[:, WHEEL_JOINT_INDICES[0]],
            "rw_vel": dof_vel[:, WHEEL_JOINT_INDICES[1]],
        }

    def _compute_vmc_contact_world_z(self, base_z: np.ndarray, gravity_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """计算左右腿 VMC C 点的世界 z 坐标。

        C 点 (body frame) = [L0·cos(φ₀), ±hip_y, hip_z - L0·sin(φ₀)]
        C 点世界 z = base_z + R[2,:] · C_body = base_z - gravity_body · C_body
        """
        hip_y = np.array([0.187, -0.187], dtype=self._np_dtype)
        hip_z = 0.141

        c_world_z = []
        for leg in range(2):
            L0 = self._vmc_L0[:, leg]
            phi0 = self._vmc_phi0[:, leg]
            c_body = np.stack([
                L0 * np.cos(phi0),
                np.full_like(L0, hip_y[leg]),
                hip_z - L0 * np.sin(phi0),
            ], axis=1)
            c_world_z.append(base_z + np.sum(gravity_body * c_body, axis=1))

        return c_world_z[0], c_world_z[1]

    def _populate_info(self, info: dict, sensors: dict, tilt: np.ndarray, base_body_contact_force: np.ndarray,
                       terminated: np.ndarray, rw_theta: np.ndarray, lw_theta: np.ndarray,
                       kin: dict) -> None:
        swing_target_theta = np.full_like(rw_theta, np.deg2rad(0.0))
        leg_r = self._vmc_L0[:, 1].reshape(-1)
        leg_l = self._vmc_L0[:, 0].reshape(-1)
        c_left_z, c_right_z = self._compute_vmc_contact_world_z(sensors["base_pos"][:, 2], sensors["gravity_body"])
        wheel_world_z = np.minimum(c_left_z, c_right_z)
        g_body = sensors["gravity_body"]
        base_h_scan = combined_base_height_from_scan(self)
        base_height = base_h_scan + 0.141 * sensors["upvector"][:, 2]
        body_corner_min_z = base_h_scan - np.abs(g_body[:, 0]) * 0.28 - np.abs(g_body[:, 1]) * 0.14
        theta_err = np.maximum(np.abs(rw_theta - swing_target_theta), np.abs(lw_theta - swing_target_theta))
        info.update({
            "tilt": tilt,
            "base_z": sensors["base_pos"][:, 2],
            "base_height_above_ground": base_height,
            "body_corner_min_z": body_corner_min_z,
            "theta_err": theta_err,
            "torques": kin["torques"],
            "qacc": kin["qacc"],
            "swing_rw_theta": rw_theta,
            "swing_lw_theta": lw_theta,
            "swing_vel_sq": kin["swing_vel_sq"],
            "wheel_vel_avg": kin["wheel_vel_avg"],
            "rw_vel": kin["rw_vel"],
            "lw_vel": kin["lw_vel"],
            "swing_target_theta": swing_target_theta,
            "swing_err_rw": rw_theta - swing_target_theta,
            "swing_err_lw": lw_theta - swing_target_theta,
            "terminated": terminated,
            "base_body_contact_force": base_body_contact_force,
            "left_front_leg_contact_force": sensors["left_front_leg_contact_force"],
            "left_front_child1_leg_contact_force": sensors["left_front_child1_leg_contact_force"],
            "left_front_leg1_contact_force": sensors["left_front_leg1_contact_force"],
            "left_rear_child1_leg2_contact_force": sensors["left_rear_child1_leg2_contact_force"],
            "left_rear_leg_contact_force": sensors["left_rear_leg_contact_force"],
            "left_rear_leg2_contact_force": sensors["left_rear_leg2_contact_force"],
            "left_rear_child1_leg1_contact_force": sensors["left_rear_child1_leg1_contact_force"],
            "right_front_leg_contact_force": sensors["right_front_leg_contact_force"],
            "right_front_child1_leg_contact_force": sensors["right_front_child1_leg_contact_force"],
            "right_front_leg1_contact_force": sensors["right_front_leg1_contact_force"],
            "right_rear_child1_leg2_contact_force": sensors["right_rear_child1_leg2_contact_force"],
            "right_rear_leg_contact_force": sensors["right_rear_leg_contact_force"],
            "right_rear_leg2_contact_force": sensors["right_rear_leg2_contact_force"],
            "right_rear_child1_leg1_contact_force": sensors["right_rear_child1_leg1_contact_force"],
            "collision_stacked": np.column_stack([
                sensors["base_body_contact_force"].reshape(-1),
                sensors["left_front_leg_contact_force"].reshape(-1),
                sensors["left_front_child1_leg_contact_force"].reshape(-1),
                sensors["left_front_leg1_contact_force"].reshape(-1),
                sensors["left_rear_child1_leg2_contact_force"].reshape(-1),
                sensors["left_rear_leg_contact_force"].reshape(-1),
                sensors["left_rear_leg2_contact_force"].reshape(-1),
                sensors["left_rear_child1_leg1_contact_force"].reshape(-1),
                sensors["right_front_leg_contact_force"].reshape(-1),
                sensors["right_front_child1_leg_contact_force"].reshape(-1),
                sensors["right_front_leg1_contact_force"].reshape(-1),
                sensors["right_rear_child1_leg2_contact_force"].reshape(-1),
                sensors["right_rear_leg_contact_force"].reshape(-1),
                sensors["right_rear_leg2_contact_force"].reshape(-1),
                sensors["right_rear_child1_leg1_contact_force"].reshape(-1),
            ]).astype(self._np_dtype),
            "prev_body_vx": self._prev_body_vx,
            "body_vx": sensors["local_linvel"][:, 0],
            "body_vy": sensors["local_linvel"][:, 1],
            "upvector": sensors["upvector"],
            "gravity_body": sensors["gravity_body"],
            "leg_r": leg_r,
            "leg_l": leg_l,
            "_prev_leg_len": self._prev_leg_len.copy(),
            "wheel_world_z": wheel_world_z,
            "spin_mask": self._spin_mask,
            "contact_filt": self._contact_filt,
            "_action_t2": self._action_t2,
            "_prev2_actions": self._prev2_actions,
            "_action_t3": self._action_t3,
            "_base_mass": self._cached_base_mass,
            "_base_com": self._cached_base_com,
            "_ground_friction": self._cached_ground_friction,
            "_vmc_ff": self._vmc_ff,
            "_vmc_L0": self._vmc_L0,
            "_vmc_L0_dot": self._vmc_L0_dot,
            "_vmc_theta_dot": self._vmc_theta_dot,
        })

    def _accumulate_curriculum_stats(self, info: dict, local_linvel: np.ndarray, commands: np.ndarray, gyro: np.ndarray) -> None:
        self._episode_vel_sum += np.abs(local_linvel[:, 0])
        self._episode_cmd_vel_sum += np.abs(commands[:, 0])

        fwd = commands[:, 0] > 0.05
        bwd = commands[:, 0] < -0.05
        self._episode_vel_sum_fwd += np.where(fwd & (local_linvel[:, 0] > 0), np.abs(local_linvel[:, 0]), 0.0)
        self._episode_cmd_vel_sum_fwd += np.where(fwd, np.abs(commands[:, 0]), 0.0)
        self._episode_vel_sum_bwd += np.where(bwd & (local_linvel[:, 0] < 0), np.abs(local_linvel[:, 0]), 0.0)
        self._episode_cmd_vel_sum_bwd += np.where(bwd, np.abs(commands[:, 0]), 0.0)

        tilt = np.asarray(info.get("tilt", np.zeros(self._num_envs)), dtype=self._np_dtype)
        err_rw = np.abs(np.asarray(info.get("swing_err_rw", np.zeros(self._num_envs)), dtype=self._np_dtype))
        err_lw = np.abs(np.asarray(info.get("swing_err_lw", np.zeros(self._num_envs)), dtype=self._np_dtype))
        self._episode_tilt_sum += tilt
        self._episode_swing_err_sum += np.maximum(err_rw, err_lw)

        rc = self._cfg.reward_config
        s = rc.scales
        sigma_lin = self._curriculum.tracking_sigma_lin
        gravity_body = np.asarray(
            info.get("gravity_body", np.zeros((self._num_envs, 3))), dtype=self._np_dtype
        )
        cos_pitch = np.cos(np.arcsin(np.clip(gravity_body[:, 0], -1.0, 1.0)))
        vx_horiz = local_linvel[:, 0] * cos_pitch
        lin_vel_error = np.square(commands[:, 0] - vx_horiz)
        # 课程用静态 σ_base；不再对 |cmd|≤1 额外减半，与日志 tracking_lin_vel 同口径。
        sigma = sigma_lin

        tracking_total = s["tracking_lin_vel"] * np.exp(-lin_vel_error / sigma)
        max_tracking = max(float(s["tracking_lin_vel"]), 1e-6)
        self._episode_tracking_reward_sum += tracking_total / max_tracking
        self._curriculum.record_step_tracking(tracking_total / max_tracking)

        roll_aware_height = np.asarray(info.get("base_height_above_ground", np.zeros(self._num_envs)), dtype=self._np_dtype)
        target_height = np.asarray(info["commands"][:, 3], dtype=self._np_dtype)
        self._episode_height_err_sum += np.abs(roll_aware_height - target_height)
    
    #打印调试信息，每200步打印一次
    def _debug_print(
        self, info, reward, commands, sensors, tilt, base_body_contact_force,
    ):
        steps = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        if int(steps[0]) % 200 != 0:
            return

        gyro = sensors["gyro"]
        upvector = sensors["upvector"]
        gravity_body = sensors["gravity_body"]
        dof_vel = sensors["dof_vel"]
        base_pos = sensors["base_pos"]
        world_linvel = sensors["world_linvel"]
        local_linvel = sensors["local_linvel"]
        torques = info.get("torques", np.zeros_like(dof_vel))

        tilt_raw = float(np.mean(np.sqrt(np.sum(np.square(upvector[:, :2]), axis=1))))
        base_z_mean = float(np.mean(base_pos[:, 2]))
        base_h_mean = float(np.mean(info.get("base_height_above_ground", base_pos[:, 2])))
        cos_pitch_raw = float(np.mean(np.cos(np.arcsin(np.clip(gravity_body[:, 0], -1.0, 1.0)))))
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

        all_vmc_l = float(np.rad2deg(np.mean(getattr(self, '_theta_vmc_l', np.zeros(self._num_envs)))))
        all_vmc_r = float(np.rad2deg(np.mean(getattr(self, '_theta_vmc_r', np.zeros(self._num_envs)))))

        fwd_mask = commands[:, 0] > 0.02
        bwd_mask = commands[:, 0] < -0.02
        left_mask = commands[:, 2] > 0.05
        right_mask = commands[:, 2] < -0.05

        swing_target = info["swing_target_theta"]
        swing_err_rw = info["swing_err_rw"]
        swing_err_lw = info["swing_err_lw"]

        all_cmd_vx = float(np.mean(commands[:, 0]))
        all_vx = float(np.mean(local_linvel[:, 0]))
        all_wheel_avg = float(np.mean(info["wheel_vel_avg"])) * 0.06
        all_rw_v = float(np.mean(info["rw_vel"])) * 0.06
        all_lw_v = float(np.mean(info["lw_vel"])) * 0.06
        all_cmd_ang = float(np.mean(commands[:, 2]))
        all_w_signed = float(np.mean(gyro[:, 2]))
        L = 0.380
        all_tg_rw = all_cmd_vx + 0.5 * all_cmd_ang * L
        all_tg_lw = all_cmd_vx - 0.5 * all_cmd_ang * L
        all_theta_tg = float(np.rad2deg(np.mean(swing_target)))
        all_rw_err = float(np.rad2deg(np.mean(np.abs(swing_err_rw))))
        all_lw_err = float(np.rad2deg(np.mean(np.abs(swing_err_lw))))
        all_rw_theta = float(np.rad2deg(np.mean(info["swing_rw_theta"])))
        all_lw_theta = float(np.rad2deg(np.mean(info["swing_lw_theta"])))

        if left_mask.sum() > 0:
            left_cmd_ang = float(np.mean(commands[left_mask, 2]))
            left_w = float(np.mean(gyro[left_mask, 2]))
        else:
            left_cmd_ang = 0.0
            left_w = 0.0

        if right_mask.sum() > 0:
            right_cmd_ang = float(np.mean(commands[right_mask, 2]))
            right_w = float(np.mean(gyro[right_mask, 2]))
        else:
            right_cmd_ang = 0.0
            right_w = 0.0

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
            fwd_vmc_l = float(np.rad2deg(np.mean(self._theta_vmc_l[fwd_mask])))
            fwd_vmc_r = float(np.rad2deg(np.mean(self._theta_vmc_r[fwd_mask])))
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
            fwd_vmc_l = all_vmc_l
            fwd_vmc_r = all_vmc_r

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
            bwd_vmc_l = float(np.rad2deg(np.mean(self._theta_vmc_l[bwd_mask])))
            bwd_vmc_r = float(np.rad2deg(np.mean(self._theta_vmc_r[bwd_mask])))
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
            bwd_vmc_l = all_vmc_l
            bwd_vmc_r = all_vmc_r

        print(f"[state] total_reward=" f"{float(np.mean(reward)):+.4f}")
        ff_mean = float(np.mean(self._vmc_ff / (self._base_vmc_ff + 1e-8)))
        as_mean = float(np.mean(self._action_smoothing))
        friction0 = float(np.mean(self._cached_ground_friction[:, 0]))
        friction1 = float(np.mean(self._cached_ground_friction[:, 1]))
        pitch_all = np.rad2deg(np.arcsin(np.clip( gravity_body[:, 0], -1.0, 1.0)))
        roll_all  = np.rad2deg(np.arcsin(np.clip(-gravity_body[:, 1], -1.0, 1.0)))
        pitch_mean = float(np.mean(pitch_all))
        roll_mean  = float(np.mean(roll_all))
        pitch_abs  = float(np.mean(np.abs(pitch_all)))
        roll_abs   = float(np.mean(np.abs(roll_all)))
        up_x = float(np.mean(upvector[:, 0]))
        up_y = float(np.mean(upvector[:, 1]))
        up_z = float(np.mean(upvector[:, 2]))
        h_tg_mean = float(np.mean(commands[:, 3]))
        h_mean = float(np.mean(info.get("base_height_above_ground", np.zeros_like(base_z_mean))))
        vmc_l0_l = float(np.mean(self._vmc_L0[:, 0]))
        vmc_l0_r = float(np.mean(self._vmc_L0[:, 1]))
        leg_tgt = h_tg_mean
        roll_rad = np.deg2rad(roll_mean)
        tgt_l = leg_tgt + 0.21 * np.tan(roll_rad)
        tgt_r = leg_tgt - 0.21 * np.tan(roll_rad)
        pol_lr = float(np.mean(self._vmc_targets[:, 0, 1]))
        pol_rr = float(np.mean(self._vmc_targets[:, 1, 1]))
        yaw_err = info.get("yaw_error", np.zeros(self._num_envs, dtype=self._np_dtype))
        yaw_err_deg = float(np.rad2deg(np.mean(np.abs(yaw_err))))

        vmc_phi0_l = float(np.rad2deg(np.mean(self._vmc_phi0[:, 0])))
        vmc_phi0_r = float(np.rad2deg(np.mean(self._vmc_phi0[:, 1])))
        pol_lf = float(np.mean(self._vmc_targets[:, 0, 0]))
        pol_rf = float(np.mean(self._vmc_targets[:, 1, 0]))
        vmc_l_x = float(np.mean(self._vmc_L0[:, 0] * np.cos(self._vmc_phi0[:, 0])))
        vmc_l_z = float(np.mean(self._vmc_L0[:, 0] * np.sin(self._vmc_phi0[:, 0])))
        vmc_r_x = float(np.mean(self._vmc_L0[:, 1] * np.cos(self._vmc_phi0[:, 1])))
        vmc_r_z = float(np.mean(self._vmc_L0[:, 1] * np.sin(self._vmc_phi0[:, 1])))

        promote_n, demote_n = self._curriculum.pop_cmd_curriculum_stats()
        print(f"  [cmd_curriculum] promote={promote_n} demote={demote_n} (since last debug)")
        print(f"  [DR]  ff×{ff_mean:.2f}  α={as_mean:.2f}  τ_scale={float(np.mean(self._torque_scale)):.2f}  noise={float(np.mean(self._actuator_noise_std)):.3f}  fric=({friction0:.2f},{friction1:.2f})")
        print(f"  [pose] tilt={np.rad2deg(tilt.mean()):+.1f}°  pitch={pitch_mean:+.1f}°(|{pitch_abs:.1f}°)  roll={roll_mean:+.1f}°(|{roll_abs:.1f}°)  upvec=({up_x:+.2f},{up_y:+.2f},{up_z:+.2f})  cos={cos_pitch_raw:.3f}  corner_z={float(np.mean(info.get('body_corner_min_z', np.zeros(1)))):.3f}")

        left_contact_force = sensors.get("left_wheel_contact_force", np.zeros((self._num_envs, 1)))
        right_contact_force = sensors.get("right_wheel_contact_force", np.zeros((self._num_envs, 1)))
        wheel_contact_force = np.maximum(left_contact_force, right_contact_force)
        print(f"  [contact] lw={float(np.mean(left_contact_force)):.3f}  rw={float(np.mean(right_contact_force)):.3f}  "
              f"max={float(np.max(wheel_contact_force)):.3f}  "
              f"airborne={int(np.sum(wheel_contact_force < 1e-6))}/{self._num_envs}  "
              f"wheel_envs={int(np.sum(wheel_contact_force > 1e-6))}")
        body_contact_parts = {
            "base": sensors.get("base_body_contact_force"),
            "lf_leg": sensors.get("left_front_leg_contact_force"),
            "lf_c1": sensors.get("left_front_child1_leg_contact_force"),
            "lf_l1": sensors.get("left_front_leg1_contact_force"),
            "lr_c2": sensors.get("left_rear_child1_leg2_contact_force"),
            "lr_leg": sensors.get("left_rear_leg_contact_force"),
            "lr_leg2": sensors.get("left_rear_leg2_contact_force"),
            "lr_c1": sensors.get("left_rear_child1_leg1_contact_force"),
            "rf_leg": sensors.get("right_front_leg_contact_force"),
            "rf_c1": sensors.get("right_front_child1_leg_contact_force"),
            "rf_l1": sensors.get("right_front_leg1_contact_force"),
            "rr_c2": sensors.get("right_rear_child1_leg2_contact_force"),
            "rr_leg": sensors.get("right_rear_leg_contact_force"),
            "rr_leg2": sensors.get("right_rear_leg2_contact_force"),
            "rr_c1": sensors.get("right_rear_child1_leg1_contact_force"),
        }
        nonzero_parts = []
        body_envs = 0
        for name, val in body_contact_parts.items():
            if val is not None:
                m = float(np.mean(val))
                if m > 1e-6:
                    nonzero_parts.append(f"{name}={m:.1f}")
                    body_envs = max(body_envs, int(np.sum(np.asarray(val).flatten() > 1e-6)))
        if nonzero_parts:
            print(f"  [contact] body: {'  '.join(nonzero_parts)}  envs={body_envs}")

        print(f"  [h]    base_h={base_h_mean:.3f}(abs:{base_z_mean:.3f})  h={h_mean:.3f}/tg={h_tg_mean:.3f}  |  L0=({vmc_l0_l:.3f},{vmc_l0_r:.3f})  roll_tgt=({tgt_l:.3f},{tgt_r:.3f})")
        print(f"  [joint] jpos=({float(np.rad2deg(np.mean(self._debug_joint_pos[:, 0]))):+.1f},{float(np.rad2deg(np.mean(self._debug_joint_pos[:, 1]))):+.1f},{float(np.rad2deg(np.mean(self._debug_joint_pos[:, 2]))):+.1f},{float(np.rad2deg(np.mean(self._debug_joint_pos[:, 3]))):+.1f})°  tgt=({np.rad2deg(pol_lf):+.1f},{np.rad2deg(pol_rf):+.1f},{np.rad2deg(pol_lr):+.1f},{np.rad2deg(pol_rr):+.1f})°")

        policy_lw_raw = float(np.mean(info["current_actions"][:, 4]))
        policy_rw_raw = float(np.mean(info["current_actions"][:, 5]))
        print(f"  [vel]  vx={vx_mean:+.3f}(tg:{vx_cmd_mean:+.3f})  |vx|={vx_abs:.3f}(tg:{vx_cmd_abs:.3f})  vz={vz_raw:+.3f}  w={angv_raw:+.3f}(tg:{angv_cmd_abs:.3f})  yaw_err={yaw_err_deg:+.1f}°")
        if hasattr(self, '_spawn'):
            try:
                terrain_types = np.asarray(
                    self._spawn.get_terrain_type_for(np.arange(self._num_envs)),
                    dtype=object,
                )
                levels = np.asarray(getattr(self._spawn, "levels", np.zeros(self._num_envs)))
                pos_cmd = commands[:, 0] > 0.02
                neg_cmd = commands[:, 0] < -0.02
                parts = []

                def _dir_stats(sel: np.ndarray) -> str:
                    count = int(np.count_nonzero(sel))
                    if count == 0:
                        return "[0] -"
                    actual = float(np.mean(local_linvel[sel, 0]))
                    target = float(np.mean(commands[sel, 0]))
                    return (
                        f"[{count}] vx={actual:+.2f}/tg={target:+.2f}/"
                        f"e={actual - target:+.2f}"
                    )

                for terrain_type in sorted(set(terrain_types.tolist())):
                    mask = terrain_types == terrain_type
                    n = int(np.count_nonzero(mask))
                    if n == 0:
                        continue
                    pos = mask & pos_cmd
                    neg = mask & neg_cmd
                    zero = n - int(np.count_nonzero(pos)) - int(np.count_nonzero(neg))
                    lim_hi = float(np.mean(self._eff_vx_bounds[mask, 1]))
                    lim_lo = float(np.mean(self._eff_vx_bounds[mask, 0]))
                    lim_txt = f"lim=[{lim_lo:+.2f},{lim_hi:+.2f}]"
                    parts.append(
                        f"{terrain_type}(n={n},lvl={float(np.mean(levels[mask])):.1f},{lim_txt}) "
                        f"+{_dir_stats(pos)}  -{_dir_stats(neg)}  z={zero}"
                    )
                if parts:
                    print("  [terrain_vel] +fwd  -bwd  z=|cmd_vx|<=0.02")
                    for part in parts:
                        print(f"    {part}")
            except Exception as exc:
                print(f"  [terrain_vel] unavailable: {exc}")
        print(f"  [wheel] avg={wheel_v_raw * 0.06:+.3f}m/s  lw={lw_vel_raw * 0.06:+.3f}  rw={rw_vel_raw * 0.06:+.3f}  |  pol_lw={policy_lw_raw:+.1f}(v={policy_lw_raw * 0.06:+.2f})  pol_rw={policy_rw_raw:+.1f}(v={policy_rw_raw * 0.06:+.2f})  |  max={max(float(np.max(info['rw_vel'])), float(np.max(info['lw_vel']))) * 0.06:+.2f}  min={min(float(np.min(info['rw_vel'])), float(np.min(info['lw_vel']))) * 0.06:+.2f}")
        print(f"  [vmc]   θ_l={all_vmc_l:+.1f}°  θ_r={all_vmc_r:+.1f}°  φ₀=({vmc_phi0_l:.1f},{vmc_phi0_r:.1f})°  C=({vmc_l_x:+.3f},{vmc_l_z:+.3f})({vmc_r_x:+.3f},{vmc_r_z:+.3f})  θ̇_vel={swing_vel_raw:+.1f}°/s  step={int(self._episode_steps.mean())}")

        if fwd_mask.sum() > 0:
            fwd_policy_lw = float(np.mean(info['current_actions'][fwd_mask, 4]))
            fwd_policy_rw = float(np.mean(info['current_actions'][fwd_mask, 5]))
        else:
            fwd_policy_lw = policy_lw_raw
            fwd_policy_rw = policy_rw_raw
        if bwd_mask.sum() > 0:
            bwd_policy_lw = float(np.mean(info['current_actions'][bwd_mask, 4]))
            bwd_policy_rw = float(np.mean(info['current_actions'][bwd_mask, 5]))
        else:
            bwd_policy_lw = policy_lw_raw
            bwd_policy_rw = policy_rw_raw

        print(f"  [fwd]  cmd_vx={fwd_cmd_vx:+.2f}  vx={fwd_vx:+.2f}  wheel={fwd_wheel_avg:+.2f}(lw:{fwd_lw_v:+.2f}/rw:{fwd_rw_v:+.2f})  pol=({fwd_policy_lw:+.1f},{fwd_policy_rw:+.1f})  θ_l={fwd_vmc_l:+.1f}°  θ_r={fwd_vmc_r:+.1f}°")
        print(f"  [bwd]  cmd_vx={bwd_cmd_vx:+.2f}  vx={bwd_vx:+.2f}  wheel={bwd_wheel_avg:+.2f}(lw:{bwd_lw_v:+.2f}/rw:{bwd_rw_v:+.2f})  pol=({bwd_policy_lw:+.1f},{bwd_policy_rw:+.1f})  θ_l={bwd_vmc_l:+.1f}°  θ_r={bwd_vmc_r:+.1f}°")

        sigma_mean = float(getattr(self._cfg.reward_config, "base_height_sigma", 0.05))
        height_err_mean = float(np.mean(np.abs(info.get("base_height_above_ground", np.zeros_like(commands[:, 3])) - commands[:, 3])))
        height_err_max = float(np.max(np.abs(info.get("base_height_above_ground", np.zeros_like(commands[:, 3])) - commands[:, 3])))
        print(f"  [h_rwd] sigma={sigma_mean:.4f}  err_mean={height_err_mean:.4f}  err_max={height_err_max:.4f}")

        wheel_z = info.get("wheel_world_z")
        if wheel_z is not None:
            wheel_z_mean = float(np.mean(wheel_z))
            wheel_z_min = float(np.min(wheel_z))
            lvl3 = int(np.sum(wheel_z > 0.35))
            lvl2 = int(np.sum((wheel_z > 0.18) & (wheel_z <= 0.35)))
            lvl1 = int(np.sum((wheel_z > 0.10) & (wheel_z <= 0.18)))
            lvl0 = int(np.sum(wheel_z <= 0.10))
            print(f"  [wheel_z] mean={wheel_z_mean:.3f} min={wheel_z_min:.3f}  L3={lvl3} L2={lvl2} L1={lvl1} L0={lvl0}")

        vmc_F_r = float(np.mean(self._vmc_F[:, 1]))
        vmc_F_l = float(np.mean(self._vmc_F[:, 0]))
        vmc_T_r = float(np.mean(self._vmc_T[:, 1]))
        vmc_T_l = float(np.mean(self._vmc_T[:, 0]))
        print(f"  [torque] rms={float(np.sqrt(np.mean(np.sum(np.square(torques), axis=1)))):+.2f}  rw={float(np.mean(np.abs(torques[:, 5]))):+.2f}  lw={float(np.mean(np.abs(torques[:, 4]))):+.2f}  R_leg={float(np.mean(np.abs(torques[:, 1]))):+.2f}/{float(np.mean(np.abs(torques[:, 3]))):+.2f}  L_leg={float(np.mean(np.abs(torques[:, 0]))):+.2f}/{float(np.mean(np.abs(torques[:, 2]))):+.2f}")
        print(f"  [VMC]   F=({vmc_F_l:+.1f},{vmc_F_r:+.1f})N  T=({vmc_T_l:+.1f},{vmc_T_r:+.1f})N·m")
        print(f"  [term]  mode={'tilt_only' if self._term_loosened else 'full'}  rate={((self._term_vertex + self._term_tilt + self._term_height_low + self._term_swing) / max(self._term_window_steps, 1) * 1000):.1f}/1k  touch={self._hit_vertex / max(self._term_window_steps, 1) * 1000:.0f}  tilt={self._hit_tilt / max(self._term_window_steps, 1) * 1000:.0f}  h_low={self._hit_height_low / max(self._term_window_steps, 1) * 1000:.0f}  swing={self._hit_swing / max(self._term_window_steps, 1) * 1000:.0f}  body_contact_force={float(np.mean(base_body_contact_force)):.2f}({float(np.max(base_body_contact_force)):.2f})  base_h={base_h_mean:.3f}")
        print(f"  [info]  step={int(self._episode_steps.mean())}  total={self._total_steps}  cur={self._curriculum.scale:.2f}"
              f"  vx_range=[{float(np.mean(self._eff_vx_bounds[:, 0])):+.2f},{float(np.mean(self._eff_vx_bounds[:, 1])):+.2f}]"
              f"  vx_cfg=[{float(self._curriculum.cmd_vel_range[0, 0]):+.2f},{float(self._curriculum.cmd_vel_range[0, 1]):+.2f}]"
              f"  vx_cmd=[{float(np.min(commands[:, 0])):+.2f},{float(np.max(commands[:, 0])):+.2f}]"
              f"  yaw_range=[{float(self._curriculum.cmd_yaw_range[0]):+.2f},{float(self._curriculum.cmd_yaw_range[1]):+.2f}]  σ_ang={self._curriculum.tracking_sigma_ang:.2f}  σ_yaw={self._curriculum.tracking_sigma_yaw:.2f}"
              f"  (vx_range=实际生效/地形夹紧后  vx_cfg=课程名义  vx_cmd=本步样本)"
              )
        if hasattr(self, '_update_state_timings'):
            t = self._update_state_timings
            b = info.get('timing', {})
            print(f"  [timing] total={t['total']:.1f}ms  sensor={t['sensor_read']:.1f}  pre_step={t['pre_step_total']:.1f}  term={t['detect_term']:.1f}  kin={t['kinematics']:.1f}  info={t['populate_info']:.1f}  reward={t['compute_reward']:.1f}  obs={t['compute_obs']:.1f}  print={t['debug_print']:.1f}")
            # ── solver 诊断：ncon / nefc / niter 采样 ──
            _nc_mn, _nc_mx, _ne_mn, _ne_mx, _ni_mn, _ni_mx = -1.0, -1, -1.0, -1, -1.0, -1
            _st_mn, _st_mx = -1.0, -1.0
            try:
                from unilab.base.backend.mujoco.backend import _sample_solver_stats
                _stats = _sample_solver_stats(self._backend._model, self._backend._physics_state,
                                              self._backend._num_envs, n_sample=32)
                if _stats:
                    (_nc_mn, _nc_mx, _ne_mn, _ne_mx, _ni_mn, _ni_mx,
                     _st_mn, _st_mx) = _stats
                    import sys, time as _time
                    _ptime = _time.process_time()
                    sys.stderr.write(
                        f"[solver_diag] proc_cpu_s={_ptime:.1f}  "
                        f"ncon max={_nc_mx}/mean={_nc_mn:.1f}  "
                        f"nefc max={_ne_mx}/mean={_ne_mn:.1f}  "
                        f"niter max={_ni_mx}/mean={_ni_mn:.1f}  "
                        f"mj_step_us max={_st_mx:.0f}/mean={_st_mn:.0f}\n")
            except Exception:
                pass
            print(f"  [step]   backend: physics={b.get('backend_physics_ms',0):.1f}ms (pool={b.get('backend_pool_step_ms',0):.1f}+copy={b.get('backend_state_copy_ms',0):.1f})  "
                  f"cpu/w={b.get('backend_pool_cpu_ratio',0):.2f}  "
                  f"ss={b.get('backend_pool_substeps','-')}  "
                  f"ncon={_nc_mx}/{_nc_mn:.0f}  nefc={_ne_mx}/{_ne_mn:.0f}  niter={_ni_mx}/{_ni_mn:.0f}  "
                  f"mj_step_us={_st_mx:.0f}/{_st_mn:.0f}  "
                  f"set_ctrl={b.get('backend_set_ctrl_ms',0):.1f}  refresh={b.get('backend_refresh_cache_ms',0):.1f}  "
                  f"apply_action={b.get('apply_action_ms',0):.1f}  reset_done={b.get('reset_done_ms',0):.1f}  env_step_total={b.get('env_step_total_ms',0):.1f}ms")
            # ── 碰撞诊断：统计激活的接触段数 ──
            _leg_sensors = [k for k in sensors if k.endswith("_contact_force") and "leg" in k and "wheel" not in k and "base" not in k]
            leg_contact_norms = np.column_stack([sensors[k].ravel() for k in _leg_sensors])  # (E, N_legs)
            leg_active = (leg_contact_norms > 5.0).sum(axis=1)
            body_contact_arr = sensors.get("base_body_contact_force", np.zeros((self._num_envs, 1))).ravel()
            body_active = int((body_contact_arr > 100.0).sum())
            wheel_contact_l = sensors.get("left_wheel_contact_force", np.zeros((self._num_envs, 1))).ravel()
            wheel_contact_r = sensors.get("right_wheel_contact_force", np.zeros((self._num_envs, 1))).ravel()
            wheel_active = int(((np.abs(wheel_contact_l) > 5.0) | (np.abs(wheel_contact_r) > 5.0)).sum())
            n_high_contact = int((leg_active >= 6).sum())
            n_any_leg = int((leg_active >= 1).sum())
            print(f"  [collision] body_contact={body_active}/{self._num_envs}  wheel_contact={wheel_active}/{self._num_envs}  leg_contact={n_any_leg}/{self._num_envs}  heavy_leg(>=6)= {n_high_contact}/{self._num_envs}  avg_leg_seg={leg_active.mean():.1f}")
        if hasattr(self, '_spawn'):
            _get_terrain = getattr(self._spawn, "get_terrain_type_for", None)
            _cfg = getattr(self._spawn, "_cfg", None)
            if _get_terrain is not None and _cfg is not None and _cfg.type_curriculum_enabled:
                terrain_types = _get_terrain(np.arange(self._num_envs))
                unique_types = sorted(set(terrain_types))
                print(f"  [terrain] type_curriculum: {', '.join(unique_types)}")