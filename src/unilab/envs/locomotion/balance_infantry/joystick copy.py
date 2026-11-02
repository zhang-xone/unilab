from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import mujoco

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
    height_scan_obs,
    init_height_scan_sensor,
    raw_height_scan_obs,
    terrain_out_of_bounds,
)
from unilab.envs.locomotion.common.terrain_spawn import (
    BaseSpawnManager,
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)
from unilab.terrains import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    plateau,
    pyramid_stairs,
    pyramid_stairs_inv,
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
    WHEEL_ERR_VEL_THRESHOLD,
    WHEEL_ERR_DECAY_STEPS,
    BalanceInfantryBaseCfg,
    BalanceInfantryBaseEnv,
    forward_kinematics_fivebar,
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
            "tracking_lin_vel": 5.0,
            "tracking_ang_vel": 0.5,
            "body_accel_tracking": 1.5,
            # ── 姿态 / 高度（惩罚，量级 10~200） ───────────────────
            "roll": -1.5,
            "pitch": -1.5,
            "base_height": -100.0,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.02,

            # ── Balance 特有：腿 / 对称（惩罚） ───────────────────
            "swing_vertical": -5.0,
            "swing_symmetry": -10.0,
            "swing_vel": -1.0,

            # ── 平滑 / 能耗（小惩罚，量级 1e-5~1e-3） ─────────────
            "action_rate": -0.002,
            "dof_torques_l2": -0.00005,

            # ── 经验性小惩罚 ─────────────────────────────────────

            # ── 站立姿态保持（低速/姿态恶化时启用） ────────────────
            "stand": -30.0,
            # ── 碰撞惩罚（机体接触地面） ────────────────────────
            "collision": -1.0,
            # ── 跳跃奖励 ──────────────────────────────────
            "takeoff_extend": 2.0,         # 蹬地伸腿
            "flight": 1.0,                 # 空中就给
            "encourage_jump": 2.0,         # 向上速度(连续) + 落地结算
            "line_z": 2.0,                 # 纯上升速度
            "base_height_flight": 2.0,     # 空中目标高度
            "leg_tuck": 2.0,               # 空中收腿
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
    wheel_power_limit: float | None = None  # 轮子功率上限(W)，超限按差值扣分
    # ── 分层渐进式姿态终止 ──
    tilt_limit_tight: float = 20.0 * np.pi / 180.0
    swing_angle_limit: float = 45.0 * np.pi / 180.0  # 摆角 θ 过大终止
    min_height_tight: float = 0.22          # 高度终止线（m）
    # ── 课程完成后的终止收敛：完成后只剩 tilt（翻车）终止 ──
    # 届时关闭 vertex_touch / swing / height 三种姿态终止，只保留 tilt > tilt_limit_tight
    # 与地形出界（防止掉出地形后无限下坠）。tilt_limit_tight 本身不变。
    term_loosen_enabled: bool = False
    term_loosen_from_step: int = 0           # 生效的控制步（每 iter = num_steps_per_env 步）；0=立即
    air_spawn_max_height: float = 0.40       # 空中出生最大高度（m），随 curriculum scale 从 0 渐进至此值
    n_steps_delay: int = 50
    powerless_duration: float = 1.0  # reset 后无力时长（秒），期间扭矩=0/奖励=0/不终止
    powerless_curriculum_threshold: float = 0.125  # curriculum progress 超过此值才启用无力期（2400步≈100iter）
    powerless_probability: float = 0.2  # 每个 reset 随机启用无力期的概率
    lift_force_max: float = 00.0  # 无力期竖直拉力最大值（N），课程 scale=0 时全量，scale→1 时归零
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
    # ── 跳跃命令 ──────────────────────────────
    jump_prob: float = 0.05         # 进入跳跃模式的概率（cmd_jump=1）
    jump_crouch_height: float = 0.20   # 跳跃收腿高度（m）
    jump_launch_height: float = 0.42   # 跳跃蹬地高度（m）
    jump_crouch_steps: int = 15        # 收腿阶段步数
    jump_air_retract_steps: int = 15   # 空中收腿阶段步数
    jump_launch_trigger_height: float = 0.40  # 蹬地→收腿的高度触发阈值（h，m）
    jump_launch_max_steps: int = 40       # 蹬地阶段最大步数，超时强制转入空中收腿
    flight_height_target: float = 0.55   # 空中目标离地高度（m）
    jump_curriculum_height_threshold: float = 0.03  # 高度跟踪误差 < 此值时认为收敛，达标后才开启跳跃
    body_corner_clearance_threshold: float = 0.08  # 角点离地 < 此值（m）开始惩罚
    vel_track_ratio_threshold: float = 0.3  # 速度跟踪比例阈值：|vx_actual| / |cmd_vx| < 此值 → 进入终止计数
    vel_track_min_cmd: float = 0.2  # 只有 |cmd_vx| > 此值时才检查速度跟踪终止


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
            "flat": flat(proportion=0.2),
            "random_rough": random_rough(
                proportion=0.2,
                noise_range=(0.005, 0.025),
                noise_step=0.005,
                border_width=0.2,
                downsampled_scale=0.2,
            ),
            "wave_terrain": wave_terrain(
                proportion=0.1,
                amplitude_range=(0.0, 0.15),
                num_waves=4,
                border_width=0.2,
            ),
            "pyramid_stairs": pyramid_stairs(
                proportion=0.1,
                step_height_range=(0.005, 0.1),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "pyramid_stairs_inv": pyramid_stairs_inv(
                proportion=0.1,
                step_height_range=(0.005, 0.1),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.15,
                slope_range=(0.0, 0.35),
                platform_width=2.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.15,
                slope_range=(0.0, 0.35),
                platform_width=2.0,
                border_width=0.2,
            ),
            "plateau": plateau(
                proportion=0.1,
                platform_height_range=(0.05, 0.30),
                platform_width=3.0,
                border_width=0.4,
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

    if getattr(domain_rand, "randomize_ground_friction", False):
        base_geom_friction = env._backend.get_geom_friction()
        terrain_geom_name = env._cfg.scene.terrain.geom_name if env._cfg.scene.terrain is not None else "floor"
        ground_geom_id = env._backend.get_geom_id(terrain_geom_name)
        if base_geom_friction is None:
            raise ValueError("ground friction randomization requires a cached geom-friction table")
        if ground_geom_id is None:
            raise ValueError("ground friction randomization requires a ground geom id")
        geom_friction_template = np.asarray(base_geom_friction, dtype=np.float64)
        low, high = domain_rand.ground_friction_multiplier_range
        geom_friction = np.broadcast_to(
            geom_friction_template, (num_reset, *geom_friction_template.shape)
        ).copy()
        geom_friction[:, ground_geom_id, 0] = geom_friction_template[ground_geom_id, 0] * np.random.uniform(
            low=low, high=high, size=(num_reset,)
        )
        geom_friction[:, ground_geom_id, 1] = geom_friction_template[ground_geom_id, 1] * np.random.uniform(
            low=low, high=high, size=(num_reset,)
        )
        payload.geom_friction = geom_friction

    if getattr(domain_rand, "randomize_inertia", False):
        base_inertia = np.asarray(env._backend._model.body_inertia, dtype=np.float64)
        low, high = domain_rand.inertia_multiplier_range
        multipliers = np.random.uniform(low, high, size=(num_reset, base_inertia.shape[0]))
        body_inertia = np.broadcast_to(base_inertia, (num_reset, *base_inertia.shape)).copy()
        body_inertia *= multipliers[:, :, np.newaxis]
        payload.body_inertia = body_inertia

    if getattr(domain_rand, "randomize_dof_armature", False):
        base_armature = np.asarray(env._backend._model.dof_armature, dtype=np.float64)
        dof_armature = np.broadcast_to(base_armature, (num_reset, base_armature.size)).copy()

        leg_main_joints = ["left_front_joint", "left_rear_joint", "right_front_joint", "right_rear_joint"]
        wheel_joints = ["left_wheel_joint", "right_wheel_joint"]
        passive_joints = [
            "left_front_child1_joint", "left_front_child2_joint", "left_front_child3_joint",
            "left_rear_child1_joint",
            "right_front_child1_joint", "right_front_child2_joint", "right_front_child3_joint",
            "right_rear_child1_joint",
        ]

        for joint_names, multiplier_range in [
            (leg_main_joints, getattr(domain_rand, "leg_main_armature_multiplier_range", [0.8, 1.2])),
            (wheel_joints, getattr(domain_rand, "wheel_armature_multiplier_range", [0.8, 1.2])),
            (passive_joints, getattr(domain_rand, "passive_armature_multiplier_range", [0.8, 1.2])),
        ]:
            low, high = multiplier_range
            dof_ids = env._backend.get_joint_dof_indices(joint_names)
            dof_armature[:, dof_ids] *= np.random.uniform(low, high, size=(num_reset, len(dof_ids)))

        payload.dof_armature = dof_armature

    return None if payload.is_empty() else payload


# =====================================================================
# 地形相关 cmdvx 工具函数
# =====================================================================


def get_terrain_cmdvx_limits(terrain_types: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """根据地形类型返回 cmdvx 的上下限。

    地形 → cmdvx 范围:
      - 金字塔 (pyramid_stairs / pyramid_stairs_inv): [-1.5, 1.5]
      - 平地/坡/波浪 (flat / hf_pyramid_slope / hf_pyramid_slope_inv / wave_terrain): [-3.0, 3.0]
      - 粗糙 (random_rough): [-2.0, 2.0]
      - 高台 (plateau): 使用默认范围，高度目标单独处理
      - 其他: 默认 [-3.0, 3.0]
    """
    n = len(terrain_types)
    dtype = get_global_dtype()
    vx_low = np.full(n, -3.0, dtype=dtype)
    vx_high = np.full(n, 3.0, dtype=dtype)

    for i, t in enumerate(terrain_types):
        if t in ("pyramid_stairs", "pyramid_stairs_inv", "plateau"):
            vx_low[i] = -2.8
            vx_high[i] = 2.8
        elif t in ("flat"):
            vx_low[i] = -3.5
            vx_high[i] = 3.5
        elif t in ("random_rough"):
            vx_low[i] = -2.5
            vx_high[i] = 2.5
        elif t in ("hf_pyramid_slope", "hf_pyramid_slope_inv", "wave_terrain"):
            vx_low[i] = -2.8
            vx_high[i] = 2.8
        # 其他地形保持默认 [-3.0, 3.0]

    return vx_low, vx_high


def get_terrain_height_params(
    terrain_types: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回每个 env 的 height 命令分布参数。

    Returns (use_uniform, uni_low, uni_high, dist_params):
      - use_uniform=True → Uniform(low=uni_low, high=uni_high)
      - use_uniform=False → Beta(α, β) 映射到 [0.16, 0.42]

    地形 → 分布:
      - 台阶/高台: Beta(3, 1) → mode=0.42, ~16% 低于 0.30
      - 其他: Uniform(0.16, 0.42)
    """
    n = len(terrain_types)
    dtype = get_global_dtype()
    use_uniform = np.ones(n, dtype=bool)
    uni_low = np.full(n, 0.16, dtype=dtype)
    uni_high = np.full(n, 0.42, dtype=dtype)
    beta_a = np.zeros(n, dtype=dtype)
    beta_b = np.zeros(n, dtype=dtype)

    for i, t in enumerate(terrain_types):
        if t in ("pyramid_stairs", "pyramid_stairs_inv", "plateau"):
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
            forward_prob[i] = 0.8

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
        model = backend._model

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

        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))  # xy ±0.5m

        # 高台地形：固定在平台正中心，清零 xy 偏移
        _get_terrain = getattr(env._spawn, "get_terrain_type_for", None)
        plateau_mask = np.zeros(num_reset, dtype=bool)
        if _get_terrain is not None:
            plateau_mask = np.array(
                [t == "plateau" for t in _get_terrain(env_ids)],
                dtype=bool,
            )
        qpos[plateau_mask, 0:2] = 0.0
        extra_z = np.asarray(
            np.random.uniform(0.0, env.cfg.reward_config.air_spawn_max_height, size=(num_reset,)) * scale,
            dtype=get_global_dtype(),
        )
        ground_mask = np.random.uniform(size=(num_reset,)) < (1.0 - scale)  # 前期地面，后期空中
        extra_z[ground_mask] = 0
        high_mask = np.random.uniform(size=(num_reset,)) < 0.1  # 10% 高抛
        extra_z[high_mask] += np.random.uniform(0.0, 0.3, size=int(np.sum(high_mask)))
        qpos[:, 2] += extra_z
        qpos[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos[:, 0:3])  # 地形边界裁剪

        # -----------------------------------------------------------
        # 2. 初始朝向随机化（地形相关）
        #    4 垂直方向 (0°, 90°, 180°, 270°) 随机选一个 + 高斯噪声
        # -----------------------------------------------------------
        dirs = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=get_global_dtype())
        chosen = dirs[np.random.randint(0, 4, size=(num_reset,))]
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
        lift_scale = np.full(num_reset, max(0.0, 1.0 - 1.0 * scale), dtype=get_global_dtype())

        dr = getattr(env.cfg, "domain_rand", None)

        # 8.1 VMC / wheel / leg PD 参数随机化（独立采样）
        vmc_ff_mult = np.ones((num_reset, 1), dtype=get_global_dtype())
        wheel_kd_mult = np.ones((num_reset, 1), dtype=get_global_dtype())
        leg_kp_mult = np.ones((num_reset, 1), dtype=get_global_dtype())
        leg_kd_mult = np.ones((num_reset, 1), dtype=get_global_dtype())
        if dr is not None:
            if getattr(dr, "randomize_vmc_ff", False):
                low, high = dr.vmc_ff_multiplier_range
                vmc_ff_mult = np.random.uniform(low, high, size=(num_reset, 1)).astype(get_global_dtype())
            if getattr(dr, "randomize_wheel_kd", False):
                low, high = dr.wheel_kd_multiplier_range
                wheel_kd_mult = np.random.uniform(low, high, size=(num_reset, 1)).astype(get_global_dtype())
            if getattr(dr, "randomize_kp", False):
                low, high = dr.kp_multiplier_range
                leg_kp_mult = np.random.uniform(low, high, size=(num_reset, 1)).astype(get_global_dtype())
            if getattr(dr, "randomize_kd", False):
                low, high = dr.kd_multiplier_range
                leg_kd_mult = np.random.uniform(low, high, size=(num_reset, 1)).astype(get_global_dtype())

        # 8.2 力矩缩放随机化
        torque_scale = np.ones((num_reset, 1), dtype=get_global_dtype())
        if dr is not None and getattr(dr, "randomize_motor_torque", False):
            low, high = dr.torque_scale_range
            torque_scale = np.random.uniform(low, high, size=(num_reset, 1)).astype(get_global_dtype())

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
            "lift_force_scale": lift_scale,
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
        env._wheel_pos_err[env_ids] = 0.0
        env._wheel_err_decay_count[env_ids] = 0

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
        info_updates["_wheel_pos_err"] = env._wheel_pos_err[env_ids]

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
        vx_low = float(env._curriculum.cmd_vel_range[0, 0])
        vx_high = float(env._curriculum.cmd_vel_range[0, 1])
        yaw_low = float(env._curriculum.cmd_yaw_range[0])
        yaw_high = float(env._curriculum.cmd_yaw_range[1])

        # ── 地形相关 cmdvx 限制 ──
        terrain_types = None
        if env_ids is not None and hasattr(env, '_spawn'):
            _get_terrain = getattr(env._spawn, "get_terrain_type_for", None)
            if _get_terrain is not None:
                terrain_types = _get_terrain(env_ids)
        if terrain_types is not None:
            t_vx_low, t_vx_high = get_terrain_cmdvx_limits(terrain_types)
            t_yaw_low, t_yaw_high = get_terrain_cmd_yaw_limits(terrain_types)
            forward_bias = get_terrain_vx_forward_bias(terrain_types)
            vx_low_arr = np.full(num_reset, vx_low, dtype=dtype)
            vx_high_arr = np.full(num_reset, vx_high, dtype=dtype)
            vx_low_arr = np.maximum(vx_low_arr, t_vx_low)
            vx_high_arr = np.minimum(vx_high_arr, t_vx_high)
        else:
            vx_low_arr = np.full(num_reset, vx_low, dtype=dtype)
            vx_high_arr = np.full(num_reset, vx_high, dtype=dtype)
            t_yaw_low = None
            t_yaw_high = None
            forward_bias = None

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
        fwd_prob = np.where(stairs_mask, 0.70, 0.40).astype(dtype)
        turn_prob = np.where(stairs_mask, 0.18, 0.30).astype(dtype)
        mode_r = np.random.uniform(size=num_reset).astype(dtype)
        is_forward = mode_r < fwd_prob
        is_turn = (mode_r >= fwd_prob) & (mode_r < fwd_prob + turn_prob)
        is_rotate = mode_r >= fwd_prob + turn_prob

        vx_cmd = np.random.uniform(vx_low, vx_high, size=(num_reset,)).astype(dtype)
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
                use_uni, h_uni_low, h_uni_high, h_beta = get_terrain_height_params(terrain_types)
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

        # ── 跳跃命令：reset 时始终为 0，由中途 resample 触发 ──
        jump = np.zeros((num_reset, 1), dtype=dtype)
        commands = np.concatenate([commands, jump], axis=1)

        return commands
# =====================================================================
# 主环境类
# =====================================================================


@registry.env("BalanceInfantryJoystickFlat", sim_backend="mujoco")
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
                cell_size=float(terrain_generator.size[0] / terrain_generator.num_rows),
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

        self._vmc_ff = np.full(num_envs, ctrl_cfg.vmc.feedforward_force, dtype=self._np_dtype)
        self._base_vmc_ff = self._vmc_ff.copy()

        self._base_default_dof_pos = np.broadcast_to(
            self.get_dof_pos()[0, :4].copy(), (num_envs, 4)
        )
        self._default_dof_pos = self._base_default_dof_pos.copy()

        self._backend.set_pre_step_control(self._pre_step_motor_control)

        self._curriculum = CurriculumManager(cfg, num_envs)

        self._init_domain_randomization(BalanceInfantryJoystickDomainRandomizationProvider())
        self._init_reward_functions()

        # 延迟终止 / 差分加速度所需缓冲
        self._vertex_touch_count = np.zeros(num_envs, dtype=np.int32)
        self._tilt_count = np.zeros(num_envs, dtype=np.int32)
        self._height_low_count = np.zeros(num_envs, dtype=np.int32)
        self._swing_count = np.zeros(num_envs, dtype=np.int32)
        self._vel_track_count = np.zeros(num_envs, dtype=np.int32)
        self._last_dof_vel_for_acc = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._last_rw_body = None   
        self._last_lw_body = None
        self._episode_steps = np.zeros(num_envs, dtype=np.int32)
        self._lift_force_scale = np.ones(num_envs, dtype=self._np_dtype)
        self._powerless_active = np.zeros(num_envs, dtype=bool)
        self._term_vertex = 0
        self._term_tilt = 0
        self._term_height_low = 0
        self._term_swing = 0
        self._term_loosened = False
        self._term_vel_track = 0
        self._total_steps = 0
        self._term_window_steps = 0
        self._hit_vertex = 0
        self._hit_tilt = 0
        self._hit_height_low = 0
        self._hit_swing = 0
        self._hit_vel_track = 0

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
        self._leg_action_clip = 100.0  # curriculum 驱动，从 action_clip_initial 增长到 1.0
        self._prev_body_vx = np.zeros(num_envs, dtype=self._np_dtype)
        self._target_yaw = np.zeros(num_envs, dtype=self._np_dtype)
        # ── 手动观测归一化（33/36 维） ──
        # 组成: gyro(3) pitch(1) roll(1) cmd(4) last_act(6) wheel_err(2)
        #       sin/cos_yaw(2) dof_pos_err(4) dof_vel(6) vmc_θ(2) vmc_L0(2)
        #       [+ local_linvel(3) if obs_vel]
        self._obs_mean = np.array([
            0.0, 0.0, 0.0,     # 0-2:  gyro
            0.0,                # 3:    pitch
            0.0,                # 4:    roll
            0.0, 0.0, 0.29, 0.05,  # 5-8:  cmd(vx,vyaw,height,jump)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 9-14: last_actions
            0.0, 0.0,           # 15-16: wheel_pos_err
            0.0, 1.0,           # 17-18: sin/cos yaw_error
            0.0, 0.0, 0.0, 0.0, # 19-22: dof_pos_err
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 23-28: dof_vel
            0.0, 0.0,           # 29-30: vmc_theta
            0.28, 0.28,         # 31-32: vmc_L0
        ], dtype=self._np_dtype)
        self._obs_std = np.array([
            0.34, 0.62, 2.0,      # 0-2:  gyro (z 2.0)
            0.15,               # 3:    pitch
            0.10,               # 4:    roll
            1.5, 2.0, 0.08, 0.22,  # 5-8:  cmd
            0.6, 0.6, 0.6, 0.6, 0.6, 0.6,  # 9-14: last_actions
            8, 8,           # 15-16: wheel_pos_err
            0.6, 0.6,           # 17-18: sin/cos yaw_error
            0.3, 0.3, 0.3, 0.3, # 19-22: dof_pos_err
            1.5, 1.5, 1.5, 1.5, 22.0, 22.0,  # 23-28: dof_vel (leg 1.5, wheel 22)
            0.4, 0.4,           # 29-30: vmc_theta
            0.05, 0.05,         # 31-32: vmc_L0
        ], dtype=self._np_dtype)
        if self._cfg.obs_vel:
            self._obs_mean = np.append(self._obs_mean, np.array([0.0, 0.0, 0.0], dtype=self._np_dtype))
            self._obs_std = np.append(self._obs_std, np.array([1.5, 0.5, 0.5], dtype=self._np_dtype))
        # ── 跳跃命令 ──
        self._jump_prev_cmd = np.zeros(num_envs, dtype=self._np_dtype)
        self._jump_in_progress = np.zeros(num_envs, dtype=bool)
        self._jump_debug_enter_count = 0
        self._jump_debug_done_count = 0
        self._prev_leg_len = np.zeros((num_envs, 2), dtype=self._np_dtype)  # 上一步腿长 [右, 左]，用于计算腿长速度

        # ── 跳跃空中状态（接触滤波） ──
        self.was_in_flight = np.zeros(num_envs, dtype=bool)
        self._contact_filt = np.zeros((num_envs, 2), dtype=bool)
        self._last_contacts = np.zeros((num_envs, 2), dtype=bool)
        self._base_air_time = np.zeros(num_envs, dtype=self._np_dtype)
        self._landing_air_time = np.zeros(num_envs, dtype=self._np_dtype)
        self._base_height_phi_hold = np.zeros(num_envs, dtype=self._np_dtype)
        self._jump_max_height = np.zeros(num_envs, dtype=self._np_dtype)

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
        self._torque_scale = np.ones(num_envs, dtype=self._np_dtype)
        self._actuator_noise_std = np.zeros(num_envs, dtype=self._np_dtype)
        self._action_delay_steps = np.zeros(num_envs, dtype=np.int32)
        self._prev_targets_buffer = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._action_t2 = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._action_t3 = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._current_targets = np.zeros((num_envs, NUM_BALANCE_ACTIONS), dtype=self._np_dtype)
        self._cmd_vel_ref = float(self._cfg.commands.vel_limit[1][0])
        self._wheel_pos_err = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._wheel_err_decay_count = np.zeros(num_envs, dtype=np.int32)
        self._latest_obs_scaled = None

    def set_terrain_iteration(self, iteration: int) -> None:
        self._spawn.set_iteration(iteration)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        obs_dim = 33 + (3 if self._cfg.obs_vel else 0)
        return {"obs": obs_dim, "critic": 70 + self._height_scan_dim}

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
        self._jump_prev_cmd[env_ids] = 0.0
        self._jump_in_progress[env_ids] = False
        self._prev_leg_len[env_ids] = 0.0
        self.was_in_flight[env_ids] = False
        self._contact_filt[env_ids] = False
        self._last_contacts[env_ids] = False
        self._base_air_time[env_ids] = 0.0
        self._base_height_phi_hold[env_ids] = 0.0
        self._jump_max_height[env_ids] = 0.0
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
            wheel_kd = np.asarray(info.get("wheel_kd_multiplier", np.ones((len(env_ids), 1))), dtype=self._np_dtype).flatten()
            leg_kp = np.asarray(info.get("leg_kp_multiplier", np.ones((len(env_ids), 1))), dtype=self._np_dtype).flatten()
            leg_kd = np.asarray(info.get("leg_kd_multiplier", np.ones((len(env_ids), 1))), dtype=self._np_dtype).flatten()
            self._vmc_ff[env_ids] = self._base_vmc_ff[env_ids] * vmc_ff
            self._wheel_kd[env_ids] = self._base_wheel_kd[env_ids] * wheel_kd.reshape(-1, 1)
            self._leg_kp[env_ids] = self._base_leg_kp[env_ids] * leg_kp.reshape(-1, 1)
            self._leg_kd[env_ids] = self._base_leg_kd[env_ids] * leg_kd.reshape(-1, 1)

        self._action_smoothing[env_ids] = np.asarray(
            info.get("action_smoothing", np.full(len(env_ids), self._cfg.control_config.action_smoothing)),
            dtype=self._np_dtype,
        )
        self._torque_scale[env_ids] = np.asarray(
            info.get("torque_scale", np.ones(len(env_ids))),
            dtype=self._np_dtype,
        ).flatten()
        self._actuator_noise_std[env_ids] = np.asarray(
            info.get("actuator_noise_std", np.zeros(len(env_ids))),
            dtype=self._np_dtype,
        ).flatten()

        self._vertex_touch_count[env_ids] = 0
        self._tilt_count[env_ids] = 0
        self._height_low_count[env_ids] = 0
        self._swing_count[env_ids] = 0
        self._vel_track_count[env_ids] = 0
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
        self._lift_force_scale[env_ids] = info["lift_force_scale"]
        self._powerless_active[env_ids] = np.random.random(len(env_ids)) < self._cfg.reward_config.powerless_probability
        self._vmc_targets_fifo[env_ids] = 0.0
        self._vmc_targets[env_ids] = 0.0
        self._current_targets[env_ids] = self.default_angles
        self._prev_targets_buffer[env_ids] = self.default_angles
        self._action_t2[env_ids] = self.default_angles
        self._action_t3[env_ids] = self.default_angles

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
        clipped[:, [0, 1, 3, 4]] = np.clip(actions[:, [0, 1, 3, 4]], -leg_clip, leg_clip)
        clipped[:, [2, 5]] = np.clip(actions[:, [2, 5]], -wheel_clip, wheel_clip)

        alpha = self._action_smoothing  # per-env 随机化 α (训练时 DR, 部署时固定)
        mask = alpha < 1.0
        if np.any(mask):
            smoothed = alpha[mask, np.newaxis] * clipped[mask] + (1.0 - alpha[mask, np.newaxis]) * self._prev_filtered_actions[mask]
            self._prev_filtered_actions[mask] = smoothed.copy()
            clipped[mask] = smoothed

        clipped_actions = np.asarray(clipped, dtype=self._np_dtype)
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

        # 轮子：速度控制
        lw_target = exec_actions[:, 4] * ctrl_cfg.action_scale_wheel
        rw_target = exec_actions[:, 5] * ctrl_cfg.action_scale_wheel

        # DEBUG: 打印模型原始输出和目标
        if self._cfg.debug_enabled:
            if not hasattr(self, '_debug_apply_count'):
                object.__setattr__(self, '_debug_apply_count', 0)
            if not hasattr(self, '_debug_print_flag'):
                object.__setattr__(self, '_debug_print_flag', False)
        if self._cfg.debug_enabled and self._debug_print_flag:
            debug_obs = self._debug_last_obs
            print(f"\n[DEBUG #{self._debug_obs_count}] obs_raw (33d)=")
            obs_names = ["gyro_x","gyro_y","gyro_z","pitch","roll","cmd_vel","cmd_yaw","cmd_h","cmd_0",
                         "act0","act1","act2","act3","act4","act5","wheel_err_l","wheel_err_r",
                         "sin_yaw","cos_yaw","dof_err_LF","dof_err_RF","dof_err_LR","dof_err_RR",
                         "dvel_LF","dvel_RF","dvel_LR","dvel_RR","dvel_LW","dvel_RW",
                         "theta_l","theta_r","L0_l","L0_r"]
            for i in range(len(debug_obs)):
                print(f"  [{i:2d}] {obs_names[i]:14s} = {debug_obs[i]:+.8f}")
            print(f"  model_action = {exec_actions[0]}")
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

        phi1 = np.stack([
            joint_pos[:, 2] + 5*np.pi/6,       # 左腿: LR (后大腿) + 150°
            joint_pos[:, 3] + 5*np.pi/6,       # 右腿: RR (后大腿) + 150°
        ], axis=1)
        phi4 = np.stack([
            joint_pos[:, 0] + np.pi/6,          # 左腿: LF (前大腿) + 30°
            joint_pos[:, 1] + np.pi/6,          # 右腿: RF (前大腿) + 30°
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

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        lift_force = np.zeros((self._num_envs, 1, 3), dtype=self._np_dtype)
        lift_force[:, 0, 2] = (self._cfg.reward_config.lift_force_max * self._lift_force_scale
                               / self._cfg.sim_substeps)
        backend.apply_body_force(np.array([self._base_body_id]), lift_force)

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

        powerless_steps = int(self._curriculum.effective_powerless_duration / self._cfg.ctrl_dt)
        mask = (self._episode_steps < powerless_steps) & self._powerless_active
        motor_ctrl[mask] = 0.0

        motor_ctrl *= self._torque_scale[:, np.newaxis]

        noise_add = np.random.randn(*motor_ctrl.shape).astype(self._np_dtype) * self._actuator_noise_std[:, np.newaxis]
        motor_ctrl += noise_add
        np.clip(motor_ctrl, self._ctrl_lower, self._ctrl_upper, out=motor_ctrl)

        self._last_motor_ctrl = motor_ctrl.copy()

        if self._cfg.debug_enabled:
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
        wheel_pos_err = np.asarray(info.get("_wheel_pos_err", np.zeros((n, 2), dtype=dtype)), dtype=dtype)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)

        default_dof_pos = self._default_dof_pos[env_ids] if env_ids is not None else self._default_dof_pos

        vmc_cols = self._vmc_obs_columns(info, noisy=True)
        vmc_theta_l, vmc_theta_r, vmc_L0_l, vmc_L0_r = (
            vmc_cols[0], vmc_cols[1], vmc_cols[4], vmc_cols[5],
        )

        obs_cols = [
            noisy_gyro, pitch, roll, commands[:, [0, 2, 3, 4]], last_actions,
            wheel_pos_err[:, 0].reshape(-1, 1),
            wheel_pos_err[:, 1].reshape(-1, 1),
            np.sin(yaw_error).reshape(-1, 1),
            np.cos(yaw_error).reshape(-1, 1),
            (dof_pos[:, :4] - default_dof_pos[:, :4]),
            noisy_dof_vel,
            vmc_theta_l, vmc_theta_r, vmc_L0_l, vmc_L0_r,
        ]
        if self._cfg.obs_vel:
            obs_cols.append(local_linvel)
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

        wheel_pos_l_fmod = np.fmod(dof_pos[:, 4] + np.pi, 2 * np.pi) - np.pi
        wheel_pos_r_fmod = np.fmod(dof_pos[:, 5] + np.pi, 2 * np.pi) - np.pi
        critic_flat = np.asarray(np.concatenate([
            world_linvel,
            gyro, pitch, roll, commands[:, [0, 2, 3, 4]], last_actions,
            wheel_pos_l_fmod.reshape(-1, 1),
            wheel_pos_r_fmod.reshape(-1, 1),
            local_linvel,
            np.sin(yaw_error).reshape(-1, 1),
            np.cos(yaw_error).reshape(-1, 1),
        ] + self._vmc_obs_columns(info, noisy=False), axis=1), dtype=dtype)

        action_t2 = np.asarray(info.get("_action_t2", np.zeros((n, NUM_BALANCE_ACTIONS), dtype=dtype)), dtype=dtype)
        action_t3 = np.asarray(info.get("_action_t3", np.zeros((n, NUM_BALANCE_ACTIONS), dtype=dtype)), dtype=dtype)
        dof_acc = np.asarray(info.get("qacc", np.zeros((n, NUM_BALANCE_ACTIONS), dtype=dtype)), dtype=dtype)
        torques = np.asarray(info.get("torques", np.zeros((n, NUM_BALANCE_ACTIONS), dtype=dtype)), dtype=dtype)
        base_mass = np.asarray(info.get("_base_mass", np.zeros(n, dtype=dtype)), dtype=dtype).reshape(-1, 1)
        base_com = np.asarray(info.get("_base_com", np.zeros((n, 3), dtype=dtype)), dtype=dtype)
        ground_friction = np.asarray(info.get("_ground_friction", np.zeros((n, 2), dtype=dtype)), dtype=dtype)
        feedforward_force = np.asarray(info.get("_vmc_ff", np.zeros(n, dtype=dtype)), dtype=dtype).reshape(-1, 1)

        dof_pos_wrapped = dof_pos.copy()
        dof_pos_wrapped[:, 4] = wheel_pos_l_fmod
        dof_pos_wrapped[:, 5] = wheel_pos_r_fmod

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
            height_scan_obs(self, self._cfg.terrain_scan, n),
        ], axis=1, dtype=dtype)
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
        return {"obs": obs_delayed, "critic": critic}

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
            "roll": self._reward_roll,
            "pitch": self._reward_pitch,
            "tilt_precise": self._reward_tilt_precise,
            "action_rate": common_rewards.action_rate,
            "action_smooth": self._reward_action_smooth,
            "dof_vel": self._reward_dof_vel,
            "dof_acc": common_rewards.dof_acc,
            "torques": common_rewards.dof_torques_l2,
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
            "swing_vel": self._reward_swing_vel,
            "upright_swing_bonus": self._reward_upright_swing_bonus,
            # ── 轮子速度直接跟随命令 ────────────
            "wheel_target": self._reward_wheel_target,
            # ── 倒立摆侧倾稳定 ───────────────────────────────────
            # ── 机体加速度跟踪（单独一项，不与 θ_tg 耦合） ─────
            "body_accel_tracking": self._reward_body_accel_tracking,
            # ── 轮子空转/打滑惩罚 ───────────────────────────
            "wheel_slip": self._reward_wheel_slip,
            # ── 碰撞惩罚（机体接触地面） ────────
            "collision": self._reward_collision,
            # ── 终止条件按步惩罚 ────────────────
            "vertex_penalty": self._reward_vertex_penalty,
            # ── 站立惩罚：无命令时惩罚速度和摆角变化 ──────
            "stand": self._reward_stand,
            # ── 跳跃奖励：cmd_jump=1 时奖励向上速度 ──────
            # ── 跳跃奖励（6 个） ────────────────
            "takeoff_extend": self._reward_takeoff_extend,
            "flight": self._reward_flight,
            "encourage_jump": self._reward_encourage_jump,
            "line_z": self._reward_line_z,
            "base_height_flight": self._reward_base_height_flight,
            "leg_tuck": self._reward_leg_tuck,
            # ── 轮子接触惩罚 ──────────────────
            "wheel_contact": self._reward_wheel_contact,
            # ── 功率软约束 ──
            "wheel_power": self._reward_wheel_power,
        }

    # ── A. 姿态 / 高度 ──────────────────────────────────────────────

    def _reward_wheel_contact(self, ctx: RewardContext) -> np.ndarray:
        """轮子接触惩罚：非跳跃模式下，轮子应保持接地。

        每个离地的轮子扣 1 分。跳跃模式（jump_active=True）下不惩罚。
        """
        contact_filt = np.asarray(ctx.info.get("contact_filt", np.ones((ctx.num_envs, 2), dtype=bool)), dtype=bool)
        jump_active = np.asarray(ctx.info.get("jump_active", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        num_wheels_off = (2 - np.sum(contact_filt, axis=1)).astype(get_global_dtype())
        return np.asarray(num_wheels_off * (~jump_active).astype(get_global_dtype()), dtype=get_global_dtype())

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
        threshold = 1.0  # N
        info = ctx.info
        penalised = [
            np.asarray(info["base_body_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["left_front_leg_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["left_front_child1_leg_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["left_front_leg1_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["left_rear_child1_leg2_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["left_rear_leg_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["left_rear_leg2_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["left_rear_child1_leg1_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["right_front_leg_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["right_front_child1_leg_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["right_front_leg1_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["right_rear_child1_leg2_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["right_rear_leg_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["right_rear_leg2_contact_force"], dtype=get_global_dtype()).flatten(),
            np.asarray(info["right_rear_child1_leg1_contact_force"], dtype=get_global_dtype()).flatten(),
        ]
        stacked = np.stack(penalised, axis=-1)  # (N, 15)
        return (stacked > threshold).astype(get_global_dtype()).sum(axis=-1)  # (N,)
    def _reward_vertex_penalty(self, ctx: RewardContext) -> np.ndarray:
        """机体接触惩罚：步数越多扣越多。"""
        return self._vertex_touch_count.astype(get_global_dtype())

    def _reward_lin_vel_z(self, ctx: RewardContext) -> np.ndarray:
        reward = np.square(ctx.linvel[:, 2])
        jump_active = np.asarray(ctx.info.get("jump_active", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        reward = np.where(jump_active, reward * 0.2, reward)
        return reward

    def _reward_base_height(self, ctx: RewardContext) -> np.ndarray:
        rc = self._cfg.reward_config
        sigma = float(getattr(rc, "base_height_sigma", 0.10))
        actual_height = np.asarray(ctx.info["base_height_above_ground"], dtype=get_global_dtype())
        jump_active = np.asarray(ctx.info.get("jump_in_progress", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        target_height = np.asarray(ctx.info["commands"][:, 3], dtype=get_global_dtype())
        contact_filt = np.asarray(ctx.info.get("contact_filt", np.ones((ctx.num_envs, 2), dtype=bool)), dtype=bool)
        on_ground = np.any(contact_filt, axis=1)

        # 跳跃：追最大高度，目标 0.44
        self._jump_max_height = np.where(
            jump_active & ~on_ground,
            np.maximum(self._jump_max_height, actual_height),
            self._jump_max_height,
        )
        jump_err = self._jump_max_height - 0.44
        jump_phi = np.square(jump_err) / (sigma * sigma)

        # 非跳跃：追命令高度
        normal_err = actual_height - target_height
        normal_phi = np.square(normal_err) / (sigma * sigma)

        phi = np.where(jump_active, jump_phi, normal_phi)
        self._base_height_phi_hold = np.where(on_ground, phi, self._base_height_phi_hold)
        phi = np.where(on_ground, phi, self._base_height_phi_hold)
        self._jump_max_height = np.where(~jump_active, 0.0, self._jump_max_height)
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

        jump_active = np.asarray(ctx.info.get("jump_active", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        mask_jumping = jump_active.astype(get_global_dtype())
        err_scale = 1.0 - mask_jumping * 0.6

        roll = np.arcsin(np.clip(-gravity_body[:, 1], -1.0, 1.0))
        R_l = 0.21

        leg_tgt = h_tgt - 0.023  # commands[:,3] 是 height_cmd, 减 0.023 得到真实腿长目标
        target_left = leg_tgt + R_l * np.tan(roll)
        target_right = leg_tgt - R_l * np.tan(roll)

        err_l = leg_l - target_left
        err_r = leg_r - target_right

        sigma = float(getattr(rc, "leg_length_sigma", 0.02))
        sigma_sq = sigma * sigma

        raw = np.exp(-(np.square(err_l * err_scale) + np.square(err_r * err_scale)) / (2.0 * sigma_sq))
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

    def _soft_posture_gate(
        self,
        tilt: np.ndarray,
        theta_err: np.ndarray,
        base_z: np.ndarray,
        target_z: np.ndarray,
        *,
        low: float = 0.6,
        high: float = 1.2,
        tilt_center: float = 6.0,
        tilt_width: float = 4.0,
        swing_center: float = 18.0,
        swing_width: float = 6.0,
        height_low_width: float | np.ndarray = 0.03,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """连续软姿态门控：三个 sigmoid 的乘积，从 low 平滑过渡到 high。

        tilt / swing_err 越小、base_z >= target_z → gate → high（姿态好）。
        每个 sigmoid 在 center 处输出 0.5，width 控制过渡快慢。
        高度维度：只在 base_z < target_z 时启用门控（sigmoid 衰减），
        base_z >= target_z 时 g_height = 1.0（无惩罚）。
        """
        dtype = get_global_dtype()
        g_tilt = 1.0 / (1.0 + np.exp((np.rad2deg(tilt) - tilt_center) / tilt_width))
        g_swing = 1.0 / (1.0 + np.exp((np.rad2deg(theta_err) - swing_center) / swing_width))
        height_err = target_z - base_z
        g_height = np.where(height_err > 0, 1.0 / (1.0 + np.exp(height_err / height_low_width)), 1.0)
        combined = g_tilt * g_swing * g_height
        gate = np.asarray(low + (high - low) * combined, dtype=dtype)
        return gate, g_tilt, g_swing, g_height

    def _reward_tracking_lin_vel(self, ctx: RewardContext) -> np.ndarray:
        """世界坐标系下的速度追踪：将机体坐标系指令旋转到世界坐标系，再与世界坐标系速度比较。

        好处：转向时速度追踪更稳定，不会因 pitch 变化导致机体坐标系速度分量突变。
        sigma = sigma_base * cmd²：cmd=1m/s 时 sigma=0.25，cmd=2m/s 时 sigma=1.0。
        """
        commands = ctx.info["commands"]
        vx = ctx.info["body_vx"]
        gravity_body = ctx.info["gravity_body"]
        cos_pitch = np.cos(np.arcsin(np.clip(gravity_body[:, 0], -1.0, 1.0)))
        vx_horiz = vx * cos_pitch
        lin_vel_error = np.square(commands[:, 0] - vx_horiz)
        sigma_base = ctx.tracking_sigma_lin if ctx.tracking_sigma_lin is not None else ctx.tracking_sigma
        sigma_dyn = np.maximum(sigma_base * sigma_base, sigma_base * np.square(commands[:, 0]))
        sigma = sigma_dyn if ctx.tracking_sigma_dynamic else np.minimum(sigma_dyn, sigma_base)
        raw = np.exp(-lin_vel_error / sigma)
        return np.asarray(raw, dtype=get_global_dtype())

    def _reward_tracking_ang_vel(self, ctx: RewardContext) -> np.ndarray:
        """和 common_rewards.tracking_ang_vel 一样的高斯形式，只是在末尾加姿态门控。

        作用：避免 policy 为了追角速度指令（转向）而甩倾斜/乱晃腿——
        姿态好才给全额（×1.2），姿态差打折扣（×0.6），和直线速度的门控保持一致。
        sigma = sigma_base * cmd²：cmd=1rad/s 时 sigma=0.25，cmd=2rad/s 时 sigma=1.0。
        """
        commands = ctx.info["commands"]
        ang_vel_error = np.square(commands[:, 2] - ctx.gyro[:, 2])
        sigma_base = ctx.tracking_sigma_ang if ctx.tracking_sigma_ang is not None else ctx.tracking_sigma
        sigma_dyn = np.maximum(sigma_base * sigma_base, sigma_base * np.square(commands[:, 2]))
        sigma = sigma_dyn if ctx.tracking_sigma_dynamic else np.minimum(sigma_dyn, sigma_base)
        raw = np.exp(-ang_vel_error / sigma)
        return np.asarray(raw, dtype=get_global_dtype())

    def _reward_tracking_lin_vel_precise(self, ctx: RewardContext) -> np.ndarray:
        """精度加强版线速度追踪：σ = 原 σ / 10，对微小误差更敏感。"""
        commands = ctx.info["commands"]
        vx = ctx.info["body_vx"]
        gravity_body = ctx.info["gravity_body"]
        cos_pitch = np.cos(np.arcsin(np.clip(gravity_body[:, 0], -1.0, 1.0)))
        vx_horiz = vx * cos_pitch
        lin_vel_error = np.square(commands[:, 0] - vx_horiz)
        sigma_base = (ctx.tracking_sigma_lin if ctx.tracking_sigma_lin is not None else ctx.tracking_sigma) / 10.0
        sigma_dyn = np.maximum(sigma_base * sigma_base, sigma_base * np.square(commands[:, 0]))
        sigma = sigma_dyn if ctx.tracking_sigma_dynamic else np.minimum(sigma_dyn, sigma_base)
        raw = np.exp(-lin_vel_error / sigma)
        return np.asarray(raw, dtype=get_global_dtype())

    def _reward_tracking_lin_vel_quad(self, ctx: RewardContext) -> np.ndarray:
        """二次惩罚项：对大误差施加无上限的二次惩罚，弥补 exp 奖励在大误差时梯度消失的缺陷。

        返回 raw = -(error_x² + error_y²)，由 scale 控制惩罚强度。
        推荐 scale 为负值（如 -0.5），误差 1m/s 时惩罚 ~0.5，误差 2m/s 时惩罚 ~2.0。
        """
        commands = ctx.info["commands"]
        vx = ctx.info["body_vx"]
        gravity_body = ctx.info["gravity_body"]
        cos_pitch = np.cos(np.arcsin(np.clip(gravity_body[:, 0], -1.0, 1.0)))
        vx_horiz = vx * cos_pitch
        lin_vel_error = np.square(commands[:, 0] - vx_horiz)
        return np.asarray(lin_vel_error, dtype=get_global_dtype())

    def _reward_tracking_ang_vel_precise(self, ctx: RewardContext) -> np.ndarray:
        """精度加强版角速度追踪：σ = 原 σ / 10，对微小误差更敏感。"""
        commands = ctx.info["commands"]
        ang_vel_error = np.square(commands[:, 2] - ctx.gyro[:, 2])
        sigma_base = (ctx.tracking_sigma_ang if ctx.tracking_sigma_ang is not None else ctx.tracking_sigma) / 10.0
        sigma_dyn = np.maximum(sigma_base * sigma_base, sigma_base * np.square(commands[:, 2]))
        sigma = sigma_dyn if ctx.tracking_sigma_dynamic else np.minimum(sigma_dyn, sigma_base)
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
        _theta_err = ctx.info["theta_err"]
        _base_z = ctx.info["base_z"]
        _target_z = ctx.info["commands"][:, 3]
        _height_sigma = float(getattr(self._cfg.reward_config, "base_height_sigma", 0.05)) * 0.7
        posture_gate, _, _, _ = self._soft_posture_gate(
            _tilt, _theta_err, _base_z, _target_z, low=2.5, high=0.5,
            swing_center=10.0, swing_width=6.0, height_low_width=_height_sigma,
        )

        return np.asarray(raw * posture_gate, dtype=get_global_dtype())


    @staticmethod
    def _wheel_reward(omega: np.ndarray, target: np.ndarray) -> np.ndarray:
        wheel_r = 0.06
        wheel_speed = omega * wheel_r
        t_speed = np.where(np.abs(target) > 0.06, target, 0.0)
        has_cmd = np.abs(t_speed) > 1e-6
        sigma = np.where(has_cmd, np.maximum(np.abs(t_speed) * 0.5, 0.05), 0.25)
        return np.exp(-np.square(wheel_speed - t_speed) / (2.0 * np.square(sigma)))

    def _reward_wheel_target(self, ctx: RewardContext) -> np.ndarray:
        """前进+转向一起考虑：左右轮分别设置目标角速度（差速驱动）。

        给摆腿发动留出时间；几轮后目标自然回到原始值。
        """
        cmd_vx = ctx.info["commands"][:, 0]
        cmd_ang = ctx.info["commands"][:, 2]
        rw_vel = ctx.info["rw_vel"]
        lw_vel = ctx.info["lw_vel"]

        # 机体触地时轮子目标归零，防止栽倒时轮子继续加速
        base_contact_force = ctx.info.get("base_body_contact_force", np.zeros((ctx.num_envs, 1)))
        vertex_touch = base_contact_force.flatten() > 1.0
        safe_mask = ~vertex_touch
        cmd_vx = cmd_vx * safe_mask
        cmd_ang = cmd_ang * safe_mask

        # 差速驱动：左轮 = effective - 0.5*cmd_ang*L；右轮 = effective + 0.5*cmd_ang*L
        L = 0.380  # 轮距
        target_left = cmd_vx - 0.5 * cmd_ang * L
        target_right = cmd_vx + 0.5 * cmd_ang * L

        combined = (self._wheel_reward(lw_vel, target_left) + self._wheel_reward(rw_vel, target_right)) * 0.5
        gating = ctx.info["tracking_gate"]
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
            ctx.info.get("_action_t2", np.zeros_like(current)),
            dtype=get_global_dtype(),
        )
        diff = current[:, :4] - 2 * last[:, :4] + second_last[:, :4]
        return np.asarray(np.sum(np.square(diff), axis=1), dtype=get_global_dtype())

    # ── J. 跳跃奖励（6 个，参考 legged_robot 架构） ──────────────────────

    def _reward_takeoff_extend(self, ctx: RewardContext) -> np.ndarray:
        """蹬地伸腿：着地且 z 速度 > 0.15 m/s 时，奖励 L0 接近 jump_launch_height。"""
        contact_filt = np.asarray(ctx.info.get("contact_filt", np.ones((ctx.num_envs, 2), dtype=bool)), dtype=bool)
        on_ground = np.any(contact_filt, axis=1)
        z_vel = ctx.linvel[:, 2]
        jump_active = np.asarray(ctx.info.get("jump_active", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        takeoff_mask = on_ground & (z_vel > 0.15) & jump_active

        leg_l = np.asarray(ctx.info["leg_l"], dtype=get_global_dtype())
        leg_r = np.asarray(ctx.info["leg_r"], dtype=get_global_dtype())
        L0_target = ctx.jump_launch_height
        err = np.abs(leg_l - L0_target) + np.abs(leg_r - L0_target)
        raw = np.exp(-err * 4.0)
        return np.asarray(takeoff_mask.astype(get_global_dtype()) * raw, dtype=get_global_dtype())

    def _reward_flight(self, ctx: RewardContext) -> np.ndarray:
        """空中就给分：只要 was_in_flight 且 jump_active 为 True 就给 1.0。"""
        was_in_flight = np.asarray(ctx.info.get("was_in_flight", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        jump_active = np.asarray(ctx.info.get("jump_active", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        return np.asarray((was_in_flight & jump_active).astype(get_global_dtype()), dtype=get_global_dtype())

    def _reward_encourage_jump(self, ctx: RewardContext) -> np.ndarray:
        """鼓励跳跃：jump_active 时奖励向上速度（连续梯度）+ 落地时腾空时间结算。"""
        jump_active = np.asarray(ctx.info.get("jump_active", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        z_vel = ctx.linvel[:, 2]
        upward = np.maximum(0.0, z_vel) * jump_active.astype(get_global_dtype())
        landing_air_time = np.asarray(ctx.info.get("landing_air_time", np.zeros(ctx.num_envs)), dtype=get_global_dtype())
        landing_bonus = landing_air_time * 0.15 * jump_active.astype(get_global_dtype())
        return np.asarray(upward + landing_bonus, dtype=get_global_dtype())

    def _reward_line_z(self, ctx: RewardContext) -> np.ndarray:
        """纯上升速度奖励：空中 + jump_active 时奖励 max(0, z_vel)。"""
        was_in_flight = np.asarray(ctx.info.get("was_in_flight", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        jump_active = np.asarray(ctx.info.get("jump_active", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        z_vel = ctx.linvel[:, 2]
        return np.asarray((was_in_flight & jump_active).astype(get_global_dtype()) * np.maximum(0.0, z_vel), dtype=get_global_dtype())

    def _reward_base_height_flight(self, ctx: RewardContext) -> np.ndarray:
        """空中目标高度奖励：jump_active + was_in_flight 时，奖励 base_h 接近 flight_height_target。"""
        was_in_flight = np.asarray(ctx.info.get("was_in_flight", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        jump_active = np.asarray(ctx.info.get("jump_active", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        base_h = np.asarray(ctx.info["base_height_above_ground"], dtype=get_global_dtype())
        target = ctx.flight_height_target
        err = np.abs(base_h - target)
        raw = np.exp(-err * 6.0)
        return np.asarray((was_in_flight & jump_active).astype(get_global_dtype()) * raw, dtype=get_global_dtype())

    def _reward_leg_tuck(self, ctx: RewardContext) -> np.ndarray:
        """空中收腿奖励：jump_active + was_in_flight 时，奖励 L0 接近 jump_crouch_height。"""
        was_in_flight = np.asarray(ctx.info.get("was_in_flight", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        jump_active = np.asarray(ctx.info.get("jump_active", np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
        leg_l = np.asarray(ctx.info["leg_l"], dtype=get_global_dtype())
        leg_r = np.asarray(ctx.info["leg_r"], dtype=get_global_dtype())
        L0_target = ctx.jump_crouch_height
        err = np.abs(leg_l - L0_target) + np.abs(leg_r - L0_target)
        raw = np.exp(-err * 4.0)
        return np.asarray((was_in_flight & jump_active).astype(get_global_dtype()) * raw, dtype=get_global_dtype())

    def _reward_roll(self, ctx: RewardContext) -> np.ndarray:
        g = ctx.gravity
        assert g is not None
        roll = np.arcsin(np.clip(-g[:, 1], -1.0, 1.0))
        return np.asarray(np.square(roll), dtype=get_global_dtype())

    def _reward_pitch(self, ctx: RewardContext) -> np.ndarray:
        g = ctx.gravity
        assert g is not None
        pitch = np.arcsin(np.clip(g[:, 0], -1.0, 1.0))
        return np.asarray(np.square(pitch), dtype=get_global_dtype())

    def _reward_tilt_precise(self, ctx: RewardContext) -> np.ndarray:
        rc = self._cfg.reward_config
        threshold = float(getattr(rc, "tilt_precise_sigma", 0.07))
        tilt = np.asarray(ctx.info["tilt"], dtype=get_global_dtype())
        return np.clip(1.0 - tilt / threshold, 0.0, 1.0)

    def _reward_wheel_power(self, ctx: RewardContext) -> np.ndarray:
        """轮子功率惩罚：接近上限时梯度扣分（软约束）。

        ratio = P / limit,  penalty = ratio²,  即功率越逼近上限扣得越重。
        """
        limit = ctx.wheel_power_limit
        if limit is None or limit <= 0:
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())
        dof_vel = ctx.dof_vel
        if dof_vel is None:
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())
        torques = ctx.info.get("torques")
        if torques is None:
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())
        p_left = np.abs(torques[:, 4] * dof_vel[:, 4])
        p_right = np.abs(torques[:, 5] * dof_vel[:, 5])
        ratio_left = p_left / limit
        ratio_right = p_right / limit
        penalty = ratio_left * ratio_left + ratio_right * ratio_right
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
        gravity_body = ctx.info["gravity_body"]
        cos_pitch = np.cos(np.arcsin(np.clip(gravity_body[:, 0], -1.0, 1.0)))
        vx_horiz = vx * cos_pitch
        ctrl_dt = float(self._cfg.ctrl_dt)
        k_accel = 2.0  # m/s² per (m/s error)：误差 0.5 m/s → 目标 1.0 m/s²

        prev_vx = ctx.info.get("prev_body_vx", vx)
        prev_vx_horiz = prev_vx * cos_pitch
        actual_accel = (vx_horiz - prev_vx_horiz) / ctrl_dt

        # ── 死区：|cmd_vx − vx| < 0.15 m/s 时直接给满分 ──
        speed_err = cmd_vx - vx_horiz
        in_deadzone = np.abs(speed_err) < 0.10
        target_accel = k_accel * speed_err
        target_accel = np.clip(target_accel, -1.5, 1.5)

        sigma = 1.5   # (m/s²)²：err = √1.5 ≈ 1.22 m/s² 时 reward 降到 37%
        err_sq = np.square(target_accel - actual_accel)
        raw = np.where(in_deadzone, 1.0, np.exp(-err_sq / sigma))

        _tilt = ctx.info["tilt"]
        _theta_err = ctx.info["theta_err"]
        _base_z = ctx.info["base_z"]

        _target_z = ctx.info["commands"][:, 3]
        _height_sigma = float(getattr(self._cfg.reward_config, "base_height_sigma", 0.05)) * 0.7
        gating, _, _, _ = self._soft_posture_gate(
            _tilt, _theta_err, _base_z, _target_z, low=0.6, high=1.0,
            tilt_center=8.0, tilt_width=4.0,
            swing_center=18.0, swing_width=6.0, height_low_width=_height_sigma,
        )
        return np.asarray(raw * gating, dtype=get_global_dtype())

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
            wheel_power_limit=getattr(rc, 'wheel_power_limit', None),
            base_height_target=info["commands"][:, 3],
            base_height=info.get("base_height_above_ground", np.zeros(num_envs, dtype=dtype)),
            gravity=gravity_body,
            jump_launch_height=float(getattr(rc, "jump_launch_height", 0.42)),
            jump_crouch_height=float(getattr(rc, "jump_crouch_height", 0.16)),
            flight_height_target=float(getattr(rc, "flight_height_target", 0.65)),
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
        - 保持 5 维：[vx, vy, vyaw, height, jump]
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
                vx_low = float(self._curriculum.cmd_vel_range[0, 0])
                vx_high = float(self._curriculum.cmd_vel_range[0, 1])
                yaw_low = float(self._curriculum.cmd_yaw_range[0])
                yaw_high = float(self._curriculum.cmd_yaw_range[1])

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
                    vx_low_arr = np.full(num_resample, vx_low, dtype=dtype)
                    vx_high_arr = np.full(num_resample, vx_high, dtype=dtype)
                    vx_low_arr = np.maximum(vx_low_arr, t_vx_low)
                    vx_high_arr = np.minimum(vx_high_arr, t_vx_high)
                else:
                    vx_low_arr = np.full(num_resample, vx_low, dtype=dtype)
                    vx_high_arr = np.full(num_resample, vx_high, dtype=dtype)
                    t_yaw_low = None
                    t_yaw_high = None
                    forward_bias = None

                # ── 模式化命令采样: 前进 / 转向 / 原地自转 ──
                if terrain_types is not None:
                    stairs_mask = np.array(
                        [t in ("pyramid_stairs", "pyramid_stairs_inv", "plateau") for t in terrain_types],
                        dtype=bool,
                    )
                else:
                    stairs_mask = np.zeros(num_resample, dtype=bool)
                fwd_prob = np.where(stairs_mask, 0.70, 0.40).astype(dtype)
                turn_prob = np.where(stairs_mask, 0.18, 0.30).astype(dtype)
                mode_r = np.random.uniform(size=num_resample).astype(dtype)
                is_forward = mode_r < fwd_prob
                is_turn = (mode_r >= fwd_prob) & (mode_r < fwd_prob + turn_prob)
                is_rotate = mode_r >= fwd_prob + turn_prob

                vx_cmd = np.random.uniform(vx_low, vx_high, size=(num_resample,)).astype(dtype)
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
                        use_uni, h_uni_low, h_uni_high, h_beta = get_terrain_height_params(terrain_types)
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
                jump_prob = float(getattr(self._cfg.reward_config, "jump_prob", 0.0))
                jump = np.asarray(
                    (np.random.uniform(size=(num_resample,)) < jump_prob).astype(dtype),
                    dtype=dtype,
                ).reshape(-1, 1)
                new_cmd_5d = np.concatenate([new_cmd, height, jump], axis=1)
                air_mask = self.was_in_flight[resample_mask]
                if np.any(air_mask):
                    new_cmd_5d[air_mask, -1:] = commands[resample_mask][air_mask, 4:5]
                commands[resample_mask] = new_cmd_5d

        info["commands"] = commands
        info["_cmd_resample_mask"] = resample_mask

    def _update_jump_state(self, info: dict) -> None:
        """跳跃命令：仅标记 jump_active / jump_in_progress，不修改高度目标。"""
        if not self._curriculum.jump_curriculum_enabled:
            return
        commands = info["commands"]
        cmd_jump = commands[:, 4]

        # 上升沿：标记跳跃进行中
        rising = (cmd_jump > 0.5) & (self._jump_prev_cmd < 0.5)
        if np.any(rising):
            self._jump_in_progress[rising] = True
            self._jump_debug_enter_count += np.count_nonzero(rising)

        self._jump_prev_cmd = cmd_jump.copy()

        info["jump_active"] = cmd_jump > 0.5
        info["jump_in_progress"] = self._jump_in_progress.copy()
        info["commands"] = commands

    def _check_flight(self, sensors: dict, info: dict) -> None:
        """空中检测：OR 两帧防抖，两个轮子都无接触 → was_in_flight = True。

        落地时记录 landing_air_time，清除 jump_in_progress 并清零跳跃命令。
        """
        wheel_contact = np.column_stack([
            sensors["left_wheel_contact_force"].flatten(),
            sensors["right_wheel_contact_force"].flatten(),
        ])
        contact = wheel_contact > 1.0
        self._contact_filt = np.logical_or(contact, self._last_contacts)
        self._last_contacts = contact.copy()
        was_in_flight_prev = self.was_in_flight.copy()
        self.was_in_flight[:] = False
        self.was_in_flight[np.all(~self._contact_filt, axis=1)] = True

        landing = was_in_flight_prev & ~self.was_in_flight
        self._landing_air_time = np.where(landing, self._base_air_time.copy(), 0.0)

        jump_landing = landing & self._jump_in_progress
        if np.any(jump_landing):
            self._jump_in_progress[jump_landing] = False
            info["commands"][jump_landing, 4] = 0.0
            self._jump_debug_done_count += np.count_nonzero(jump_landing)

        info["jump_in_progress"] = self._jump_in_progress.copy()
        info["jump_active"] = info["commands"][:, 4] > 0.5

    # -----------------------------------------------------------------
    # update_state：状态计算 → 塞 info → 调用 _compute_reward → 构建 obs
    # -----------------------------------------------------------------

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        self._update_jump_state(state.info)
        self._curriculum.step(self.step_counter)

        sensors = self._read_sensors()
        commands = state.info["commands"]
        _dt = float(self._cfg.ctrl_dt)

        # ── 空中检测：OR 两帧防抖，两个轮子都无接触 → 在空中 ──
        self._check_flight(sensors, state.info)

        # 空中时间：飞行时累加（上升段），落地时清零
        z_vel = sensors["world_linvel"][:, 2]
        self._base_air_time = np.where(
            self.was_in_flight,
            self._base_air_time + _dt * np.clip(z_vel, 0.0, 0.50),
            0.0,
        )

        self._update_yaw(state.info, commands, _dt)

        # ── VMC 替代传感器：摆角 θ = π/2 - φ₀ - pitch (world-frame)，来自 _compute_state ──
        rw_theta = self._theta_vmc_r.reshape(-1)  # 右腿
        lw_theta = self._theta_vmc_l.reshape(-1)  # 左腿
        tilt, base_body_contact_force, terminated = self._detect_termination(
            sensors["gravity_body"], sensors["base_pos"], sensors["base_body_contact_force"], commands, sensors["upvector"], sensors["local_linvel"],
        )
        kin = self._compute_kinematics(sensors["dof_vel"], rw_theta, lw_theta, _dt)

        self._update_wheel_pos_err(sensors, commands, _dt)

        self._populate_info(state.info, sensors, tilt, base_body_contact_force, terminated, rw_theta, lw_theta, kin)

        self._prev_leg_len[:, 0] = self._vmc_L0[:, 1]
        self._prev_leg_len[:, 1] = self._vmc_L0[:, 0]

        rc = self._cfg.reward_config

        reward = self._compute_reward(
            state.info, sensors["world_linvel"], sensors["gyro"], sensors["gravity_body"],
            sensors["dof_pos"], sensors["dof_vel"], sensors["base_pos"], self._num_envs,
        )

        self._prev_body_vx = sensors["local_linvel"][:, 0].copy()
        self._accumulate_curriculum_stats(state.info, sensors["local_linvel"], commands, sensors["gyro"])

        if self._cfg.debug_enabled:
            self._debug_print(state.info, reward, commands, sensors, tilt, base_body_contact_force)

        obs = self._compute_obs(
            state.info, sensors["gyro"], sensors["gravity_body"], sensors["dof_pos"],
            sensors["dof_vel"],
            local_linvel=sensors["local_linvel"],
            world_linvel=sensors["world_linvel"],
        )
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
                ).reshape(-1, 4, 3).sum(axis=1),
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

        body_corner_min_z = base_height_from_scan(self) - np.abs(gravity_body[:, 0]) * 0.28 - np.abs(gravity_body[:, 1]) * 0.14
        vertex_touch = (body_corner_min_z < 0.03) & ~powerless_mask

        angle_limit = rc.tilt_limit_tight
        body_tilt_too_high = (tilt > angle_limit) & ~powerless_mask

        base_height = base_height_from_scan(self) + 0.141 * upvector[:, 2]
        height_collapse = (base_height < rc.min_height_tight) & ~powerless_mask

        max_swing = np.maximum(np.abs(self._theta_vmc_l), np.abs(self._theta_vmc_r))
        swing_too_high = (max_swing > rc.swing_angle_limit) & ~powerless_mask

        # 课程完成收敛：只剩 tilt 终止；命中计数照常记录，用于 [term] 诊断
        loosen_mask = np.bool_(self._term_loosen_active(rc))
        self._vertex_touch_count = np.where(vertex_touch & ~loosen_mask, self._vertex_touch_count + 1, 0)
        self._tilt_count = np.where(body_tilt_too_high, self._tilt_count + 1, 0)
        self._height_low_count = np.where(height_collapse & ~loosen_mask, self._height_low_count + 1, 0)
        self._swing_count = np.where(swing_too_high & ~loosen_mask, self._swing_count + 1, 0)

        # ── 速度跟踪终止：|vx| / |cmd_vx| < vel_track_ratio_threshold 且 |cmd_vx| > vel_track_min_cmd ──
        cmd_vx = commands[:, 0]
        vx = local_linvel[:, 0]
        vel_track_ratio_threshold = float(getattr(rc, "vel_track_ratio_threshold", 0.3))
        vel_track_min_cmd = float(getattr(rc, "vel_track_min_cmd", 0.2))
        vel_track_poor = (
            (np.abs(cmd_vx) > vel_track_min_cmd)
            & (np.abs(vx) / (np.abs(cmd_vx) + 1e-6) < vel_track_ratio_threshold)
            & ~powerless_mask
        )
        self._vel_track_count = np.where(vel_track_poor, self._vel_track_count + 1, 0)

        terminated = (
            (self._vertex_touch_count >= rc.n_steps_delay)
            | (self._tilt_count >= rc.n_steps_delay)
            | (self._height_low_count >= rc.n_steps_delay)
            | (self._swing_count >= rc.n_steps_delay)
            | (self._vel_track_count >= rc.n_steps_delay)
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
        self._hit_vel_track += int(np.sum(vel_track_poor))

        if np.any(terminated):
            self._term_vertex += int(np.sum((self._vertex_touch_count >= rc.n_steps_delay) & terminated))
            self._term_tilt += int(np.sum((self._tilt_count >= rc.n_steps_delay) & terminated))
            self._term_height_low += int(np.sum((self._height_low_count >= rc.n_steps_delay) & terminated))
            self._term_swing += int(np.sum((self._swing_count >= rc.n_steps_delay) & terminated))
            self._term_vel_track += int(np.sum((self._vel_track_count >= rc.n_steps_delay) & terminated))

        if self._term_window_steps >= 200 * n_envs:
            self._term_vertex = self._term_tilt = self._term_height_low = self._term_swing = self._term_vel_track = 0
            self._term_window_steps = 0
            self._hit_vertex = self._hit_tilt = self._hit_height_low = self._hit_swing = self._hit_vel_track = 0

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

    def _update_wheel_pos_err(self, sensors: dict, commands: np.ndarray, dt: float) -> None:
        vx = sensors["local_linvel"][:, 0]
        cmd_vx = commands[:, 0]
        cmd_wyaw = commands[:, 2]

        target_rw = (cmd_vx + 0.5 * cmd_wyaw * WHEELBASE) / WHEEL_RADIUS
        target_lw = (cmd_vx - 0.5 * cmd_wyaw * WHEELBASE) / WHEEL_RADIUS

        actual_rw = sensors["dof_vel"][:, 4]
        actual_lw = sensors["dof_vel"][:, 5]

        err_dot = np.stack([target_lw - actual_lw, target_rw - actual_rw], axis=1)

        low_speed = np.abs(vx) < WHEEL_ERR_VEL_THRESHOLD
        high_speed = ~low_speed

        self._wheel_pos_err += err_dot * dt * low_speed[:, np.newaxis]

        self._wheel_err_decay_count[low_speed] = 0

        entering_high = high_speed & (self._wheel_err_decay_count == 0)
        self._wheel_err_decay_count[entering_high] = WHEEL_ERR_DECAY_STEPS

        decaying = self._wheel_err_decay_count > 0
        if np.any(decaying):
            remain = self._wheel_err_decay_count[decaying].astype(self._np_dtype)
            self._wheel_pos_err[decaying] *= ((remain - 1.0) / remain).reshape(-1, 1)
            self._wheel_err_decay_count[decaying] -= 1

        self._wheel_pos_err = np.clip(self._wheel_pos_err, -8.0, 8.0)

    def _populate_info(self, info: dict, sensors: dict, tilt: np.ndarray, base_body_contact_force: np.ndarray,
                       terminated: np.ndarray, rw_theta: np.ndarray, lw_theta: np.ndarray,
                       kin: dict) -> None:
        swing_target_theta = np.full_like(rw_theta, np.deg2rad(0.0))
        leg_r = self._vmc_L0[:, 1].reshape(-1)
        leg_l = self._vmc_L0[:, 0].reshape(-1)
        c_left_z, c_right_z = self._compute_vmc_contact_world_z(sensors["base_pos"][:, 2], sensors["gravity_body"])
        wheel_world_z = np.minimum(c_left_z, c_right_z)
        g_body = sensors["gravity_body"]
        base_height = base_height_from_scan(self) + 0.141 * sensors["upvector"][:, 2]
        body_corner_min_z = base_height_from_scan(self) - np.abs(g_body[:, 0]) * 0.28 - np.abs(g_body[:, 1]) * 0.14
        theta_err = np.maximum(np.abs(rw_theta - swing_target_theta), np.abs(lw_theta - swing_target_theta))
        height_sigma = float(getattr(self._cfg.reward_config, "base_height_sigma", 0.05)) * 0.7
        tracking_gate, g_tilt, g_swing, g_height = self._soft_posture_gate(
            tilt, theta_err, sensors["base_pos"][:, 2], info["commands"][:, 3],
            low=0.6, high=1.2, height_low_width=height_sigma,
        )
        info.update({
            "tilt": tilt,
            "base_z": sensors["base_pos"][:, 2],
            "base_height_above_ground": base_height_from_scan(self) + 0.141 * sensors["upvector"][:, 2] ,
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
            "_wheel_pos_err": self._wheel_pos_err,
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
            "prev_body_vx": self._prev_body_vx,
            "body_vx": sensors["local_linvel"][:, 0],
            "body_vy": sensors["local_linvel"][:, 1],
            "upvector": sensors["upvector"],
            "gravity_body": sensors["gravity_body"],
            "leg_r": leg_r,
            "leg_l": leg_l,
            "_prev_leg_len": self._prev_leg_len.copy(),
            "wheel_world_z": wheel_world_z,
            "tracking_gate": tracking_gate,
            "g_tilt": g_tilt,
            "g_swing": g_swing,
            "g_height": g_height,
            "spin_mask": self._spin_mask,
            "was_in_flight": self.was_in_flight,
            "contact_filt": self._contact_filt,
            "landing_air_time": self._landing_air_time,
            "_action_t2": self._action_t2.copy(),
            "_action_t3": self._action_t3.copy(),
            "_base_mass": self._cached_base_mass.copy(),
            "_base_com": self._cached_base_com.copy(),
            "_ground_friction": self._cached_ground_friction.copy(),
            "_vmc_ff": self._vmc_ff.copy(),
            "_vmc_L0": self._vmc_L0.copy(),
            "_vmc_L0_dot": self._vmc_L0_dot.copy(),
            "_vmc_theta_dot": self._vmc_theta_dot.copy(),
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
        lin_vel_error = np.square(commands[:, 0] - local_linvel[:, 0])
        sigma = np.maximum(sigma_lin * sigma_lin, sigma_lin * np.square(commands[:, 0]))

        tracking_total = s["tracking_lin_vel"] * np.exp(-lin_vel_error / sigma)
        max_tracking = s["tracking_lin_vel"]
        self._episode_tracking_reward_sum += tracking_total / max_tracking

        roll_aware_height = np.asarray(info.get("base_height_above_ground", np.zeros(self._num_envs)), dtype=self._np_dtype)
        target_height = np.asarray(info["commands"][:, 3], dtype=self._np_dtype)
        jump_in_progress = np.asarray(info.get("jump_in_progress", np.zeros(self._num_envs, dtype=bool)), dtype=bool)
        skip_jump = jump_in_progress  # 跳跃中高度目标大幅变化，不用于 curriculum 评估
        self._episode_height_err_sum += np.where(skip_jump, 0.0, np.abs(roll_aware_height - target_height))
    
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

        print(f"  [DR]  ff×{ff_mean:.2f}  α={as_mean:.2f}  τ_scale={float(np.mean(self._torque_scale)):.2f}  noise={float(np.mean(self._actuator_noise_std)):.3f}  fric=({friction0:.2f},{friction1:.2f})")
        print(f"  [pose] tilt={np.rad2deg(tilt.mean()):+.1f}°  pitch={pitch_mean:+.1f}°(|{pitch_abs:.1f}°)  roll={roll_mean:+.1f}°(|{roll_abs:.1f}°)  upvec=({up_x:+.2f},{up_y:+.2f},{up_z:+.2f})  cos={cos_pitch_raw:.3f}  jump={int(np.median(info.get('jump_in_progress', np.zeros(self._num_envs, dtype=bool))))}  corner_z={float(np.mean(info.get('body_corner_min_z', np.zeros(1)))):.3f}")

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

        posture_gate = float(np.mean(info.get("tracking_gate", np.ones(self._num_envs))))
        g_tilt_val = float(np.mean(info.get("g_tilt", np.ones(self._num_envs))))
        g_swing_val = float(np.mean(info.get("g_swing", np.ones(self._num_envs))))
        g_height_val = float(np.mean(info.get("g_height", np.ones(self._num_envs))))

        sigma_arr_debug = float(getattr(self._cfg.reward_config, "base_height_sigma", 0.05))
        sigma_mean = sigma_arr_debug
        height_err_mean = float(np.mean(np.abs(info.get("base_height_above_ground", np.zeros_like(commands[:, 3])) - commands[:, 3])))
        height_err_max = float(np.max(np.abs(info.get("base_height_above_ground", np.zeros_like(commands[:, 3])) - commands[:, 3])))
        print(f"  [h_rwd] sigma={sigma_mean:.4f}  err_mean={height_err_mean:.4f}  err_max={height_err_max:.4f}  gate(t/s/h)={g_tilt_val:.3f}/{g_swing_val:.3f}/{g_height_val:.3f}")

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
        print(f"  [term]  mode={'tilt_only' if self._term_loosened else 'full'}  rate={((self._term_vertex + self._term_tilt + self._term_height_low + self._term_swing + self._term_vel_track) / max(self._term_window_steps, 1) * 1000):.1f}/1k  touch={self._hit_vertex / max(self._term_window_steps, 1) * 1000:.0f}  tilt={self._hit_tilt / max(self._term_window_steps, 1) * 1000:.0f}  h_low={self._hit_height_low / max(self._term_window_steps, 1) * 1000:.0f}  swing={self._hit_swing / max(self._term_window_steps, 1) * 1000:.0f}  vel={self._hit_vel_track / max(self._term_window_steps, 1) * 1000:.0f}  body_contact_force={float(np.mean(base_body_contact_force)):.2f}({float(np.max(base_body_contact_force)):.2f})  base_h={base_h_mean:.3f}")
        print(f"  [info]  step={int(self._episode_steps.mean())}  total={self._total_steps}  cur={self._curriculum.scale:.2f}  vx_range=[{float(self._curriculum.cmd_vel_range[0, 0]):+.2f},{float(self._curriculum.cmd_vel_range[0, 1]):+.2f}]  yaw_range=[{float(self._curriculum.cmd_yaw_range[0]):+.2f},{float(self._curriculum.cmd_yaw_range[1]):+.2f}]  σ_ang={self._curriculum.tracking_sigma_ang:.2f}  σ_yaw={self._curriculum.tracking_sigma_yaw:.2f}"
              )
        if hasattr(self, '_spawn'):
            _get_terrain = getattr(self._spawn, "get_terrain_type_for", None)
            _cfg = getattr(self._spawn, "_cfg", None)
            if _get_terrain is not None and _cfg is not None and _cfg.type_curriculum_enabled:
                terrain_types = _get_terrain(np.arange(self._num_envs))
                unique_types = sorted(set(terrain_types))
                print(f"  [terrain] type_curriculum: {', '.join(unique_types)}")
        if self._jump_debug_enter_count > 0 or self._jump_debug_done_count > 0:
            print(f"  [JUMP] enter={self._jump_debug_enter_count} done={self._jump_debug_done_count}")
            self._jump_debug_enter_count = 0
            self._jump_debug_done_count = 0


registry.register_env("BalanceInfantryJoystickFlat", BalanceInfantryJoystickEnv, sim_backend="motrix")