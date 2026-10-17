from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainRandConfig:
    randomize_base_mass: bool = False
    added_mass_range: list[float] = field(default_factory=lambda: [-1.5, 1.5])

    randomize_body_mass: bool = False
    body_mass_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    random_com: bool = False
    com_offset_x: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_y: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.05, 0.05])

    randomize_gravity: bool = False
    gravity_range: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, -9.81], [0.0, 0.0, -9.81]]
    )

    randomize_ground_friction: bool = False
    ground_friction_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])

    randomize_geom_solref: bool = False
    geom_solref_timeconst_range: list[float] = field(default_factory=lambda: [0.01, 0.05])
    geom_solref_dampratio_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_dof_armature: bool = False
    dof_armature_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])

    randomize_joint_friction: bool = False
    leg_main_damping_range: list[float] = field(default_factory=lambda: [0.05, 0.2])
    leg_main_frictionloss_range: list[float] = field(default_factory=lambda: [0.25, 1.0])
    wheel_damping_range: list[float] = field(default_factory=lambda: [0.0, 0.01])
    wheel_frictionloss_range: list[float] = field(default_factory=lambda: [0.05, 0.25])
    passive_damping_range: list[float] = field(default_factory=lambda: [0.01, 0.025])
    passive_frictionloss_range: list[float] = field(default_factory=lambda: [0.05, 0.1])
    gimbal_damping_range: list[float] = field(default_factory=lambda: [0.002, 0.01])
    gimbal_frictionloss_range: list[float] = field(default_factory=lambda: [0.002, 0.01])
    # 倍数随机 (基于 XML/模型默认值, 0.1~10x)
    damping_multiplier_range: list[float] = field(default_factory=lambda: [0.1, 10.0])
    frictionloss_multiplier_range: list[float] = field(default_factory=lambda: [0.1, 10.0])

    randomize_kp: bool = False
    randomize_kd: bool = False
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])

    randomize_vmc_theta_kp: bool = False
    randomize_vmc_theta_kd: bool = False
    randomize_vmc_l0_kp: bool = False
    randomize_vmc_l0_kd: bool = False
    vmc_kp_multiplier_range: list[float] = field(default_factory=lambda: [0.6, 1.4])
    vmc_kd_multiplier_range: list[float] = field(default_factory=lambda: [0.6, 1.4])
    randomize_vmc_ff: bool = False
    vmc_ff_multiplier_range: list[float] = field(default_factory=lambda: [0.6, 1.4])
    randomize_wheel_kd: bool = False
    wheel_kd_multiplier_range: list[float] = field(default_factory=lambda: [0.6, 1.4])

    randomize_motor_torque: bool = False
    torque_scale_range: list[float] = field(default_factory=lambda: [0.7, 1.0])

    actuator_noise: bool = False
    actuator_noise_std_range: list[float] = field(default_factory=lambda: [0.0, 1.0])

    randomize_inertia: bool = False
    inertia_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])

    randomize_default_dof_pos: bool = False
    randomize_default_dof_pos_range: list[float] = field(default_factory=lambda: [-0.05, 0.05])

    push_robots: bool = False
    # 静态 push_interval（旧配置方式，保持兼容）
    push_interval: int = 750  # step
    # push_interval 随 curriculum 变化（新配置方式，优先使用）
    # 前期 interval 大（推得少），后期 interval 小（推得多）
    push_interval_initial: int | None = None
    push_interval_final: int | None = None
    max_force: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.5])
    # 腿部额外推动的最大力；不配置时退化为 max_force
    leg_max_force: list[float] | None = None
    push_body_name: str | None = None

    # ---------------------------------------------------------------------------
    # 课程学习 (Curriculum Learning)：扰动强度随训练步数从 0 → 1 单调上升。
    #
    # curriculum_enabled   : 是否启用课程学习
    # curriculum_warmup_steps : 热身步数。在此之前所有乘子 = initial_scale
    # curriculum_full_steps : 饱和步数。超过此步数后乘子 = 1.0
    # curriculum_initial_scale : 初始扰动强度 (0 ~ 1)
    # curriculum_shape     : 曲线形状
    #     "linear"   (线性上升)
    #     "smooth"   (平滑 sigmoid, 避免突变)
    #     "quadratic"(先慢后快, 利于早期稳定)
    #
    # 该参数会同时作用于：
    #   * 推动力  max_force / LEG_MAX_FORCE (周期性 push)
    #   * 初始摆角 swing 范围 (reset 时刻)
    #   * 初始 pitch/roll 范围 (reset 时刻)
    #   * 初始角速度 范围 (reset 时刻)
    # ---------------------------------------------------------------------------
    curriculum_enabled: bool = False
    curriculum_warmup_steps: int = 0
    curriculum_full_steps: int = 100000
    curriculum_initial_scale: float = 0.05
    curriculum_shape: str = "smooth"