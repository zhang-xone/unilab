# ============================================================
# 模块 1：配置类（所有 @dataclass + registry 注册）
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from typing import cast
import numpy as np
import mujoco
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.dr import DomainRandomizationCapabilities, ResetPlan, IntervalRandomizationPlan
from unilab.dr.dr_utils import build_interval_push_plan, zero_actions

# 从 common 复用现成的配置类
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider

# 从你自己的 base.py 引入
from unilab.envs.locomotion.cod_2026_balance.base import (
    NUM_COD2026_ACTIONS,
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
    JOINT_ZERO_OFFSET,
    Cod2026BaseCfg,
    Cod2026BaseEnv,
    compute_cod2026_motor_ctrl,
)


# 传感器配置：对应 cod_2026_balance.xml 里 <sensor> 的 name ---
@dataclass
class Cod2026JoystickSensor:
    gyro: str = "gyro"              # 机体角速度 (3维)
    accel: str = "accel"            # 机体线加速度 (3维)
    local_linvel: str = "local_linvel"  # 机体线速度 (3维)
    upvector: str = "upvector"      # 机体朝上方向 (= world z 在 body frame 下的投影)
    base_pos: str = "base_pos"      # 机体绝对位置 (x, y, z)，z 即机体高度
    right_wheel: str = "right_wheel_pos"  # 右轮在机体坐标系下的位置 (x, y, z)
    left_wheel: str = "left_wheel_pos"    # 左轮在机体坐标系下的位置 (x, y, z)

# 奖励配置（简单占位，实际权重在 update_state 里硬编码） ---
@dataclass
class Cod2026RewardConfig:
    """阶段 1: 纯平衡任务奖励配置。"""
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "upright": 1.0,
        }
    )
    # 机体高度目标（m）：默认值作为 _update_commands 的 fallback
    base_height_target: float = 0.30

# 主配置类（注册到 registry，名字 = "Cod2026JoystickFlat"） ---
@registry.envcfg("Cod2026JoystickFlat")
@dataclass
class Cod2026JoystickCfg(Cod2026BaseCfg):
    # 场景文件路径（你的训练用 scene_flat.xml 即可）
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file="/home/xxx/ubuntu_software/unilab/UniLab/src/unilab/assets/robots/cod_2026_balance/scene_flat.xml"
        )
    )
    max_episode_seconds: float = 20.0  # 每个 episode 最长 20 秒（超过就截断重开）
    # 覆盖父类的 sensor：你的模型有 accel 和 base_zaxis
    sensor: Cod2026JoystickSensor = field(default_factory=Cod2026JoystickSensor) # type: ignore[assignment]
    # 指令配置：顺序 [vx, vy, vyaw]，与 play_interactive 中 KeyboardCommander 保持一致
    commands: Commands = field(
        default_factory=lambda: Commands(
            vel_limit=[[-2.5, -2.5, -2.5],  # [vx_min, vy_min, vyaw_min]
                    [ 2.5, 2.5,  2.5]],  # [vx_max, vy_max, vyaw_max]
            resampling_time=10.0,
            rel_standing_envs=0.1,
        )
    )
    # 领域随机化配置（每 N 步推一下机器人）
    #
    # 课程学习：前 ~50 个 episode 基本不推；到 ~500 episode 达到满强度。
    #   warmup = 2_000 步 (≈ 2 个 episode × 20s / 0.02s → 约 2000 step)
    #   full   = 500_000 步 (≈ 500 个 episode)
    #   实际值按你的训练节奏可在 YAML 里覆盖:
    #       env_cfg_override:
    #         domain_rand:
    #           curriculum_full_steps: 200000
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
    # 奖励配置（先给默认值，yaml 里也可以覆盖）
    reward_config: Cod2026RewardConfig = field(default_factory=Cod2026RewardConfig)

# ============================================================
# 模块 2：领域随机化 Provider（告诉环境 reset 时怎么做）
# ============================================================

class Cod2026JoystickDomainRandomizationProvider(LocomotionDRProvider):
    """Cod2026 平衡机器人的领域随机化 Provider。

    继承 LocomotionDRProvider 获得通用逻辑，覆盖 reset 相关方法。

    课程学习 (Curriculum Learning)
    -------------------------------
    训练前期减小所有扰动强度，随着 `env.step_counter` 递增，扰动强度平滑地
    从 ``initial_scale`` 增长到 ``1.0``。训练初期机器人只需学会站立即可获得
    高奖励，避免因为强外部推力过早地学习出不稳定的策略。

    扰动强度缩放系数 ``α`` 作用于：
      * 机体推动力 ``max_force``
      * 腿部推动力 ``LEG_MAX_FORCE``
      * reset 时的摆角 / pitch / roll / 角速度 等初始扰动
    """

    # ------------------------------------------------------------------
    # 课程学习辅助方法
    # ------------------------------------------------------------------
    def _get_curriculum_scale(self, env: Any) -> float:
        """根据 ``env.step_counter`` 计算当前扰动强度缩放系数 ``α ∈ [initial_scale, 1.0]``。"""
        dr = getattr(env.cfg, "domain_rand", None)
        if dr is None or not getattr(dr, "curriculum_enabled", False):
            return 1.0

        step_counter = getattr(env, "step_counter", 0)
        warmup = int(getattr(dr, "curriculum_warmup_steps", 0))
        full = int(getattr(dr, "curriculum_full_steps", 1))
        initial = float(getattr(dr, "curriculum_initial_scale", 0.0))
        shape = str(getattr(dr, "curriculum_shape", "linear"))

        # 热身阶段维持初始值
        if step_counter <= warmup:
            return float(initial)
        # 达到满强度后保持 1.0
        if step_counter >= full:
            return 1.0

        # 在 [warmup, full] 区间内做 0 → 1 的归一化
        progress = float(step_counter - warmup) / float(max(1, full - warmup))
        if shape == "smooth":
            # smoothstep: 3x² - 2x³，两端导数为 0，中间平滑
            t = float(progress)
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            normalized = 3.0 * t * t - 2.0 * t * t * t
        elif shape == "quadratic":
            # 先慢后快: t²，早期扰动增长更慢，让策略先稳定
            t = float(progress)
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            normalized = t * t
        else:  # linear
            normalized = float(progress)
            normalized = 0.0 if normalized < 0.0 else 1.0 if normalized > 1.0 else normalized

        return float(initial + (1.0 - initial) * normalized)

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        """校验环境是否支持间歇性 push 等能力。"""
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
        """持续 N 步的扰动：每 push_interval 步抽一次力，连续作用 3~8 步。

        课程学习: 随着 step_counter 增加，推动力从 initial_scale → 1.0 单调上升；
        同时 push_interval 从 push_interval_initial → push_interval_final 逐渐变小（推得越来越频繁）。
        """
        domain_rand = getattr(env.cfg, "domain_rand", None)
        if domain_rand is None or not getattr(domain_rand, "push_robots", False):
            return None

        # --- 延迟初始化持续推力的实例状态（Provider 不是 dataclass） ---
        if not hasattr(self, "_push_remaining_steps"):
            self._push_remaining_steps = 0
            self._push_base_force = None
            self._push_leg_body_ids = None
            self._push_leg_force = None

        backend = env._backend
        num_envs = backend._num_envs
        model = backend._model

        # ======= 情况 1：有正在进行的持续 push，直接用旧力 =======
        if self._push_remaining_steps > 0:
            self._push_remaining_steps -= 1
            return IntervalRandomizationPlan(
                push_perturbation_limit=self._push_base_force,
                body_ids=self._push_leg_body_ids,
                body_force=self._push_leg_force,
            )

        # ======= 情况 2：没在推 → 先算当前 curriculum scale，再判断是否该推 =======
        scale = self._get_curriculum_scale(env)
        # push_interval 随 curriculum 变化：前期大（500步≈5秒一推），后期小（100步≈1秒一推）
        # 兼容旧配置：如果只有 push_interval 就直接用它
        pi_init = getattr(domain_rand, "push_interval_initial", None)
        pi_final = getattr(domain_rand, "push_interval_final", None)
        if pi_init is not None and pi_final is not None:
            current_push_interval = int(pi_init - scale * (pi_init - pi_final))
        else:
            current_push_interval = int(getattr(domain_rand, "push_interval", 200))
        if step_counter % current_push_interval != 0:
            return None

        # ======= 情况 3：触发新一次 push → 抽力 + 抽持续时长 =======

        # --- body force (机体) ---
        base_force_base = domain_rand.max_force
        base_force = [float(f) * scale for f in base_force_base]
        base_body = getattr(domain_rand, "push_body_name", "base")

        # --- leg force (腿) ---
        LEG_BODY_NAMES = [
            "Right_rear_child1_link",
            "Left_rear_child1_link",
        ]
        leg_force_cfg = getattr(domain_rand, "leg_max_force", None)
        if leg_force_cfg is not None:
            LEG_MAX_FORCE_BASE = [float(x) for x in leg_force_cfg]
        else:
            LEG_MAX_FORCE_BASE = [20.0, 20.0, 3.0]
        LEG_MAX_FORCE = [f * scale for f in LEG_MAX_FORCE_BASE]

        leg_body_ids = []
        leg_body_valid_names = []
        for bn in LEG_BODY_NAMES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
            if bid >= 0:
                leg_body_ids.append(bid)
                leg_body_valid_names.append(bn)
            else:
                print(f"[WARN] leg body '{bn}' not found in model, skipping")
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

        # --- 持续步数 3~8 步（可调） ---
        duration = int(np.random.randint(3, 9))
        self._push_remaining_steps = duration
        self._push_base_force = base_force
        self._push_leg_body_ids = leg_body_ids if len(leg_body_ids) > 0 else None
        self._push_leg_force = body_force

        print(
            f"[PUSH] step={step_counter} duration={duration} "
            f"curriculum_scale={scale:.3f} "
            f"base={base_body} f={[round(x,2) for x in base_force]} "
            f"legs={leg_body_valid_names} f={[round(x,2) for x in LEG_MAX_FORCE]}"
        )

        return IntervalRandomizationPlan(
            push_perturbation_limit=base_force,
            body_ids=self._push_leg_body_ids,
            body_force=body_force,
        )

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        """reset 时：随机位置 + 随机朝向 + 采样指令。

        课程学习: 根据 ``env.step_counter`` 调整初始扰动范围。
          scale → 0: 几乎从平衡姿态直接起步, 有利于早期稳定收敛
          scale → 1: 恢复到原有扰动范围, 让策略学会从各种初始姿态恢复
        """
        num_reset = len(env_ids)

        # ---- 课程学习: 计算当前初始扰动强度缩放因子 α ----
        scale = self._get_curriculum_scale(env)

        # 1) 复制默认位置/速度
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))

        # 2) xy 位置随机偏移——扩大到 4×4 m 范围，避免每局都在原点附近
        #    不参与课程学习，因为 xy 位置不影响平衡任务难度
        qpos[:, 0:2] += np.random.uniform(-2.0, 2.0, (num_reset, 2))

        # 3) 随机 yaw 朝向（不参与课程学习）
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))

        # --- 3b) 三类随机化（机体姿态 / 轮子速度 / leg 速度）---
        # 3b-1) leg 摆角: 不碰 qpos（闭链结构，改位置会触发不可控的约束反作用）。
        #        改用 leg_qvel 注入动能（见 3b-4），等效外力冲量。

        # 3b-0) 生成 4 个互斥扰动权重（核心改动）
        #   约束：w_pitch + w_roll + w_linvel + w_angvel ≈ 2.0，每个 ∈ [0, 1]
        #   算法：sample 4 uniform → 迭代 clip(0,1) + 归一化到 sum=2，3 次收敛
        #   效果：不会出现 pitch/roll/linvel/angvel 同时大扰动的"四面受敌"场景
        raw_w = np.random.uniform(0.0, 1.0, (num_reset, 4))
        w = raw_w.copy()
        for _ in range(3):
            w = np.clip(w, 0.0, 1.0)
            s = w.sum(axis=1, keepdims=True)
            s = np.where(s < 1e-6, 1.0, s)  # 防除 0
            w = w / s * 2.0
        w = np.clip(w, 0.0, 1.0)  # 最终兜底，保证严格 ≤1
        w_pitch = w[:, 0]
        w_roll  = w[:, 1]
        w_lin   = w[:, 2]
        w_ang   = w[:, 3]

        # 3b-2) base pitch/roll 扰动: 四元数先乘 pitch(绕 y) 再乘 roll(绕 x)
        #        每个 envs 各用独立的 w_pitch / w_roll；同时乘 curriculum scale
        #        pitch_base=0.25rad(≈14°)，scale=0.1 → 最大 ±1.4°（前期几乎从平衡起步）
        pitch_base = 0.25
        roll_base  = 0.25
        pitch = np.random.uniform(
            -pitch_base * w_pitch * scale,
            +pitch_base * w_pitch * scale,
            (num_reset,),
        )
        roll = np.random.uniform(
            -roll_base * w_roll * scale,
            +roll_base * w_roll * scale,
            (num_reset,),
        )
        cpt, spt = np.cos(pitch / 2), np.sin(pitch / 2)
        q_pitch = np.stack([cpt, np.zeros(num_reset), spt, np.zeros(num_reset)], axis=1)
        cpr, spr = np.cos(roll / 2), np.sin(roll / 2)
        q_roll = np.stack([cpr, spr, np.zeros(num_reset), np.zeros(num_reset)], axis=1)
        qpos[:, 3:7] = np_quat_mul(np_quat_mul(q_roll, q_pitch), qpos[:, 3:7])

        # 3b-3) wheel 初始速度: 14-joint 中轮子索引=6,13 → qvel 索引=12,19
        #        左右轮独立随机: 满强度 ±8.0 rad/s；训练前期按 scale 缩小
        #        独立随机 = 制造差动不对称 = 强迫策略学会用差动控制平衡
        wheel_vel_base = 4.0
        wheel_qvel_offsets = np.array([12, 19], dtype=np.int32)
        wheel_vel = np.random.uniform(
            -wheel_vel_base * scale,
            wheel_vel_base * scale,
            (num_reset, len(wheel_qvel_offsets)),
        )
        qvel[:, wheel_qvel_offsets] = qvel[:, wheel_qvel_offsets] + wheel_vel

        # 3b-4) leg joint 初速度: 14-joint 中主腿索引=0,4,7,11 → qvel 索引=6,10,13,17
        #        满强度: ±20.0 rad/s；训练前期按 scale 缩小
        #        独立随机: 左右腿不同初始动能 → 强迫策略处理不对称
        swing_vel_base = 3.0
        leg_qvel_offsets = np.array([6, 10, 13, 17], dtype=np.int32)
        swing_vel = np.random.uniform(
            -swing_vel_base * scale,
            swing_vel_base * scale,
            (num_reset, len(leg_qvel_offsets)),
        )
        # 6=右前qvel, 10=右后qvel, 13=左前qvel, 17=左后qvel
        qvel[:, 6]  = qvel[:, 6]  + swing_vel[:, 0]  # 右前: 独立随机
        qvel[:, 10] = qvel[:, 10] + swing_vel[:, 1]  # 右后: 独立随机
        qvel[:, 13] = qvel[:, 13] + swing_vel[:, 2]  # 左前: 独立随机
        qvel[:, 17] = qvel[:, 17] + swing_vel[:, 3]  # 左后: 独立随机

        # 4) body linvel/angvel 扰动（等效外力冲量: Δv = F·dt/m）
        #    每个 envs 各用独立的 w_lin / w_ang；同时乘 curriculum scale
        #    linvel_base=0.5 m/s, angvel_base=2.5 rad/s
        #    scale=0.1 → linvel 最大 ±0.05 m/s, angvel 最大 ±0.25 rad/s（前期几乎不动）
        linvel_base = 0.5
        angvel_base = 1.5
        # 把 w_lin/w_ang 扩到 (num_reset, 3) 维：三个分量共享同一个权重（不拆分 xyz）
        w_lin_3 = np.broadcast_to(w_lin[:, np.newaxis], (num_reset, 3))
        w_ang_3 = np.broadcast_to(w_ang[:, np.newaxis], (num_reset, 3))
        qvel[:, 0:3] = np.asarray(
            np.random.uniform(
                -linvel_base * w_lin_3 * scale,
                +linvel_base * w_lin_3 * scale,
                size=(num_reset, 3),
            ),
            dtype=get_global_dtype(),
        )
        qvel[:, 3:6] = np.asarray(
            np.random.uniform(
                -angvel_base * w_ang_3 * scale,
                +angvel_base * w_ang_3 * scale,
                size=(num_reset, 3),
            ),
            dtype=get_global_dtype(),
        )

        # 5) 采样速度指令（vel_x / yaw_vel 也受课程学习缩放：前期只有小指令）
        commands = self._sample_commands(env, num_reset, scale)

        # 6) 构造 info_updates: reset 后 last/current actions 都从 0 开始
        #    ⚠️ actions ∈ [-1, 1]，用 -JOINT_ZERO_OFFSET/scale（≈ -1.5）会超出
        #       policy 正常输出范围，导致 action_rate 惩罚爆炸
        #    → 像 go2w 一样直接用零初始化，让 PD 自然把关节驱动到 JOINT_ZERO_OFFSET
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
            randomization=None,  # 先不做质量/摩擦等物理参数随机化
        )

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
        gravity: Any,
        dof_pos: Any,
        dof_vel: Any,
    ) -> dict[str, np.ndarray]:
        """reset 之后立刻算一次 obs —— 直接调用 env 的 _compute_obs。

        注意：gyro/gravity/dof_pos/dof_vel 已经被上层切成只含 reset 的 env（shape = [len(env_ids), ...]）
        wheel 位置也要切，否则维度对不上。
        """
        del linvel
        # reset 时：wheel 位置也要切片到只含 reset 的 envs
        rw_all, lw_all = env.get_wheel_pos()
        rw = rw_all[env_ids]
        lw = lw_all[env_ids]
        return env._compute_obs(info_updates, gyro, gravity, dof_pos, dof_vel, rw, lw)

    def _sample_commands(
        self,
        env: Any,
        num_reset: int,
        curriculum_scale: float = 1.0,
    ) -> np.ndarray:
        """采样指令：[vx, vy, vyaw, height]（与 play_interactive.KeyboardCommander 对齐）

        从 env.cfg.commands.vel_limit 读取 vx/vyaw 范围，从 env.cfg.commands.height_limit 读取 height 范围，
        这样 play_interactive 的探针检查才能正常工作。

        - vx: 80% 概率给非零指令，20% 保持 0
        - vy: 两轮机器人无侧向速度，始终为 0
        - vyaw: 80% 概率给非零指令，20% 保持 0
        - height: 在 [height_limit[0], height_limit[1]] 范围内均匀随机
        """
        vel_limit = np.asarray(env.cfg.commands.vel_limit, dtype=get_global_dtype())
        low = vel_limit[0]   # [vx_min, vy_min, vyaw_min]
        high = vel_limit[1]  # [vx_max, vy_max, vyaw_max]

        height_limit = np.asarray(env.cfg.commands.height_limit, dtype=get_global_dtype())
        h_low = float(height_limit[0])
        h_high = float(height_limit[1])

        commands = np.zeros((num_reset, 4), dtype=get_global_dtype())
        # vx: 80% 概率给非零指令
        moving_x = np.random.uniform(size=num_reset) < 0.8
        commands[:, 0] = np.where(
            moving_x,
            np.random.uniform(low[0] * curriculum_scale, high[0] * curriculum_scale, size=num_reset),
            0.0,
        )
        # vyaw: 80% 概率给非零指令
        turning = np.random.uniform(size=num_reset) < 0.8
        commands[:, 2] = np.where(
            turning,
            np.random.uniform(low[2] * curriculum_scale, high[2] * curriculum_scale, size=num_reset),
            0.0,
        )
        # vy: 两轮机器人始终为 0（已由 zeros 初始化）
        # height: 在高度范围内均匀随机（不参与 curriculum 缩放）
        commands[:, 3] = np.random.uniform(h_low, h_high, size=num_reset)
        return commands

# ============================================================
# 模块 3：主环境类
# ============================================================

@registry.env("Cod2026JoystickFlat", sim_backend="mujoco")
class Cod2026JoystickEnv(Cod2026BaseEnv):
    _cfg: Cod2026JoystickCfg  # type: ignore[override]

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

        # Cache wheel body ids for world-frame body kinematics
        try:
            self._right_wheel_body_id = self._backend.get_body_ids(
                ["Right_Wheel_link"]
            )[0]
            self._left_wheel_body_id = self._backend.get_body_ids(
                ["Left_Wheel_link"]
            )[0]
        except Exception:
            self._right_wheel_body_id = None
            self._left_wheel_body_id = None

        # --- torque 上下界（从 actuator 定义读取，对齐 go2w）---
        ctrl_range = np.asarray(
            self._backend.get_actuator_ctrl_range(), dtype=np.float64
        )
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)

        # --- PD 增益（每关节独立，存成 (num_envs, N) 数组，对齐 go2w）---
        # leg_kp/leg_kd：4 维（仅腿关节），compute 中用局部索引 enumerate(LEG_JOINT_INDICES) 访问
        leg_kp_arr = np.asarray(self._cfg.control_config.leg_kp, dtype=self._np_dtype)
        leg_kd_arr = np.asarray(self._cfg.control_config.leg_kd, dtype=self._np_dtype)
        # wheel_kd：2 维 [R_wheel_kd, L_wheel_kd]，compute 中用 idx - NUM_LEG_ACTIONS 取局部索引
        wheel_cfg = np.asarray(self._cfg.control_config.wheel_kd, dtype=self._np_dtype)
        self._motor_kp = np.broadcast_to(leg_kp_arr, (num_envs, NUM_LEG_ACTIONS)).copy()
        self._motor_kd = np.broadcast_to(leg_kd_arr, (num_envs, NUM_LEG_ACTIONS)).copy()
        self._wheel_kd = np.broadcast_to(wheel_cfg, (num_envs, NUM_WHEEL_ACTIONS)).copy()
        self._last_motor_ctrl = np.zeros((num_envs, NUM_COD2026_ACTIONS), dtype=self._np_dtype)

        # --- 注册：在 MuJoCo step 之前，用此函数把 targets 转 torque ---
        self._backend.set_pre_step_control(self._pre_step_motor_control)

        self._init_domain_randomization(Cod2026JoystickDomainRandomizationProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + upvector(3) + dof_pos(6) + dof_vel(6) + last_actions(6) + commands(4) + theta(2) + L(2) = 32
        # 名字用 "obs"，跟 rsl_rl wrapper 的 hardcode 对齐
        return {"obs": 32}

    # ==================== apply_action：策略 action → targets（对齐 go2w）====================
    def reset(self, env_indices: np.ndarray):
        """覆盖 NpEnv.reset：物理状态 reset 后，还要清零 episode 级的计数器和速度缓存。

        否则 _vertex_touch_count 在 reset 后如果新姿态恰好也贴地，
        会从 40 继续累加到 41，立刻又 terminated=True。
        """
        obs, info = super().reset(env_indices)

        # ---- 清零终止计数器 ----
        if hasattr(self, "_vertex_touch_count") and self._vertex_touch_count is not None:
            self._vertex_touch_count[env_indices] = 0
        if hasattr(self, "_body_low_count") and self._body_low_count is not None:
            self._body_low_count[env_indices] = 0
        if hasattr(self, "_tilt_count") and self._tilt_count is not None:
            self._tilt_count[env_indices] = 0

        # ---- 清零差分速度缓存（reset 后上一帧位置无效）----
        if hasattr(self, "_last_dof_vel") and self._last_dof_vel is not None:
            self._last_dof_vel[env_indices] = 0.0
        if hasattr(self, "_last_rw_angle") and self._last_rw_angle is not None:
            self._last_rw_angle[env_indices] = 0.0
        if hasattr(self, "_last_lw_angle") and self._last_lw_angle is not None:
            self._last_lw_angle[env_indices] = 0.0
        if hasattr(self, "_last_tilt") and self._last_tilt is not None:
            self._last_tilt[env_indices] = 0.0

        # ---- 清零积分项（新 episode 从 0 开始累积）----
        if hasattr(self, "_integral_height") and self._integral_height is not None:
            self._integral_height[env_indices] = 0.0
        if hasattr(self, "_integral_tilt") and self._integral_tilt is not None:
            self._integral_tilt[env_indices] = 0.0
        if hasattr(self, "_integral_linvel") and self._integral_linvel is not None:
            self._integral_linvel[env_indices] = 0.0
        if hasattr(self, "_integral_swing") and self._integral_swing is not None:
            self._integral_swing[env_indices] = 0.0

        # ---- 清除 state 上的 terminated 标志（否则这一步返回给调用者时 terminated 还是 True）----
        if self._state is not None:
            self._state.terminated[env_indices] = False
            self._state.truncated[env_indices] = False

        return obs, info

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """策略 action → targets，不直接算 torque

        腿关节: target_pos = action * action_scale + default_angles
        轮子关节: target_vel = action * wheel_action_scale
        返回的 targets 被 _pre_step_motor_control 拿去算 PD torque。
        """
        clipped_actions = np.asarray(
            np.clip(
                actions,
                -self._cfg.control_config.clip_actions,
                self._cfg.control_config.clip_actions,
            ),
            dtype=self._np_dtype,
        )
        state.info["last_actions"] = state.info.get(
            "current_actions", np.zeros_like(clipped_actions)
        )
        state.info["current_actions"] = clipped_actions
        exec_actions = clipped_actions

        # targets 预分配 + 按索引 in-place 写入（joint 顺序：腿+轮子交错）
        targets = np.zeros(
            (clipped_actions.shape[0], NUM_COD2026_ACTIONS), dtype=self._np_dtype
        )
        for idx in [0, 1, 3, 4]:  # 腿关节: target_pos
            targets[:, idx] = (
                exec_actions[:, idx] * self._cfg.control_config.action_scale
                + self.default_angles[idx]
            )
        for idx in [2, 5]:  # 轮子关节: target_vel
            targets[:, idx] = (
                exec_actions[:, idx] * self._cfg.control_config.wheel_action_scale
            )
        return targets

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        """MuJoCo step 前被调用：targets → PD → torque

        对齐 go2w/joystick.py _pre_step_motor_control。
        """
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
        # 缓存最新状态（在 _compute_reward 的调试打印中统一输出，不每帧打）
        self._debug_joint_pos = joint_pos
        self._debug_joint_vel = joint_vel
        self._debug_motor_ctrl = motor_ctrl
        return motor_ctrl

    def get_accel(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.accel),
            dtype=get_global_dtype(),
        )

    def get_upvector(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.upvector),
            dtype=get_global_dtype(),
        )

    def get_wheel_pos(self) -> tuple[np.ndarray, np.ndarray]:
        """读取左右轮子在机体坐标系下的 (x, y, z) 位置。

        返回: (rw_pos, lw_pos)，形状都是 (num_envs, 3)
        """
        rw = np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.right_wheel),
            dtype=get_global_dtype(),
        )
        lw = np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.left_wheel),
            dtype=get_global_dtype(),
        )
        return rw, lw

    def get_base_pos(self) -> np.ndarray:
        """机体在世界坐标系下的位置 (x, y, z)，z 即机体高度。"""
        return np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.base_pos),
            dtype=get_global_dtype(),
        )

    def _compute_obs(
        self,
        info: dict,
        gyro: np.ndarray,
        upvector: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        rw: np.ndarray,   # 轮子位置从外面传进来
        lw: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """把传感器数据拼成策略观测。

        ⚠️ dof_pos 已经是偏置后的值（= 真实 joint_pos - JOINT_ZERO_OFFSET）
        → 策略观测到的 "零点" 对应 "前腿水平向前" 姿态
        """
        n = gyro.shape[0]

        last_actions = info.get(
            "last_actions",
            np.zeros((n, NUM_COD2026_ACTIONS), dtype=get_global_dtype()),
        )
        commands = info.get(
            "commands",
            np.zeros((n, 4), dtype=get_global_dtype()),
        )
        # ========== 用传进来的 rw/lw 计算 theta 和 L ==========
        rw_x, rw_z = rw[:, 0], rw[:, 2]
        rw_L = np.sqrt(rw_x**2 + rw_z**2)
        rw_theta = -np.arctan2(rw_x, -rw_z)
        lw_x, lw_z = lw[:, 0], lw[:, 2]
        lw_L = np.sqrt(lw_x**2 + lw_z**2)
        lw_theta = -np.arctan2(lw_x, -lw_z)
        # ======================================================

        obs_flat = np.concatenate([
            gyro, upvector, dof_pos, dof_vel, last_actions, commands,
            rw_theta.reshape(-1, 1),   # 加右腿 theta
            lw_theta.reshape(-1, 1),   # 加左腿 theta
            rw_L.reshape(-1, 1),        # 加右腿 L
            lw_L.reshape(-1, 1),        # 加左腿 L
        ], axis=1)

        # 手动观测缩放：把量纲差异拉平到 ~[-1, 1]，帮助 NN 稳定学习
        _OBS_SCALE = np.array([
            # gyro(3): rad/s, 正常姿态下很小
            1.0, 1.0, 1.0,
            # upvector(3): [-1,1]
            1.0, 1.0, 1.0,
            # dof_pos(6): leg joint rad
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            # dof_vel(6): 腿关节 ±5，轮子 ±10 — 不做强缩放，让 NN 自己适应量级
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            # last_actions(6): ~[-1, 1]
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            # commands(4): [vx, vy, vyaw, height] — height 约 0.3m
            1.0, 1.0, 1.0, 0.3,
            # theta(2): 腿摆角 rad
            0.1, 0.1,
            # L(2): 腿长 m，约 0.15
            0.15, 0.15,
        ], dtype=obs_flat.dtype)

        obs_scaled = np.clip(obs_flat / _OBS_SCALE, -100.0, 100.0)

        return {"obs": obs_scaled}

    # ==================== commands 初始化/刷新（对齐 go2w 模式 + KeyboardCommander）====================
    def _update_commands(self, info: dict) -> None:
        """跟 go2w/_update_commands 模式一致：
        ① 如果 info["commands"] 尚未设置（首次 step 前），给默认值；
        ② 后续可加周期重采样。

        commands 结构为 [vx, vy, vyaw, height]
        - 前3维与 play_interactive.KeyboardCommander 保持一致
        - 第4维为高度指令，默认用 reward_config.base_height_target
        """
        commands = info.get("commands")
        dtype = get_global_dtype()
        n = self._num_envs
        if commands is None or np.asarray(commands).shape != (n, 4):
            default_height = float(self._cfg.reward_config.base_height_target)
            info["commands"] = np.concatenate(
                [np.zeros((n, 3), dtype=dtype), np.full((n, 1), default_height, dtype=dtype)],
                axis=1,
            )

    # ==================== 奖励函数（按 DDTRobotTita 分类重新设计）====================
    # 参考: 分类如下：
    #   A. 姿态/高度  B. 根节点惩罚  C. 关节惩罚  D. 动作惩罚  E. 轮子惩罚  F. 终止

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        gyro = self.get_gyro()
        upvector = self.get_upvector()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        base_pos = self.get_base_pos()
        base_linvel = self.get_local_linvel()
        rw, lw = self.get_wheel_pos()

        # ============== body-frame 摆角（额外一份奖励，与 body 倾斜无关）==============
        rw_theta_body = -np.arctan2(rw[:, 0], -rw[:, 2])
        lw_theta_body = -np.arctan2(lw[:, 0], -lw[:, 2])

        torques = self._last_motor_ctrl.copy()
        state.info["torques"] = torques

        commands = state.info["commands"]
        dtype = get_global_dtype()
        # ============== 辅助量 ==============
        # commands[:, 0] = vx, commands[:, 1] = vy（两轮机器人始终=0）
        # commands[:, 2] = vyaw（对齐 play_interactive.KeyboardCommander）
        # commands[:, 3] = height（新增高度指令
        leg_idx = np.asarray((0, 1, 3, 4))
        wheel_idx = np.asarray((2, 5))
        h_target = commands[:, 3]
        _dt = float(self._cfg.ctrl_dt)

        # ================== swing angle（world-frame, sagittal, signed）==================
        # 三步：
        #   1) 求 body right (0,-1,0) 和 body forward (1,0,0) 在 world 里的单位向量
        #   2) vec_sag = (wheel_world - base_world) − (WO · right_world) · right_world
        #      （去掉轮子在身体两侧的固定几何分量，只保留 sagittal 平面里的前后/竖直）
        #   3) θ = sign * arccos(-vec_sag_z / |vec_sag|)
        #      前负后正：vec_sag · forward_world >= 0 → 前方 → 负
        #
        # MuJoCo quat = (w, x, y, z)。R[:,1] = body y 在 world 中的方向
        # body right = -body_y = -R[:, 1]
        base_pos_world = self._backend.get_base_pos()
        rw_pos_world = self._backend.get_body_pos_w(
            np.array([self._right_wheel_body_id], dtype=np.int32)
        )[:, 0, :]
        lw_pos_world = self._backend.get_body_pos_w(
            np.array([self._left_wheel_body_id], dtype=np.int32)
        )[:, 0, :]
        rw_vec_world = rw_pos_world - base_pos_world
        lw_vec_world = lw_pos_world - base_pos_world

        q = self._backend.get_base_quat()   # (n, 4), order = (qw, qx, qy, qz)
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        right_x = -2.0 * (qx * qy - qw * qz)
        right_y = -(1.0 - 2.0 * (qx * qx + qz * qz))
        right_z = -2.0 * (qy * qz + qw * qx)
        r_norm = np.sqrt(right_x * right_x + right_y * right_y + right_z * right_z) + 1e-9
        right_x, right_y, right_z = right_x / r_norm, right_y / r_norm, right_z / r_norm

        fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
        fwd_y = 2.0 * (qx * qy + qw * qz)
        fwd_z = 2.0 * (qx * qz - qw * qy)
        f_norm = np.sqrt(fwd_x * fwd_x + fwd_y * fwd_y + fwd_z * fwd_z) + 1e-9
        fwd_x, fwd_y, fwd_z = fwd_x / f_norm, fwd_y / f_norm, fwd_z / f_norm

        def _swing_angle_world(vec_world):
            proj_len = (
                vec_world[:, 0] * right_x
                + vec_world[:, 1] * right_y
                + vec_world[:, 2] * right_z
            )
            vs_x = vec_world[:, 0] - proj_len * right_x
            vs_y = vec_world[:, 1] - proj_len * right_y
            vs_z = vec_world[:, 2] - proj_len * right_z
            vs_norm = np.sqrt(vs_x * vs_x + vs_y * vs_y + vs_z * vs_z) + 1e-9
            cos_phi = np.clip(-vs_z / vs_norm, -1.0, 1.0)
            phi = np.arccos(cos_phi)               # 与世界竖直向下的夹角，非负
            along_front = vs_x * fwd_x + vs_y * fwd_y + vs_z * fwd_z
            sign = np.where(along_front >= 0.0, -1.0, +1.0)   # 前负后正
            return sign * phi

        rw_angle_world = _swing_angle_world(rw_vec_world)
        lw_angle_world = _swing_angle_world(lw_vec_world)

        tilt = np.arccos(np.clip(upvector[:, 2], 0.0, 1.0))

        # ============== 积分项（消除稳态偏差，σ 限幅）==============
        # 比例奖励只看当前误差 → 策略可以卡在"偏差 2cm"的局部最优。
        # 积分奖励惩罚"持续偏差" → 偏差越久，扣越多，迫使归零。
        #
        # 限幅 = σ 的倍数（不用手写魔法数字，与主奖励尺度一致）：
        #   高度：σ_h = 0.03m →  ±10×σ_h = ±0.30 m·s（偏差 3cm 累积 10s 饱和）
        #   倾斜：σ_tilt ≈ 0.122rad → ±8×σ_tilt ≈ ±0.98 rad·s（倾斜 10° 累积 5s 饱和）
        #   线速：σ_lin_vel = 0.5 m/s → ±2×σ_lin_vel = ±1.00 m（偏差 0.3m/s 累积 3s 饱和）
        #
        # 抗积分饱和（Anti-windup）：
        #   1. np.clip 硬限幅（到限就停）
        #   2. terminated 环境不再累积
        #   3. reset 时清零
        if not hasattr(self, "_integral_height"):
            self._integral_height = np.zeros(base_pos.shape[0], dtype=dtype)
        if not hasattr(self, "_integral_tilt"):
            self._integral_tilt = np.zeros(base_pos.shape[0], dtype=dtype)
        if not hasattr(self, "_integral_linvel"):
            self._integral_linvel = np.zeros(base_pos.shape[0], dtype=dtype)
        if not hasattr(self, "_integral_swing"):
            self._integral_swing = np.zeros(base_pos.shape[0], dtype=dtype)

        alive = ~state.terminated if self._state is not None else np.ones(base_pos.shape[0], dtype=bool)


        # ---- 摆角 / 机体 阻尼所需：角速度（用上一帧位置差分近似）----
        # 注意：这里已经将机体坐标系摆角替换为世界坐标系摆角
        if not hasattr(self, "_last_rw_angle"):
            self._last_rw_angle = np.zeros_like(rw_angle_world)
        if not hasattr(self, "_last_lw_angle"):
            self._last_lw_angle = np.zeros_like(lw_angle_world)
        if not hasattr(self, "_last_tilt"):
            self._last_tilt = np.zeros_like(tilt)
        rw_vel = (rw_angle_world - self._last_rw_angle) / _dt
        lw_vel = (lw_angle_world - self._last_lw_angle) / _dt
        tilt_vel = (tilt - self._last_tilt) / _dt
        self._last_rw_angle = rw_angle_world.copy()
        self._last_lw_angle = lw_angle_world.copy()
        self._last_tilt = tilt.copy()

        # ============== 终止 ==============
        # vertex_touch 条件不变，但延迟 N 步终止，给策略恢复时间
        # body_too_low 也同样延迟
 
        body_length = 0.6
        sin_tilt = np.sqrt(1.0 - np.square(upvector[:, 2]))
        body_vertex_z = base_pos[:, 2] - 0.5 * body_length * np.abs(sin_tilt)
 
        vertex_touch = body_vertex_z < 0.206      # 顶点贴地（条件不变）
        body_too_low = base_pos[:, 2] < 0.22    # 高度太低（条件不变）
 
        # 连续触地计数器：累计 N_STEPS_DELAY 步才终止
        N_STEPS_DELAY = 4
        if not hasattr(self, "_vertex_touch_count"):
            self._vertex_touch_count = np.zeros(base_pos.shape[0], dtype=np.int32)
        if not hasattr(self, "_body_low_count"):
            self._body_low_count = np.zeros(base_pos.shape[0], dtype=np.int32)
        if not hasattr(self, "_tilt_count"):
            self._tilt_count = np.zeros(base_pos.shape[0], dtype=np.int32)

        self._vertex_touch_count = np.where(
            vertex_touch, self._vertex_touch_count + 1, 0
        )
        self._body_low_count = np.where(
            body_too_low, self._body_low_count + 1, 0
        )
        # tilt 过大终止：35° ≈ 0.61 rad，连续 N_STEPS_DELAY 步才终止，给策略恢复时间
        TILT_LIMIT = 35.0 * np.pi / 180.0
        body_tilt_too_high = tilt > TILT_LIMIT
        self._tilt_count = np.where(
            body_tilt_too_high, self._tilt_count + 1, 0
        )

        vertex_terminated = self._vertex_touch_count >= N_STEPS_DELAY
        body_terminated = self._body_low_count >= N_STEPS_DELAY
        tilt_terminated = self._tilt_count >= N_STEPS_DELAY
        terminated = vertex_terminated | body_terminated | tilt_terminated

        # --- 平均存活时间追踪 ---
        if not hasattr(self, "_ep_step_count"):
            self._ep_step_count = np.zeros(base_pos.shape[0], dtype=np.int32)
            self._ep_len_history = []
        just_died = np.where(terminated)[0]
        if len(just_died) > 0:
            for c in self._ep_step_count[just_died]:
                if c > 0:
                    self._ep_len_history.append(int(c))
            if len(self._ep_len_history) > 2000:
                self._ep_len_history = self._ep_len_history[-2000:]
        self._ep_step_count = np.where(terminated, 0, self._ep_step_count + 1)
        # --- 结束 ---

        r_fall = np.where(terminated, -40.0, 0.0)
        # vertex_touch 但还没到 N_STEPS_DELAY 步 → 每步轻扣，鼓励恢复
        r_fall += np.where((vertex_touch & ~vertex_terminated), -10, 0.0)
        r_fall += np.where((body_too_low & ~body_terminated), -10, 0.0)
        r_fall += np.where((body_tilt_too_high & ~tilt_terminated), -10, 0.0)
        # 存活门控：活着拿 100% 正向奖励，死了只拿 20%
        # 正向奖励都乘 gate，惩罚项（负数）保持不变——死了依然扣
        gate = np.where(
            (self._vertex_touch_count > 0) | (self._body_low_count > 0) | (self._tilt_count > 0), 0.8, 1.0
        )
        # ============== A.  高度/摆角/机体 ==============

        # A1 高度：σ=2cm, w=+100（对齐 TITA base_height_l2 理念，精确到目标高度）
        σ_h = 0.02
        height_err = np.square(base_pos[:, 2] - h_target)
        r_height = +70.0 * np.exp(-height_err / σ_h**2)
        # 高度误差惩罚：-5e3 * (0.04)² = -16
        height_wall = np.maximum(np.abs(base_pos[:, 2] - h_target) - 0.04, 0.0)
        r_height_wall = -1e5 * np.square(height_wall)   # ← 新加的
        # A1b 高度积分：消除高度稳态偏差，饱和值：-2e4 * (0.09)² = -0.9
        r_int_height = -1e4 * np.square(self._integral_height)
        r_lin_vel_z = -0.05 * np.square(base_linvel[:, 2])

        # A6 腿于竖直夹角：world-frame + body-frame 两份奖励
        σ_swing = 10.0 * np.pi / 180.0
        # --- world-frame：轮子在世界坐标系竖直下方 --- 
        swing_total_world = np.abs(rw_angle_world) + np.abs(lw_angle_world)
        r_swing_world = +8.0 * np.exp(-np.square(swing_total_world) / (2.0 * σ_swing)**2)
        sw_wall_world = np.maximum(swing_total_world - 20.0 * np.pi / 180.0, 0.0)
        r_swing_wall_world = -80.0 * np.square(sw_wall_world)
        # --- body-frame：轮子在机体坐标系竖直下方（不被 body 倾斜污染）--- 
        swing_total_body = np.abs(rw_theta_body) + np.abs(lw_theta_body)
        r_swing_body = +10.0 * np.exp(-np.square(swing_total_body) / (2.0 * σ_swing)**2)
        sw_wall_body = np.maximum(swing_total_body - 15.0 * np.pi / 180.0, 0.0)
        r_swing_wall_body = -80.0 * np.square(sw_wall_body)
        # --- 汇总：两份一起拿 --- 
        r_swing = r_swing_world + r_swing_body
        r_swing_wall = r_swing_wall_world + r_swing_wall_body
        # 积分项：用 world+body 之和累积，稳态偏差越大会被扣得越狠
        # 摆角积分：消除摆角稳态偏差，饱和值：-120 × 0.122² = −1.8 
        r_int_swing = -120.0 * np.square(self._integral_swing)
        #摆角阻尼 1 rad/s → -0.5/步，10 rad/s → -50/步（接近 r_fall 的量级，足够威慑）
        r_swing_damping = np.clip(-0.3 * (np.square(rw_vel) + np.square(lw_vel)), -5.0, 0.0)

        # A3b 双腿同步：|rw - lw| 越小越好（world + body 两份）
        σ_sym = 6.0 * np.pi / 180.0
        r_symmetry_world = +3.0 * np.exp(-np.square(rw_angle_world - lw_angle_world) / σ_sym**2)
        r_symmetry_body = +3.0 * np.exp(-np.square(rw_theta_body - lw_theta_body) / σ_sym**2)
        r_symmetry = r_symmetry_world + r_symmetry_body
        # 摆角不同步惩罚：左右腿摆角差越大扣得越狠（world + body 两份都扣）
        swing_asym_world = np.clip(-200.0 * np.square(rw_angle_world - lw_angle_world), -5.0, 0.0)
        swing_asym_body = np.clip(-200.0 * np.square(rw_theta_body - lw_theta_body), -5.0, 0.0)
        r_swing_asymmetry = swing_asym_world + swing_asym_body
        r_symmetry_wall_body = -100.0 * np.square(rw_theta_body - lw_theta_body)
        
        # A2 机体朝上：小角奖励，大角惩罚（更陡的软墙）
        #   σ_tilt = 10°：tilt=5-10° 时也能拿到一些 upright 奖励
        #   tilt_penalty 权重 -300：tilt=10°→-9.2, tilt=5°→-2.3, 两者差 7 分
        #   配合极低的 body_damping，策略能自由尝试把身体竖起来
        #   signed_tilt: 正=前倾，负=后倾
        signed_tilt = upvector[:, 0]
        σ_tilt = 5.5 * np.pi / 180.0
        r_upright = +80.0 * np.exp(-np.square(tilt) / σ_tilt**2)
        # 用 upvector 的两个水平分量平方和 → 同时惩罚 pitch 和 roll
        tilt_horiz_sq = upvector[:, 0]**2 + upvector[:, 1]**2
        # A2b 倾斜惩罚：追速度时本就该前倾，此时放宽惩罚
        #   cmd_mag=0 → 全额惩罚; cmd_mag=2.5 → 惩罚减到 40%
        cmd_mag = np.clip(np.abs(commands[:, 0]) + np.abs(commands[:, 2]), 0.0, 2.5)
        tilt_tolerance = 1.0 - 0.6 * (cmd_mag / 2.5)  # 不动时1.0，全速时0.4
        r_tilt_penalty = -600.0 * tilt_tolerance * tilt_horiz_sq
        # A2b 倾斜积分：消除倾角稳态偏差，饱和值：-200 * (0.16)² ≈ -5.1
        r_int_tilt = -200.0 * np.square(self._integral_tilt)
        #机体阻尼
        r_body_damping = np.clip(-200 * np.square(tilt_vel), -20.0, 0.0) 
        
        # A2b 保持水平（正向奖励）       
        σ_level = 0.015
        level_err = np.square(base_pos[:, 2] - body_vertex_z)
        r_keep_level = +80.0 * np.exp(-level_err / σ_level**2)
        # A2c 顶点离地body_vertex_z 越低扣越多，离地 >0.25 不扣
        # （和下方 vertex_touch 的硬计数配合，两层保护。系数从 0 开始渐进调大）
        vertex_wall = np.maximum(0.25 - body_vertex_z, 0.0)
        r_vertex_wall = -1.5e4 * np.square(vertex_wall)

        # A4 线速度 跟踪（对齐 TITA track_lin_vel_xy_exp=9.5 主奖励）：
        #   commands[:, 0] = x 方向速度指令
        #   base_linvel[:, 0] = 当前 x 速度
        σ_lin_vel = 0.2
        lin_vel_x_err_sq = np.square(base_linvel[:, 0] - commands[:, 0])
        r_track_lin_vel = +240 * np.exp(-lin_vel_x_err_sq / σ_lin_vel**2)
        # A4b 线速度偏离惩罚：速度偏离指令主动扣分，不只是少拿奖励
        #   cmd_vx=1.0 m/s, err=0.3 → -0.45/步; err=0.5 → -1.25/步
        r_lin_vel_err = -200.0 * lin_vel_x_err_sq
        # A4c 线速度积分：消除速度稳态偏差，饱和值：-20 × 0.5² = −5
        r_int_linvel = -20.0 * np.square(self._integral_linvel)
        
        # 倾斜-轮速/角速度的目标（前倾 → 正轮速推回去，同时需要正的 pitch 角速度）
        target_wheel_vel = 8.0 * signed_tilt + commands[:, 0] / 0.085
        #   倒立摆思路：前倾时需要让 pitch 角速度为正（身体向前倾，轮速向前推身体）
        #   粗略期望：ω_pitch ≈ K × tilt，只作为"不该为零"的目标方向
        target_ang_vel_pitch = 3.0 * signed_tilt

        # A5 角速度 跟踪（对齐 TITA track_ang_vel_z_exp=6.5）
        ang_vel_z_err_sq = np.square(gyro[:, 2] - commands[:, 2])
        σ_ang_vel = 0.1
        r_track_ang_vel = +60.0 * np.exp(-ang_vel_z_err_sq / σ_ang_vel**2)
        # A5b yaw 角速度偏离惩罚：偏离指令主动扣分
        #   err=0.3 rad/s → -1.8/步; err=1.0 rad/s → -20/步
        r_ang_vel_err = -40.0 * ang_vel_z_err_sq
        # 机体角速度 xy：偏离目标才扣，不惩罚绝对值；设上限避免单帧过度扣分
        #   gyro[:,0]=roll → 目标 0；gyro[:,1]=pitch → 目标 target_ang_vel_pitch
        #   clip(-5, 0)：最多扣 -5/步，防异常抖动时惩罚爆炸
        # gyro[:, 0] = roll 轴角速度，平衡时应该≈0
        r_ang_vel_xy = np.clip(
            -0.5 * np.square(gyro[:, 0]),
            -10.0, 0.0
        )

        # 轮子速度：偏离"倾斜+指令所需的预期轮速"才扣，不惩罚绝对值
        r_wheel_vel = -0.000 * (
            np.square(dof_vel[:, 2] - target_wheel_vel)
            + np.square(dof_vel[:, 5] - target_wheel_vel)
        )
        # 轮子不对称惩罚：左右速差越大扣越多
        #   权重降低：让策略先学会"用轮子推身体回去"，再学左右同步
        r_wheel_asymmetry = -0.5 * np.square(dof_vel[:, 2] - dof_vel[:, 5])

        # A2d 轮子方向奖励：有速度指令时，轮子朝指令方向转加分
        #   cmd_vx > 0 → 奖励正轮速; cmd_vx < 0 → 奖励负轮速
        #   只在有指令时生效（|cmd_vx| > 0.1），幅度随 |cmd_vx| 渐增
        cmd_vx = commands[:, 0]
        cmd_active = np.where(np.abs(cmd_vx) > 0.1, 1.0, 0.0)
        wheel_vel_avg = 0.5 * (dof_vel[:, 2] + dof_vel[:, 5])
        # 轮速的"正确方向" = sign(cmd_vx)，奖励 wheel_vel_avg * sign(cmd_vx) > 0
        wheel_dir_match = wheel_vel_avg * np.sign(cmd_vx + 1e-8)
        r_wheel_forward = +30.0 * cmd_active * np.clip(wheel_dir_match / 5.0, 0.0, 1.0)

        # 倾斜-轮速一致性：每个轮子各自朝向目标，防止"正转+反转=0均值"的漏洞
        σ_wsync = 1.5
        r_wheel_sync = +60 * (
            np.exp(-np.square(dof_vel[:, 2] - target_wheel_vel) / σ_wsync**2)
            + np.exp(-np.square(dof_vel[:, 5] - target_wheel_vel) / σ_wsync**2)
        )

        # 比例奖励的 σ 值（与 r_height / r_upright / r_track_lin_vel 对齐）

        INT_HEIGHT_LIMIT = 1 * σ_h
        INT_TILT_LIMIT = 3 * σ_tilt
        INT_LINVEL_LIMIT = 1 * σ_lin_vel
        INT_SWING_LIMIT = 1 * σ_swing

        # 高度积分：累积 (base_z - h_target)
        int_height_inc = (base_pos[:, 2] - h_target) * _dt
        self._integral_height = np.clip(
            self._integral_height + np.where(alive, int_height_inc, 0.0),
            -INT_HEIGHT_LIMIT, INT_HEIGHT_LIMIT,
        )

        # 倾斜积分：累积 signed_tilt（带符号的前后倾，正=前倾负=后倾，有正有负才能消除稳态偏差）
        # 旧版累积无符号 tilt → 永远正增长，3 秒就饱和，不符合"消除稳态偏差"语义
        int_tilt_inc = signed_tilt * _dt
        self._integral_tilt = np.clip(
            self._integral_tilt + np.where(alive, int_tilt_inc, 0.0),
            -INT_TILT_LIMIT, INT_TILT_LIMIT,
        )

        # 线速度积分：累积 (vx - cmd_vx)
        int_linvel_inc = (base_linvel[:, 0] - commands[:, 0]) * _dt
        self._integral_linvel = np.clip(
            self._integral_linvel + np.where(alive, int_linvel_inc, 0.0),
            -INT_LINVEL_LIMIT, INT_LINVEL_LIMIT,
        )

        # 摆角积分：累积 (rw_angle_world + lw_angle_world)（两条腿的平均摆角，有正有负）
        # 旧版累积 |rw|+|lw| → 永远正增长，1 秒就饱和
        # 新版：两腿平均（前摆为负后摆为正，前摆过度和后摆过度相互抵消
        swing_mean = 0.5 * (rw_angle_world + lw_angle_world)
        int_swing_inc = swing_mean * _dt
        self._integral_swing = np.clip(
            self._integral_swing + np.where(alive, int_swing_inc, 0.0),
            -INT_SWING_LIMIT, INT_SWING_LIMIT,
        )
        # ============== 积分项 END ==============
              
        # ============== C. 关节 ==============
        # joint_acc
        if not hasattr(self, "_last_dof_vel"):
            joint_acc = np.zeros_like(dof_vel)
        else:
            joint_acc = (dof_vel - self._last_dof_vel) / _dt
        self._last_dof_vel = dof_vel.copy()
        # clip joint_acc：轮子速度跳变会让 joint_acc 无限大，导致 r_joint_acc = -800/步
        joint_acc = np.clip(joint_acc, -500.0, 500.0)

        leg_torques = torques[:, leg_idx]
        r_joint_torques = -1.0e-3 * np.sum(np.square(leg_torques), axis=1)
        r_joint_acc = -2.5e-7 * np.sum(np.square(joint_acc[:, leg_idx]), axis=1)
        leg_dof_vel = dof_vel[:, leg_idx]
        leg_power = np.sum(np.abs(leg_torques * leg_dof_vel), axis=1)
        r_joint_power = 0*-2e-5 * leg_power

        # ============== D. 动作 ==============
        current = state.info.get("current_actions", np.zeros_like(dof_pos))
        last = state.info.get("last_actions", np.zeros_like(dof_pos))
        r_action_rate = -0.07 * np.sum(np.square(current - last), axis=1)

        # ============== 汇总 ==============
        reward = (
            # 正向奖励 × gate：
            gate * (
                r_height + r_upright + r_keep_level + r_swing + r_symmetry
                + r_track_lin_vel + r_track_ang_vel
            )
            # 高度类
            + r_height_wall + r_int_height
            # 机体类
            + r_tilt_penalty + r_int_tilt + r_body_damping + r_vertex_wall
            # 摆角类
            + r_swing_wall + r_symmetry_wall_body + r_swing_asymmetry + r_int_swing + r_swing_damping
            # 轮子类
            + r_wheel_vel + r_wheel_asymmetry + r_wheel_sync + r_wheel_forward
            # 速度/角度跟踪
            + r_lin_vel_err + r_int_linvel + r_ang_vel_err
            # 其他
            + r_lin_vel_z + r_ang_vel_xy
            + r_joint_torques + r_joint_acc + r_joint_power
            + r_action_rate + r_fall
        )

        # --- 调试打印 ---
        if not hasattr(self, "_reward_print_counter"):
            self._reward_print_counter = 0
        self._reward_print_counter += 1
        if self._reward_print_counter % 100 == 0:
            print(
                f"[reward] 高度: h={r_height.mean():+.3f} hw={r_height_wall.mean():+.3f} ih={r_int_height.mean():+.3f}  "
                f"机体: level={r_keep_level.mean():+.3f} up={r_upright.mean():+.3f} tilt_pen={r_tilt_penalty.mean():+.3f} vw={r_vertex_wall.mean():+.3f} it={r_int_tilt.mean():+.3f} bod={r_body_damping.mean():+.3f}  "
                f"摆角: swing={r_swing.mean():+.3f} sw_wall={r_swing_wall.mean():+.3f} symm={r_symmetry.mean():+.3f} sym_wall={r_symmetry_wall_body.mean():+.3f} sw_asym={r_swing_asymmetry.mean():+.3f} iswing={r_int_swing.mean():+.3f} swingd={r_swing_damping.mean():+.3f}  "
                f"轮子: wheelvel={r_wheel_vel.mean():+.3f} wasym={r_wheel_asymmetry.mean():+.3f} wsync={r_wheel_sync.mean():+.3f} wf={r_wheel_forward.mean():+.3f}  "
                f"跟踪: tvx={r_track_lin_vel.mean():+.3f} lve={r_lin_vel_err.mean():+.3f} iv={r_int_linvel.mean():+.3f} tvz={r_track_ang_vel.mean():+.3f} ave={r_ang_vel_err.mean():+.3f}  "
                f"其他: lvz={r_lin_vel_z.mean():+.3f} wxy={r_ang_vel_xy.mean():+.3f} jtor={r_joint_torques.mean():+.3f} jacc={r_joint_acc.mean():+.3f} jpow={r_joint_power.mean():+.3f} actr={r_action_rate.mean():+.3f} fall={r_fall.mean():+.1f} gate={gate.mean():+.2f} total={reward.mean():+.2f}"
            )
            print(f"[cmd]   vx_cmd={commands[:, 0].mean():+.3f} m/s (|vx_cmd|={np.abs(commands[:, 0]).mean():.3f})  yaw_cmd={commands[:, 2].mean():+.3f} rad/s (|yaw_cmd|={np.abs(commands[:, 2]).mean():.3f}={np.rad2deg(np.abs(commands[:, 2]).mean()):+.1f}°/s)  h_cmd={commands[:, 3].mean():.3f} m")
            print(f"[actual] |vx|={np.abs(base_linvel[:, 0]).mean():.3f} m/s  |yaw|={np.abs(gyro[:, 2]).mean():.3f} rad/s ({np.rad2deg(np.abs(gyro[:, 2]).mean()):+.1f}°/s)  |vx_err|={np.abs(base_linvel[:, 0] - commands[:, 0]).mean():.3f}  |yaw_err|={np.abs(gyro[:, 2] - commands[:, 2]).mean():.3f}")
            if len(self._ep_len_history) > 0:
                avg_steps = np.mean(self._ep_len_history)
                print(f"[ep]     avg_steps={avg_steps:.0f}  avg_time={avg_steps * 0.02:.2f}s  (最近 {len(self._ep_len_history)} 个 episode)")
            print(f"[state]  base_z={base_pos[:, 2].mean():.3f} m (tgt={h_target.mean():.3f}) tilt={np.rad2deg(tilt.mean()):+.1f}°  swing_rw={np.rad2deg(rw_angle_world.mean()):.0f}° swing_lw={np.rad2deg(lw_angle_world.mean()):.0f}° swing_rw_body={np.rad2deg(rw_theta_body.mean()):.0f}° swing_lw_body={np.rad2deg(lw_theta_body.mean()):.0f}° body_vertex_z={body_vertex_z.mean():.4f}  terminated={terminated.sum()}/{terminated.size} (body_low={(base_pos[:, 2] < 0.26).sum()} vtx_touch={(body_vertex_z < 0.22).sum()} tilt_high={(tilt > np.deg2rad(35.0)).sum()})")
            # --- PD & 轮子状态（每 200 步打印一次，不每帧刷屏）---
            if hasattr(self, "_debug_motor_ctrl") and self._debug_motor_ctrl is not None:
                mc = self._debug_motor_ctrl  # (num_envs, 6): [Rf, Rr, Rw, Lf, Lr, Lw]
                jv = self._debug_joint_vel   # (num_envs, 6)
                leg_mc = mc[:, (0,1,3,4)]
                wheel_mc = mc[:, (2,5)]
                wheel_vel = jv[:, (2,5)]
                print(f"  └ leg_torque=[Rf {leg_mc[:,0].mean():+.1f}  Rr {leg_mc[:,1].mean():+.1f}  Lf {leg_mc[:,2].mean():+.1f}  Lr {leg_mc[:,3].mean():+.1f}]  range {leg_mc.min():+.1f}~{leg_mc.max():+.1f}")
                print(f"  └ wheel_torque=[Rw {wheel_mc[:,0].mean():+.1f}  Lw {wheel_mc[:,1].mean():+.1f}]  vel=[Rw {wheel_vel[:,0].mean():+.2f} rad/s  Lw {wheel_vel[:,1].mean():+.2f} rad/s]  ctrl_limit=[{self._ctrl_lower.min():+.1f}, {self._ctrl_upper.max():+.1f}]")
        
        obs = self._compute_obs(state.info, gyro, upvector, dof_pos, dof_vel, rw, lw)
        return state.replace(
            obs=obs,
            reward=reward.astype(dtype),
            terminated=terminated,
            truncated=state.truncated,
        )