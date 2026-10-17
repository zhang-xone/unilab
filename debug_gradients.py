"""梯度流诊断脚本：在训练中实时监控梯度、loss、obs 统计。

用法 (需要先 start_training.sh):
    cd /home/xxx/ubuntu_software/unilab/UniLab
    python -c "
import debug_gradients
debug_gradients.monkey_patch_ppo_grad()
    "
然后运行训练即可看到梯度诊断输出。

或直接：
    python debug_gradients.py --plot
"""

import argparse
import os
import sys
from collections import deque
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


class PpoGradientMonitor:
    """Monkey-patches PPO.update() to log gradient / loss statistics."""

    def __init__(self, log_every: int = 1, warmup: int = 3):
        self.log_every = log_every
        self.warmup = warmup
        self.call_count = 0
        self.history: list[dict[str, float]] = []

    def __call__(self, original_update, *args, **kwargs):
        self.call_count += 1

        # 调用原始 update
        result = original_update(*args, **kwargs)

        if self.call_count <= self.warmup:
            return result

        if self.call_count % self.log_every != 0:
            return result

        # 收集梯度信息
        algo = args[0]  # self
        ac = algo.actor_critic

        total_grad = 0.0
        zero_grad_params = 0
        max_grad_val = 0.0
        nan_grad = False
        total_params = 0

        grad_by_layer: dict[str, float] = {}

        for name, param in ac.named_parameters():
            if param.grad is None:
                continue
            total_params += 1
            g = param.grad.detach()
            gn = g.norm().item()
            total_grad += gn**2

            if gn < 1e-10:
                zero_grad_params += 1
            if torch.isnan(g).any():
                nan_grad = True

            max_g = g.abs().max().item()
            if max_g > max_grad_val:
                max_grad_val = max_g

            # 按层分组
            parts = name.split(".")
            layer = parts[0] if len(parts) == 1 else ".".join(parts[:2])
            grad_by_layer[layer] = grad_by_layer.get(layer, 0.0) + gn**2

        total_grad = total_grad**0.5

        # 打印
        import torch

        action_std = ac.distribution.std_param.item() if hasattr(ac, "distribution") else -1

        result_dict = result if isinstance(result, tuple) else (result,)
        if len(result_dict) == 0:
            result_dict = (0.0,)

        surr_loss = 0.0  # will be accessed from return
        val_loss = 0.0
        mean_kl = 0.0

        # The original update returns (mean_value_loss, mean_surrogate_loss, mean_kl, ...)
        if len(result) >= 3:
            val_loss = float(result[0])
            surr_loss = float(result[1])
            mean_kl = float(result[2])

        status = "❌NAN" if nan_grad else ("⚠️ZERO" if zero_grad_params > total_params * 0.3 else "✅")

        print(f"\n[GRAD #{self.call_count}] {status} | "
              f"|g|={total_grad:.4f} "
              f"surr={surr_loss:.4f} val={val_loss:.4f} kl={mean_kl:.6f} "
              f"σ={action_std:.4f} "
              f"z={zero_grad_params}/{total_params} "
              f"max_g={max_grad_val:.4f}")

        for layer, gn2 in sorted(grad_by_layer.items(), key=lambda x: -x[1]):
            if gn2 > 0.001:
                print(f"   {layer:30s} |g|={gn2**0.5:.4f}")

        self.history.append({
            "iter": self.call_count,
            "grad_norm": total_grad,
            "surr_loss": surr_loss,
            "val_loss": val_loss,
            "kl": mean_kl,
            "action_std": action_std,
            "zero_grad_ratio": zero_grad_params / max(total_params, 1),
        })

        return result


def monkey_patch_ppo_grad(log_every: int = 1, warmup: int = 3):
    """Monkey-patch PPO.update() to add gradient monitoring.

    Call BEFORE training starts.
    """
    import importlib
    import torch

    from unilab.algos.torch.ppo.rsl_rl.algorithms.ppo import PPO

    monitor = PpoGradientMonitor(log_every=log_every, warmup=warmup)
    original_update = PPO.update

    def patched_update(self, *args, **kwargs):
        return monitor(original_update.__get__(self, type(self)), *args, **kwargs)

    PPO.update = patched_update
    print(f"✅ PPO.update() 已注入梯度监控 (log_every={log_every}, warmup={warmup})")
    return monitor


def check_obs_distribution(task: str, num_steps: int = 100):
    """独立检查：env 的观测、奖励、优势分布统计（不依赖 PPO 训练）。"""
    print(f"\n{'='*60}")
    print(f"观测/奖励分布诊断: task={task}, steps={num_steps}")
    print(f"{'='*60}")

    import numpy as np

    if task == "jump":
        from unilab.envs.locomotion.balance_infantry.jump import (
            BalanceInfantryJumpEnv,
            BalanceInfantryJumpCfg,
        )
        env_cls = BalanceInfantryJumpEnv
        cfg_cls = BalanceInfantryJumpCfg
    else:
        from unilab.envs.locomotion.balance_infantry.joystick import (
            BalanceInfantryJoystickEnv,
            BalanceInfantryJoystickCfg,
        )
        env_cls = BalanceInfantryJoystickEnv
        cfg_cls = BalanceInfantryJoystickCfg

    cfg = cfg_cls()
    cfg.ctrl_dt = 0.02
    cfg.max_episode_seconds = 10.0
    cfg.obs_vel = False
    cfg.obs_manual_norm = False
    cfg.reward_config.base_height_target = 0.25
    cfg.control_config.action_delay_range = [1, 4]
    cfg.control_config.obs_delay_range = [1, 5]
    cfg.commands.resampling_time = 3.0
    cfg.commands.vx_range = [-3.0, 3.0]
    cfg.commands.vyaw_range = [-5.0, 5.0]
    cfg.commands.height_limit = [0.20, 0.35]
    cfg.commands.jumph_range = [0.0, 0.3]
    cfg.commands.rel_standing_envs = 0.3
    cfg.noise_config.scale_gyro = 0.01
    cfg.noise_config.scale_joint_vel = 0.01
    cfg.noise_config.scale_vmc_phi0 = 0.005
    cfg.noise_config.scale_vmc_phi0_dot = 0.2
    cfg.noise_config.scale_vmc_L0 = 0.005
    cfg.noise_config.scale_vmc_L0_dot = 0.05
    cfg.terrain_scan.num_scan = 64
    cfg.terrain_scan.scan_distance = 3.0
    cfg.terrain_scan.scan_height_offset = 0.0
    cfg.termination_config.max_tilt = 0.8
    cfg.cmd_curriculum_enabled = False
    cfg.jump_curriculum_enabled = False
    cfg.debug_enabled = False

    env = env_cls(cfg, num_envs=4)

    env.init_state()
    env_ids = np.arange(4, dtype=np.int32)
    env.reset(env_ids)

    all_rewards = []
    all_obs = []

    # 简单的随机策略 rollout
    for step in range(num_steps):
        actions = np.random.uniform(-1, 1, (4, 6)).astype(np.float32)
        state = env.step(actions)
        all_rewards.append(state.reward)
        all_obs.append(state.obs["obs"])

    all_rewards = np.array(all_rewards)
    all_obs = np.array(all_obs)

    # 奖励统计
    print(f"\n奖励统计 ({num_steps} steps):")
    rew_flat = all_rewards.reshape(-1)
    print(f"  mean={np.mean(rew_flat):+.6f}  std={np.std(rew_flat):.6f}  "
          f"min={np.min(rew_flat):+.6f}  max={np.max(rew_flat):+.6f}")
    for i in range(4):
        env_rew = all_rewards[:, i]
        print(f"  env[{i}]: mean={np.mean(env_rew):+.6f}  std={np.std(env_rew):+.6f}  "
              f"sum={np.sum(env_rew):+.6f}")

    # 观测统计
    print(f"\n观测统计:")
    obs_flat = all_obs.reshape(-1, all_obs.shape[-1])
    obs_mean = np.mean(obs_flat, axis=0)
    obs_std = np.std(obs_flat, axis=0)
    obs_min = np.min(obs_flat, axis=0)
    obs_max = np.max(obs_flat, axis=0)

    labels = [
        "gyro_x", "gyro_y", "gyro_z", "pitch", "roll",
        "cmd_vx", "cmd_vyaw", "cmd_h", "cmd_j",
        "a0", "a1", "a2", "a3", "a4", "a5",
        "whl_l", "whl_r", "sin_y", "cos_y",
        "deLF", "deRF", "deLR", "deRR",
        "dvLF", "dvRF", "dvLR", "dvRR", "dvLW", "dvRW",
        "th_l", "th_r", "L0_l", "L0_r",
    ]

    for i in range(all_obs.shape[-1]):
        label = labels[i] if i < len(labels) else f"col_{i}"
        if obs_std[i] < 1e-6:
            print(f"  [{i:2d}] {label:8s}  mean={obs_mean[i]:+10.4f}  std={obs_std[i]:10.6f}  ⚠️ ZERO_STD")
        elif obs_std[i] > 20:
            print(f"  [{i:2d}] {label:8s}  mean={obs_mean[i]:+10.4f}  std={obs_std[i]:10.4f}  ⚠️ HUGE_STD")
        elif abs(obs_mean[i]) > 10:
            print(f"  [{i:2d}] {label:8s}  mean={obs_mean[i]:+10.4f}  std={obs_std[i]:10.4f}  ⚠️ OFFSET")
        elif np.any(np.isnan(obs_flat[:, i])):
            print(f"  [{i:2d}] {label:8s}  mean={obs_mean[i]:+10.4f}  std={obs_std[i]:10.4f}  ⚠️ NaN")

    # 检查 cmd 是否有在变化的
    cmd_cols = [5, 6, 7, 8]
    cmd_std = obs_std[cmd_cols]
    if np.all(cmd_std < 1e-6):
        print(f"\n  ❌ 所有 cmd 列 std=0! 命令没有变化!")
    elif np.any(cmd_std < 1e-6):
        for i in cmd_cols:
            if obs_std[i] < 1e-6:
                label = labels[i] if i < len(labels) else f"col_{i}"
                print(f"\n  ⚠️ {label} std=0 (命令可能永远不变)")

    # 检查 last_actions 和 sensor 值是否有变化
    early_cols = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14]
    for i in early_cols:
        if obs_std[i] < 1e-6:
            label = labels[i] if i < len(labels) else f"col_{i}"
            print(f"  ⚠️ {label} std=0 (sensor/action 冻结)")

    print(f"\n✅ 检查完成，正常维度: {np.sum(obs_std > 1e-6)}/{all_obs.shape[-1]}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["jump", "joystick"], default="jump")
    parser.add_argument("--plot", action="store_true", help="plot gradient history")
    args = parser.parse_args()

    if args.plot:
        print("Plot not yet supported; run with --task to check obs distribution")
        sys.exit(0)

    check_obs_distribution(args.task)