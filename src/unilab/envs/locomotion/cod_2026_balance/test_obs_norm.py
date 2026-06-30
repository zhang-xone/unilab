"""最精简测试：跑几百步，看观测各维度的实际值范围。

用法:
    cd /home/xxx/ubuntu_software/unilab/UniLab
    uv run python3 src/unilab/envs/locomotion/cod_2026_balance/test_obs_norm.py
"""

from __future__ import annotations

import sys
import numpy as np

from unilab.envs.locomotion.cod_2026_balance.joystick import (
    Cod2026JoystickCfg,
    Cod2026JoystickEnv,
)

# ---------- 初始化 ----------
NUM_ENVS = 16
NUM_STEPS = 300

cfg = Cod2026JoystickCfg()
env = Cod2026JoystickEnv(cfg, num_envs=NUM_ENVS, backend_type="mujoco")

# ---------- 收集 ----------
obs_scaled_list = []
rew_list = []
term_count = 0

# 先全量 reset
env.reset(np.arange(NUM_ENVS))

# 策略：小幅度随机动作（模拟训练初期的策略）
ACTION_STD = 0.1

for step in range(NUM_STEPS):
    actions = (np.random.randn(NUM_ENVS, 6) * ACTION_STD).astype(np.float32)

    # 正确的 API：step 返回 NpEnvState
    state = env.step(actions)

    obs_arr = state.obs["obs"]
    obs_scaled_list.append(obs_arr.copy())
    rew_list.append(state.reward.copy())
    term_count += int(state.terminated.sum())

    if (step + 1) % 50 == 0:
        sys.stdout.write(
            f"  step {step+1}/{NUM_STEPS}  "
            f"term_rate={term_count/((step+1)*NUM_ENVS):.4f}  "
            f"reward_mean={np.mean(rew_list[-1000:]):.4f}\n"
        )

env.close()

# ---------- 统计 ----------
all_obs = np.concatenate(obs_scaled_list, axis=0)

obs_mean = all_obs.mean(axis=0)
obs_std = all_obs.std(axis=0)
obs_abs_mean = np.mean(np.abs(all_obs), axis=0)

labels = [
    "gyro_x", "gyro_y", "gyro_z",
    "up_x", "up_y", "up_z",
    "dof_pos_0", "dof_pos_1", "dof_pos_2", "dof_pos_3", "dof_pos_4", "dof_pos_5",
    "dof_vel_0", "dof_vel_1", "dof_vel_2", "dof_vel_3", "dof_vel_4(w)", "dof_vel_5(w)",
    "last_a_0", "last_a_1", "last_a_2", "last_a_3", "last_a_4", "last_a_5",
    "cmd_vx", "cmd_vy", "cmd_vyaw", "cmd_h",
    "rw_theta", "lw_theta",
    "rw_L", "lw_L",
]

print()
print("=" * 110)
print(f"观测各维度统计（NN 输入，被 _OBS_SCALE 缩放后）  共 {all_obs.shape[0]} 个样本")
print("=" * 110)
print(f"{'idx':>3}  {'label':<14} {'mean':>9} {'|mean|':>9} {'std':>9} {'min':>9} {'max':>9}  {'judge':>8}")
print("-" * 110)

issues = []
for i in range(len(obs_mean)):
    label = labels[i] if i < len(labels) else f"[{i}]"
    std_val = obs_std[i]

    if std_val < 0.05:
        judge = "TOO_SML"
        issues.append((i, label, "std太小，NN会忽略"))
    elif std_val > 5.0:
        judge = "TOO_BIG"
        issues.append((i, label, "std太大，会淹没其他信号"))
    else:
        judge = "OK"

    print(f"{i:>3}  {label:<14} {obs_mean[i]:>9.4f} {obs_abs_mean[i]:>9.4f} {std_val:>9.4f} "
          f"{all_obs[:, i].min():>9.4f} {all_obs[:, i].max():>9.4f}  {judge:>8}")

print()
print(f"整体 |obs| 均值 = {np.mean(np.abs(all_obs)):.4f}")
print(f"整体 std 均值   = {np.mean(obs_std):.4f}")
print(f"std 最小 = {obs_std.min():.4f}（{labels[obs_std.argmin()]}）")
print(f"std 最大 = {obs_std.max():.4f}（{labels[obs_std.argmax()]}）")
print()

if not issues:
    print("✅ 所有维度 std 都在合理范围")
else:
    print("⚠️  有问题的维度：")
    for idx, label, desc in issues:
        print(f"  [{idx:3d}] {label:<14} {desc}")