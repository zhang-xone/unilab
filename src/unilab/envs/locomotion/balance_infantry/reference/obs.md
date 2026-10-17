# 观测向量 (Observation Vector) 构成说明

> 唯一权威来源：`joystick.py` `_compute_obs()` + `_compute_gravity_body()` + `_update_vmc_state()`

---

## 坐标系约定

右手坐标系，同 MuJoCo body frame：

| 轴 | 方向 | 右手法则旋转 |
|---|---|---|
| **X** | 前进 (Forward) | ωx > 0: 身体绕 X 轴右滚（右侧下沉） |
| **Y** | 左侧 (Left) | ωy > 0: 身体绕 Y 轴低头（nose down） |
| **Z** | 上方 (Up) | ωz > 0: 身体绕 Z 轴左转（逆时针俯视） |

gravity_body（世界 Z 轴在 body 系的投影）：

```
重力在 body 系的表达（实际是"向上"向量，非重力方向）
gx = 2*(qx*qz - qw*qy)     X 分量
gy = 2*(qy*qz + qw*qx)     Y 分量
gz = 1 - 2*(qx² + qy²)     Z 分量

平放时: [0, 0, 1]    (向上)
前倾时: gx > 0        (向上向量偏 forward)
左倾时: gy < 0        (向上向量偏 right)
```

---

## 单帧观测 (obs) — 33 维

obs 构造来源：`_compute_obs()` → `obs_cols` 拼接顺序

```
索引    维度  名称              来源 / 公式 / 方向
────────────────────────────────────────────────────────────────────────────
[0-2]    3   gyro              body frame 角速度 [ωx, ωy, ωz] (rad/s)
                                   ωx: 绕 X 轴，右手法则，正=右滚
                                   ωy: 绕 Y 轴，右手法则，正=低头
                                   ωz: 绕 Z 轴，右手法则，正=左转

[3]      1   pitch             = -arcsin(gx)
                                   gx = gravity_body_x = 2*(qx*qz - qw*qy)
                                   前倾时 gx > 0 → pitch < 0
                                   前下为负，后上为正

[4]      1   roll              = arcsin(-gy)
                                   gy = gravity_body_y = 2*(qy*qz + qw*qx)
                                   左倾时 gy < 0 → -gy > 0 → roll > 0
                                   右倾时 gy > 0 → -gy < 0 → roll < 0
                                   右下为负，左上为正

[5]      1   cmd_vel           commands[:, 0]  前进速度命令 (m/s)，前为正
[6]      1   cmd_yaw_rate      commands[:, 2]  偏航角速度命令 (rad/s)，左转为正
                                   训练: 随机采样
                                   部署: 云台跟随模式下 = kp * (gimbal_yaw - chassis_yaw)
[7]      1   cmd_height        commands[:, 3]  目标高度命令 (m)
[8]      1   cmd_standing      commands[:, 4]  站立模式标志 (固定约 0.05)

[9-14]   6   last_actions      上一步策略网络的原始 6 维输出 (action ∈ [-1, 1])
                                   [9]  LF (left_front)
                                   [10] RF (right_front)
                                   [11] LR (left_rear)
                                   [12] RR (right_rear)
                                   [13] LW (left_wheel)
                                   [14] RW (right_wheel)

[15]     1   wheel_pos_err_l   左轮位置积分误差 (rad)，前进方向为正
[16]     1   wheel_pos_err_r   右轮位置积分误差 (rad)，前进方向为正

[17]     1   sin_yaw           sin(yaw_error)
[18]     1   cos_yaw           cos(yaw_error)
                                   yaw_error = target_yaw - base_yaw (wrapped to ±π)
                                   target_yaw += cmd_yaw_rate * dt
                                   cmd_yaw_rate > 0 → target_yaw 增大 → 需左转追赶
                                   部署云台模式: yaw_error = gimbal_yaw - chassis_yaw (直接差，不复用积分)

[19-22]  4   dof_pos_err       = dof_pos[:4] - default_dof_pos[:4]
                                   default_dof_pos = [0, 0, 0, 0] (rad), 来自 home keyframe
                                   dof_pos=0 时: φ₄=30°(上翘), φ₁=150°(上翘), 虚拟腿竖直向下
                                   L/R 正方向一致: 正=关节角度>0 → theta↓ → 腿后摆
                                                    负=关节角度<0 → theta↑ → 腿前摆
                                   [19] LF  [20] RF  [21] LR  [22] RR
                                   实测: theta≈-0.38 后摆时, dof_err 全为正 (LF≈+0.90, RF≈+0.71, LR≈+0.09, RR≈+0.05)

[23-28]  6   dof_vel           关节速度 (rad/s)，方向与 dof_pos 一致，正=关节往正方向转动
                                   仿真内: raw qvel = d(qpos)/dt, 四腿 L/R 同向
                                   实机 C 端: RF/RR 有符号翻转 (RLobs.c: -Motor[2/3].vel)
                                   [23] LF  [24] RF  [25] LR  [26] RR
                                   [27] LW (左轮)   [28] RW (右轮)

[29]     1   theta_l           左腿 VMC 摆角 (rad)
                                   = π/2 - phi0_l - pitch
                                   正=腿前摆 (leg ahead of body, 世界系往前倾)
                                   负=腿后摆 (leg behind body, 世界系往后仰)

[30]     1   theta_r           右腿 VMC 摆角 (rad)
                                   = π/2 - phi0_r - pitch
                                   正=腿前摆 (leg ahead of body, 世界系往前倾)
                                   负=腿后摆 (leg behind body, 世界系往后仰)

[31]     1   L0_l              左腿五连杆虚拟腿长 (m)，动态值，来自 fk_fivebar
[32]     1   L0_r              右腿五连杆虚拟腿长 (m)，动态值，来自 fk_fivebar
```

---

## DOF 关节映射

来源：`base.py` `JOINT_SENSOR_PREFIXES`，与 `chassis.xml` actuator 顺序一致：

| 索引 | 传感器名 | 五连杆角色 | 角度转换 | 默认 φ (dof_pos=0) |
|---|---|---|---|---|
| 0 | `left_front_joint` | φ₄_l (LF) | `phi4_l = dof_pos[0] + π/6` | 30° |
| 1 | `right_front_joint` | φ₄_r (RF) | `phi4_r = dof_pos[1] + π/6` | 30° |
| 2 | `left_rear_joint` | φ₁_l (LR) | `phi1_l = dof_pos[2] + 5π/6` | 150° |
| 3 | `right_rear_joint` | φ₁_r (RR) | `phi1_r = dof_pos[3] + 5π/6` | 150° |
| 4 | `left_wheel_joint` | 左轮 (LW) | - | - |
| 5 | `right_wheel_joint` | 右轮 (RW) | - | - |

逆推: `dof_pos_err[LF] = φ₄ - π/6`, `dof_pos_err[LR] = φ₁ - 5π/6`

dof_pos 和 dof_vel 以 MuJoCo 关节正方向为准，不做符号翻转（部署时按实际电机方向匹配）。

---

## Encoder 输入 (obs_history_norm) — 165 维

```python
# encoder 输入 = 5 帧历史 obs 拼接
obs_history = concat([obs_t4, obs_t3, obs_t2, obs_t1, obs_t0])  # 5×33=165

# 归一化 (部署用固定参数，训练用 running norm)
oh_norm = (obs_history - enc_mean) / (enc_std + NORM_EPS)
```

> `obs_vel=True` 时 obs 变为 36 维 (33+3)，encoder 输入变为 180 维。
> 默认 `obs_vel=False`，即 165 维。


## Critic 观测

Critic 输入 = actor obs (33 维) + height_scan 等特权信息，共 `70 + height_scan_dim` 维。
Critic 只在训练时使用，部署不需要。

---

## 五连杆运动学 (VMC theta & L0)

来源：`joystick.py` `_update_vmc_state()` + `_compute_state()`

五连杆参数：
```
L1=0.215, L2=0.254, L3=0.254, L4=0.215, L5=0.0
```

完整流程：
```
# 1. 电机 → 五连杆角度
phi1_l = dof_pos[2] + 5π/6     左腿后大腿 (LR→φ₁)
phi4_l = dof_pos[0] + π/6      左腿前大腿 (LF→φ₄)
phi1_r = dof_pos[3] + 5π/6     右腿后大腿 (RR→φ₁)
phi4_r = dof_pos[1] + π/6      右腿前大腿 (RF→φ₄)

# 2. 五连杆正运动学
L0_l, phi0_l, phi2_l, phi3_l, sigma1_l = fk_fivebar(phi1_l, phi4_l)
L0_r, phi0_r, phi2_r, phi3_r, sigma1_r = fk_fivebar(phi1_r, phi4_r)

# 3. VMC 观测
pitch = -arcsin(gravity_body_x)
theta_l = π/2 - phi0_l - pitch    (VMC 摆角，世界系相对垂直)
theta_r = π/2 - phi0_r - pitch
L0_l   = fk 输出腿长              (动态值，非固定常量)
L0_r   = fk 输出腿长
```

---

## 输出动作 (action) — 6 维

策略网络输出 6 维 `action ∈ [-1, 1]`，经 PD 控制器转换为关节力矩：

| 索引 | 关节 | scale | PD 目标 (rad) |
|---|---|---|---|
| 0 | LF | 0.5 | `default_dof_pos[0] + action[0] * 0.5` |
| 1 | RF | 0.5 | `default_dof_pos[1] + action[1] * 0.5` |
| 2 | LR | 0.5 | `default_dof_pos[2] + action[2] * 0.5` |
| 3 | RR | 0.5 | `default_dof_pos[3] + action[3] * 0.5` |
| 4 | LW | 10.0 | `action[4] * 10.0` (速度控制) |
| 5 | RW | 10.0 | `action[5] * 10.0` (速度控制) |

---

## Actor 归一化

训练时 `obs_manual_norm=False`（默认），`obs_flat` 原始值直接进网络，
网络内部用 running statistics 做自适应归一化（RL 框架自动处理）。

部署时（MCU 推理）无 running norm，改用离线统计的固定参数，
来源（自动生成，与训练网络导出一致）：

```
norm_params.h:
  obs_mean[33], obs_std[33]     # 单帧 obs 归一化
  enc_mean[165], enc_std[165]   # encoder 输入 (5 帧历史) 归一化
  NORM_EPS = 1e-2

推理流程:
  oh = (obs_history - enc_mean) / (enc_std + NORM_EPS)   # 165 维
  o  = (obs - obs_mean) / (obs_std + NORM_EPS)           # 33 维
  action = ONNX([o, encoder(oh)])                        # 36 → 6
```

> 注意：训练时 obs 中的 gyro 和 dof_vel 经过了 `_apply_obs_noise()`
> 添加高斯噪声 + drift bias。部署时跳过噪声，直接用原始传感器值。

---

## 部署校验清单

| 检查项 | 方法 |
|---|---|
| gyro 方向 | 右手系验证：右滚时 ωx > 0，低头时 ωy > 0，左转时 ωz > 0 |
| pitch 符号 | 前下为负，后上为正 |
| roll 符号 | 右下为负，左上为正 |
| cmd_yaw_rate 方向 | > 0 时机器人左转 |
| dof_pos 符号 | 以 MuJoCo 正方向为准，部署时按实际电机校准 |
| theta 计算 | 与 joystick.py `_compute_state()` 输出对比，误差 < 1e-4 |
| L0 计算 | 与 fk_fivebar 输出对比，为动态值（约 0.28） |
| 归一化 | 归一化后 obs 各维度在 [-3, 3] 内 |
| ONNX 输出 | 与相同输入下的 PyTorch 模型输出对比，误差 < 1e-5 |