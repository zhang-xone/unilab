/**
 * @file    RLobs.c
 * @brief   RL 观测预处理：将传感器原始数据组装为与训练一致的观测向量
 * @note    观测维度 OBS_DIM = 31 (obs_vel = false)
 *          布局: [gyro(3), pitch(1), roll(1), commands(4), last_actions(6),
 *                  sin_yaw(1), cos_yaw(1),
 *                  dof_pos_err(4), dof_vel(6), vmc_theta(2), vmc_L0(2)]
 *
 *          --- 坐标系约定 (同 MuJoCo body frame) ---
 *          X: 前进方向 (Forward)
 *          Y: 左侧      (Left)
 *          Z: 上方      (Up)
 *          gyro:  body frame 角速度 [ωx, ωy, ωz] (rad/s)
 *          pitch: arcsin(gx), 抬头为正
 *          roll:  arcsin(-gy), 右倾为正
 *
 *          注意 - 部署时需确认实际 IMU 坐标系与以上约定一致：
 *          1. 平放时 gyro ≈ [0,0,0], gravity_body ≈ [0,0,-1]
 *          2. 前倾时 pitch > 0, gyro[1] > 0
 *          3. 左倾时 roll > 0,  gyro[0] > 0
 *
 *          --- 归一化 ---
 *          如果需要归一化，载入 obs_mean / obs_std 后调用 rl_obs_normalize()。
 *          或者在 ONNX 模型内部做归一化。
 */

#include <math.h>
#include <string.h>
#include "RLobs.h"

/* =============================================================================
 *  常量定义
 * ============================================================================= */

/* 观测维度 (obs_vel = false) */
#define OBS_DIM 31

/* 五连杆杆长 (m)，与 mujoco.yaml vmc 段一致 */
#define VMC_L1 0.215f
#define VMC_L2 0.254f
#define VMC_L3 0.254f
#define VMC_L4 0.215f
#define VMC_L5 0.0f

/* 轮子参数，与 base.py WHEEL_RADIUS / WHEELBASE 一致 */
#define WHEEL_RADIUS    0.06f
#define WHEELBASE       0.380f

/* 控制周期 (s)，与 mujoco.yaml ctrl_dt 一致 */
#define CTRL_DT 0.02f

/* PI 常数 */
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#define PI       M_PI
#define PI_2     (PI * 0.5f)
#define PI_6     (PI / 6.0f)
#define PI_5_6   (5.0f * PI / 6.0f)

/* dof_pos 的机械偏移量
 * phi1 = dof_pos[LR] + 5π/6
 * phi4 = dof_pos[LF] +  π/6
 *
 * 对应 RL 中 index 含义 (与 base.py JOINT_SENSOR_PREFIXES 一致):
 *   0 = left_front_joint  (LF, phi4_l)
 *   1 = right_front_joint (RF, phi4_r)
 *   2 = left_rear_joint   (LR, phi1_l)
 *   3 = right_rear_joint  (RR, phi1_r)
 */
#define DOF_OFFSET_LF  ( PI_6)      /* phi4_l = dof_pos[0] + π/6 */
#define DOF_OFFSET_RF  ( PI_6)      /* phi4_r = dof_pos[1] + π/6 */
#define DOF_OFFSET_LR  ( PI_5_6)    /* phi1_l = dof_pos[2] + 5π/6 */
#define DOF_OFFSET_RR  ( PI_5_6)    /* phi1_r = dof_pos[3] + 5π/6 */

/* 默认关节角度 (home keyframe 值) */
#define DEFAULT_DOF_POS_LF  0.0f
#define DEFAULT_DOF_POS_RF  0.0f
#define DEFAULT_DOF_POS_LR  0.0f
#define DEFAULT_DOF_POS_RR  0.0f

/*
 *  -------- 部署适配说明 --------
 *
 *  1. BalanceInfantry 结构体无需额外字段。
 *
 *  2. 轮子速度来源 (选其一):
 *     方案 A: 扩展 sensors_info 结构体，在 get_sensor_message() 中添加:
 *             bal->sensors_info.wheel_vel[0] = LeftMotor->measure.rotate_speed;   // 或 speed_rpm * π/30
 *             bal->sensors_info.wheel_vel[1] = RightMotor->measure.rotate_speed;
 *     方案 B: 用宏直接访问 Motor 的 speed 字段 (见下方 WHEEL_VEL_LEFT/RIGHT)
 */

/* 轮子速度宏 - 根据实际 DJIMotorInstance 的测量字段调整 */
#ifndef WHEEL_VEL_LEFT
#define WHEEL_VEL_LEFT(bal)  ((bal)->sensors_info.wheel_vel[0])
#endif
#ifndef WHEEL_VEL_RIGHT
#define WHEEL_VEL_RIGHT(bal) ((bal)->sensors_info.wheel_vel[1])
#endif

/** 安全 arcsin，裁剪到 [-1, 1] */
static inline float safe_asin(float x)
{
    if (x >  1.0f) x =  1.0f;
    if (x < -1.0f) x = -1.0f;
    return asinf(x);
}

/** 角度归一化到 [-π, π) */
static inline float normalize_angle(float angle)
{
    while (angle >=  PI) angle -= 2.0f * PI;
    while (angle < -PI) angle += 2.0f * PI;
    return angle;
}

/* =============================================================================
 *  重力投影 (gravity_body)
 *  ---------------------------------------------------------------------------
 *  从 base_link 的四元数 [qw, qx, qy, qz] 计算重力在 body frame 各轴的投影。
 *  世界重力方向: [0, 0, -1]
 *
 *  gx = 2*(x*z - w*y)
 *  gy = 2*(y*z + w*x)
 *  gz = 1 - 2*(x*x + y*y)
 *
 *  备选: 如果实际 IMU 不输出四元数，可以直接用加速度计归一化后的读数作为近似，
 *  或从欧拉角 (yaw/pitch/roll) 反算。
 * ============================================================================= */

/**
 * @brief 从四元数计算 body frame 重力投影
 * @param qw, qx, qy, qz  base_link 姿态四元数 (world → body)
 * @param gx, gy, gz      输出重力投影 [gx, gy, gz]
 */
static void quat_to_gravity_body(float qw, float qx, float qy, float qz,
                                  float *gx, float *gy, float *gz)
{
    *gx = 2.0f * (qx * qz - qw * qy);
    *gy = 2.0f * (qy * qz + qw * qx);
    *gz = 1.0f - 2.0f * (qx * qx + qy * qy);
}

/**
 * @brief 从欧拉角计算 body frame 重力投影 (备选方案)
 *
 * Euler 约定: ZYX (yaw-pitch-roll), 与大多数 IMU 一致。
 * pitch > 0 抬头, roll > 0 右倾。
 *
 * gx =  sin(pitch)
 * gy = -cos(pitch) * sin(roll)
 * gz = -cos(pitch) * cos(roll)
 *
 * @param pitch 俯仰角 (rad)
 * @param roll  横滚角 (rad)
 * @param gx, gy, gz  输出
 */
static void euler_to_gravity_body(float pitch, float roll,
                                   float *gx, float *gy, float *gz)
{
    float cp = cosf(pitch);
    *gx = sinf(pitch);
    *gy = -cp * sinf(roll);
    *gz = -cp * cosf(roll);
}

/* =============================================================================
 *  五连杆正运动学 (fk_fivebar)
 *  ---------------------------------------------------------------------------
 *  端口:
 *    phi1: 左后腿 (rear)  主动电机转角 (rad)
 *    phi4: 左前腿 (front) 主动电机转角 (rad)
 *
 *  输出:
 *    L0:   腿长 (五连杆端点 C 到基座中心距离, m)
 *    phi0: 腿与水平面夹角 (rad)
 *
 *  注意: 此函数与 C 代码中 MotionSolve() 一致，但如果 MotionSolve 已调用，
 *  可以直接使用 bal->L_left[0] 和 bal->phi_left[0]。
 * ============================================================================= */

/**
 * @brief 五连杆正运动学
 * @param phi1   左后主动电机转角 (rad)
 * @param phi4   左前主动电机转角 (rad)
 * @param L0     输出腿长 (m)
 * @param phi0   输出腿角 (rad)
 */
static void fk_fivebar(float phi1, float phi4, float *L0, float *phi0)
{
    float xB = VMC_L1 * cosf(phi1);
    float yB = VMC_L1 * sinf(phi1);
    float xD = VMC_L5 + VMC_L4 * cosf(phi4);
    float yD = VMC_L4 * sinf(phi4);

    float dx = xD - xB;
    float dy = yD - yB;
    float BD_sq = dx * dx + dy * dy;

    float A0 = 2.0f * VMC_L2 * dx;
    float b0 = 2.0f * VMC_L2 * dy;
    float C0 = VMC_L2 * VMC_L2 + BD_sq - VMC_L3 * VMC_L3;

    float disc = A0 * A0 + b0 * b0 - C0 * C0;
    if (disc < 0.0f) disc = 0.0f;
    float temp = b0 + sqrtf(disc);
    float phi2 = 2.0f * atan2f(temp, A0 + C0);

    float xC = xB + VMC_L2 * cosf(phi2);
    float yC = yB + VMC_L2 * sinf(phi2);
    float xC_shifted = xC - VMC_L5 / 2.0f;

    *L0   = sqrtf(xC_shifted * xC_shifted + yC * yC);
    *phi0 = atan2f(yC, xC_shifted);
}

/* =============================================================================
 *  RL 观测构建
 * ============================================================================= */

/**
 * @brief 组装 31 维 RL 观测向量
 *
 * @param obs            输出观测向量 (长度 OBS_DIM=31)
 * @param bal            机器人状态结构体指针
 * @param commands       遥控命令 [vx, dyaw, cmd_height, standing] (4D)
 *                       vx:       前进速度命令  (m/s)
 *                       dyaw:     偏航角速度命令 (rad/s)
 *                       cmd_height: 目标高度命令  (m)
 *                       standing: 站立模式标志   (0/1)
 * @param last_actions   上一步的 6 维 raw action [LF, RF, LR, RR, LW, RW]
 * @param target_yaw     目标偏航角 (rad)，由 dyaw 积分得到
 *
 * @note 调用前需确保:
 *       1. get_sensor_message() 已调用，IMU 和电机数据已更新
 *       2. MotionSolve() 已调用，L_left[0], phi_left[0] 等已计算
 *       3. getTheta() 已调用 (如需要 theta 相关值)
 */
void rl_build_observation(float obs[OBS_DIM],
                           const BalanceInfantry *bal,
                           const float commands[4],
                           const float last_actions[6],
                           float target_yaw)
{
    int idx = 0;

    /* ── 1. gyro (3D): body frame 角速度 ── */
    obs[idx++] = bal->GyroData[0];  /* ωx, roll rate */
    obs[idx++] = bal->GyroData[1];  /* ωy, pitch rate */
    obs[idx++] = bal->GyroData[2];  /* ωz, yaw rate */

    /* ── 2. pitch & roll: 从重力投影计算 ── */
    float gx, gy, gz;

    /*
     * 方案 A: 如果有四元数，直接算 (与仿真完全一致):
     *   quat_to_gravity_body(qw, qx, qy, qz, &gx, &gy, &gz);
     *
     * 方案 B: 从欧拉角反算 (如果 IMU 输出欧拉角):
     *   float pitch_raw = bal->Ins_Angle[2];
     *   float roll_raw  = bal->Ins_Angle[1];
     *   euler_to_gravity_body(pitch_raw, roll_raw, &gx, &gy, &gz);
     *
     * 方案 C: 直接用归一化后的加速度计读数 (机器人近静止时):
     *   float acc_norm = sqrtf(ax*ax + ay*ay + az*az);
     *   gx = ax / acc_norm; gy = ay / acc_norm; gz = az / acc_norm;
     */

    /* 默认使用欧拉角方案 (与 LQR 已有数据兼容) */
    {
        float pitch_raw = bal->Ins_Angle[2];
        float roll_raw  = bal->Ins_Angle[1];
        euler_to_gravity_body(pitch_raw, roll_raw, &gx, &gy, &gz);
    }

    float pitch = safe_asin(gx);
    float roll  = safe_asin(-gy);
    obs[idx++] = pitch;
    obs[idx++] = roll;

    /* ── 3. commands (4D): [vx, dyaw, cmd_height, standing] ── */
    for (int i = 0; i < 4; i++) {
        obs[idx++] = commands[i];
    }

    /* ── 4. last_actions (6D): [LF, RF, LR, RR, LW, RW] ── */
    for (int i = 0; i < 6; i++) {
        obs[idx++] = last_actions[i];
    }

    /* ── 5. yaw error (2D): sin & cos ── */
    float yaw_error = normalize_angle(target_yaw - bal->Ins_Angle[0]);
    obs[idx++] = sinf(yaw_error);
    obs[idx++] = cosf(yaw_error);

    /* ── 6. dof_pos_err (4D): 关节位置相对默认值的偏差 ── */
    /*
     * DOF 映射 (base.py JOINT_SENSOR_PREFIXES):
     *   [0] left_front_joint  (LF) → Motor[1] (phi4_l)
     *   [1] right_front_joint (RF) → Motor[2] (phi4_r)
     *   [2] left_rear_joint   (LR) → Motor[0] (phi1_l)
     *   [3] right_rear_joint  (RR) → Motor[3] (phi1_r)
     *
     * C 代码中:
     *   phi_left[4] = Motor[1].pos
     *   phi_right[4] = -Motor[2].pos
     *   phi_left[1] = Motor[0].pos + PI
     *   phi_right[1] = -Motor[3].pos + PI
     *
     * RL 中:
     *   dof_pos[LF] = phi4_l - π/6
     *   dof_pos[RF] = phi4_r - π/6
     *   dof_pos[LR] = phi1_l - 5π/6
     *   dof_pos[RR] = phi1_r - 5π/6
     */
    float dof_pos_lf, dof_pos_rf, dof_pos_lr, dof_pos_rr;

    /* LF: phi_left[4] = Motor[1].pos */
    dof_pos_lf = bal->DM8009_Motor[1].para.pos - DOF_OFFSET_LF;
    /* RF: phi_right[4] = -Motor[2].pos */
    dof_pos_rf = -bal->DM8009_Motor[2].para.pos - DOF_OFFSET_RF;
    /* LR: phi_left[1] = Motor[0].pos + PI */
    dof_pos_lr = bal->DM8009_Motor[0].para.pos + PI - DOF_OFFSET_LR;
    /* RR: phi_right[1] = -Motor[3].pos + PI */
    dof_pos_rr = -bal->DM8009_Motor[3].para.pos + PI - DOF_OFFSET_RR;

    obs[idx++] = dof_pos_lf - DEFAULT_DOF_POS_LF;
    obs[idx++] = dof_pos_rf - DEFAULT_DOF_POS_RF;
    obs[idx++] = dof_pos_lr - DEFAULT_DOF_POS_LR;
    obs[idx++] = dof_pos_rr - DEFAULT_DOF_POS_RR;

    /* ── 7. dof_vel (6D): 关节速度 ── */
    /*
     * 同 dof 顺序，加上轮子速度 (rad/s):
     *   [0] LF vel: Motor[1].vel     [3] RR vel: -Motor[3].vel
     *   [1] RF vel: -Motor[2].vel    [4] LW vel: 左轮 (由WHEEL_VEL_LEFT宏定义)
     *   [2] LR vel: Motor[0].vel     [5] RW vel: 右轮 (由WHEEL_VEL_RIGHT宏定义)
     */
    obs[idx++] = bal->DM8009_Motor[1].para.vel;     /* LF */
    obs[idx++] = -bal->DM8009_Motor[2].para.vel;    /* RF */
    obs[idx++] = bal->DM8009_Motor[0].para.vel;     /* LR */
    obs[idx++] = -bal->DM8009_Motor[3].para.vel;    /* RR */
    obs[idx++] = WHEEL_VEL_LEFT(bal);               /* LW */
    obs[idx++] = WHEEL_VEL_RIGHT(bal);              /* RW */

    /* ── 8. vmc_theta (2D): 腿相对世界垂直的角度 ── */
    /*
     * theta = π/2 - phi0 - pitch
     * 与 getTheta() 中计算一致:
     *   bal->theta_left = PI/2 - bal->phi_left[0] - bal->state_vector_l[4]
     *
     * 如果 MotionSolve() 和 getTheta() 已调用，可直接用:
     *   obs[idx++] = bal->theta_left;
     *   obs[idx++] = bal->theta_right;
     *
     * 下面用显式方式计算，与 Python fk_fivebar 路径一致:
     */
    {
        float dof_pos_lr_raw = bal->DM8009_Motor[0].para.pos + PI;
        float dof_pos_lf_raw = bal->DM8009_Motor[1].para.pos;
        float dof_pos_rr_raw = bal->DM8009_Motor[3].para.pos;  /* 负号在下面处理 */
        float dof_pos_rf_raw = bal->DM8009_Motor[2].para.pos;  /* 负号在下面处理 */

        float phi1_l = dof_pos_lr_raw;           /* = phi_left[1] */
        float phi4_l = dof_pos_lf_raw;           /* = phi_left[4] */
        float phi1_r = -dof_pos_rr_raw + PI;     /* = phi_right[1] */
        float phi4_r = -dof_pos_rf_raw;          /* = phi_right[4] */

        float L0_l, phi0_l, L0_r, phi0_r;
        fk_fivebar(phi1_l, phi4_l, &L0_l, &phi0_l);
        fk_fivebar(phi1_r, phi4_r, &L0_r, &phi0_r);

        float vmc_theta_l = PI_2 - phi0_l - pitch;
        float vmc_theta_r = PI_2 - phi0_r - pitch;

        obs[idx++] = vmc_theta_l;
        obs[idx++] = vmc_theta_r;
    }

    /* ── 10. vmc_L0 (2D): 腿长 ── */
    /*
     * 如果 MotionSolve() 已调用，可直接用:
     *   obs[idx++] = bal->L_left[0];
     *   obs[idx++] = bal->L_right[0];
     *
     * 注意: 上面的 fk_fivebar 调用后 L0_l/L0_r 是局部变量。
     * 在部署时，如果 MotionSolve 已执行过，直接用 bal->L_left[0] 更高效。
     */
    obs[idx++] = bal->L_left[0];   /* L0_l */
    obs[idx++] = bal->L_right[0];  /* L0_r */

    /* 最终检查 */
    /* idx 应为 OBS_DIM (31) */
}

/* =============================================================================
 *  归一化
 *  ---------------------------------------------------------------------------
 *  当模型训练时使用了 obs_normalization，部署侧需要做相同归一化:
 *    obs_normalized[i] = (obs[i] - obs_mean[i]) / (obs_std[i] + eps)
 *
 *  obs_mean / obs_std 在模型导出时一并产出。
 *  如果 ONNX 模型内部已包含归一化层，则不需要调用此函数。
 * ============================================================================= */

/**
 * @brief 对观测向量做均值-方差归一化
 *
 * @param obs       输入/输出观测向量 (in-place)
 * @param obs_mean  观测均值 (长度 OBS_DIM)
 * @param obs_std   观测标准差 (长度 OBS_DIM)
 * @param eps       防止除零的小量 (默认 1e-6)
 */
void rl_obs_normalize(float obs[OBS_DIM],
                       const float obs_mean[OBS_DIM],
                       const float obs_std[OBS_DIM],
                       float eps)
{
    for (int i = 0; i < OBS_DIM; i++) {
        obs[i] = (obs[i] - obs_mean[i]) / (obs_std[i] + eps);
    }
}

/* =============================================================================
 *  历史观测缓冲
 *  ---------------------------------------------------------------------------
 *  训练时使用了 obs_history_length=5，策略的 encoder 接收 5 帧历史:
 *    encoder_input = [obs_{t-4}, obs_{t-3}, obs_{t-2}, obs_{t-1}, obs_t]
 *  即 5 × OBS_DIM = 155 维。
 *
 *  如果 ONNX 模型期望 155 维输入，需维护环形缓冲区。
 * ============================================================================= */

#define OBS_HISTORY_LENGTH 5
#define ENC_INPUT_DIM    (OBS_DIM * OBS_HISTORY_LENGTH)  /* 155 */

/**
 * @brief 将当前观测推入历史缓冲，返回拼接后的 encoder 输入
 *
 * @param enc_input     输出 encoder 输入 (155 维)
 * @param history_buf   历史环形缓冲区 (OBS_DIM * OBS_HISTORY_LENGTH)
 * @param history_pos   当前写入位置 (调用方负责维护，每次 push 后 +1)
 * @param obs           当前帧观测 (31 维)
 */
void rl_push_history(float enc_input[ENC_INPUT_DIM],
                      float history_buf[ENC_INPUT_DIM],
                      int *history_pos,
                      const float obs[OBS_DIM])
{
    /* 写入当前帧到 buffer 对应位置 */
    int offset = (*history_pos % OBS_HISTORY_LENGTH) * OBS_DIM;
    memcpy(&history_buf[offset], obs, OBS_DIM * sizeof(float));

    /* 拼接: 按时间顺序 [oldest ... newest] */
    int start = (*history_pos + 1) % OBS_HISTORY_LENGTH;
    for (int i = 0; i < OBS_HISTORY_LENGTH; i++) {
        int src_offset = ((start + i) % OBS_HISTORY_LENGTH) * OBS_DIM;
        int dst_offset = i * OBS_DIM;
        memcpy(&enc_input[dst_offset], &history_buf[src_offset],
               OBS_DIM * sizeof(float));
    }

    *history_pos += 1;
}

/* =============================================================================
 *  RL Action 后处理管线
 *  ---------------------------------------------------------------------------
 *  ONNX → raw_action(6D) → clip → smooth → delay_FIFO → target → torque+VMC
 *
 *  步骤:
 *  1. clip:      raw_action → clipped_action (leg_clip=100.0, wheel_clip=1.0)
 *  2. smooth:    EMA 低通滤波 filtered = (1-smooth)*raw + smooth*old
 *  3. last_act:  保存 stop_gradient 版本作下一帧观测
 *  4. delay:     FIFO 队列, 延迟 ACTION_DELAY=2 个控制步
 *  5. target:    leg: target = action*scale + default, clip to joint_limits
 *                wheel: target_vel = action * wheel_scale
 *  6. torque:    leg: kp*(target-pos) - kd*vel
 *                wheel: wheel_kd*(target_vel - vel)
 *  7. VMC:       叠加 VMC 重力前馈力矩 (leg only)
 *
 *  raw_action 6D: [LF, RF, LR, RR, LW, RW]
 * ============================================================================= */

/* ── Action 超参数 (与 mujoco.yaml control_config 一致) ── */
#define ACTION_SCALE_JOINT 0.5f
#define ACTION_SCALE_WHEEL 10.0f

/* PD 参数 (per-joint) */
#define LEG_KP_LF  40.0f
#define LEG_KP_RF  40.0f
#define LEG_KP_LR  40.0f
#define LEG_KP_RR  40.0f
#define LEG_KD_LF  2.0f
#define LEG_KD_RF  2.0f
#define LEG_KD_LR  2.0f
#define LEG_KD_RR  2.0f
#define WHEEL_KD_LW 0.1f
#define WHEEL_KD_RW 0.1f

/* Action 裁剪限幅 */
#define LEG_CLIP   100.0f
#define WHEEL_CLIP 1.0f
#define LEG_CLIP_INDICES_COUNT  4
#define WHEEL_CLIP_INDICES_COUNT 2
static const int LEG_CLIP_INDICES[LEG_CLIP_INDICES_COUNT] = {0, 1, 3, 4};
static const int WHEEL_CLIP_INDICES[WHEEL_CLIP_INDICES_COUNT] = {2, 5};

/* Action 延迟 FIFO 深度 (训练中 ACTION_DELAY=2, buffer 大小=3) */
#define ACTION_DELAY 2
#define ACTION_FIFO_SIZE (ACTION_DELAY + 1)  /* 3 */

/* VMC 重力前馈力 (N) */
#define VMC_FEEDFORWARD_FORCE 80.0f

/* 关节限位 (rad)，与 base.py JOINT_LIMITS 一致 */
#define JOINT_LIMIT_LF_LO -0.7f
#define JOINT_LIMIT_LF_HI  0.7f
#define JOINT_LIMIT_RF_LO -0.7f
#define JOINT_LIMIT_RF_HI  0.7f
#define JOINT_LIMIT_LR_LO -1.6f
#define JOINT_LIMIT_LR_HI  1.6f
#define JOINT_LIMIT_RR_LO -1.6f
#define JOINT_LIMIT_RR_HI  1.6f

/* ────────────────────────────────────────────────────────────────────
 *  rl_action_clip
 *  ────────────────────────────────────────────────────────────────────
 *  对 raw_action 做安全裁剪。
 *  LEG_CLIP=100.0 几乎不裁剪，WHEEL_CLIP=1.0 有效限制轮子命令。
 * ──────────────────────────────────────────────────────────────────── */
static void rl_action_clip(float action[6])
{
    for (int i = 0; i < LEG_CLIP_INDICES_COUNT; i++) {
        int idx = LEG_CLIP_INDICES[i];
        if (action[idx] >  LEG_CLIP) action[idx] =  LEG_CLIP;
        if (action[idx] < -LEG_CLIP) action[idx] = -LEG_CLIP;
    }
    for (int i = 0; i < WHEEL_CLIP_INDICES_COUNT; i++) {
        int idx = WHEEL_CLIP_INDICES[i];
        if (action[idx] >  WHEEL_CLIP) action[idx] =  WHEEL_CLIP;
        if (action[idx] < -WHEEL_CLIP) action[idx] = -WHEEL_CLIP;
    }
}

/* ────────────────────────────────────────────────────────────────────
 *  rl_action_smooth
 *  ────────────────────────────────────────────────────────────────────
 *  EMA 低通滤波: filtered = (1-smooth) * raw + smooth * old_filtered
 *  smooth=0: 不过滤,  smooth=0.8: 强平滑 (默认)
 * ──────────────────────────────────────────────────────────────────── */
static void rl_action_smooth(float filtered[6],
                              const float raw[6],
                              float smooth)
{
    for (int i = 0; i < 6; i++) {
        filtered[i] = (1.0f - smooth) * raw[i] + smooth * filtered[i];
    }
}

/* ────────────────────────────────────────────────────────────────────
 *  rl_action_delay_fifo_push
 *  ────────────────────────────────────────────────────────────────────
 *  FIFO 动作延迟队列: 每控制步 push, pop 延迟 ACTION_DELAY 步的 action.
 *
 *  fifo[]: 环形缓冲 (调用方维护, 大小 ACTION_FIFO_SIZE * 6)
 *  pos:    FIFO 写入位置 (调用方维护)
 *  in:     当前步的 filtered_action (6D)
 *  out:    延迟后的 delayed_action (6D)
 * ──────────────────────────────────────────────────────────────────── */
static void rl_action_delay_fifo_push(float fifo[ACTION_FIFO_SIZE * 6],
                                       int *pos,
                                       const float in[6],
                                       float out[6])
{
    int wp = *pos % ACTION_FIFO_SIZE;

    /* 写入当前帧 */
    for (int i = 0; i < 6; i++) {
        fifo[wp * 6 + i] = in[i];
    }

    /* 读取延迟 ACTION_DELAY 步的帧 */
    int rp = (*pos - ACTION_DELAY + ACTION_FIFO_SIZE) % ACTION_FIFO_SIZE;
    for (int i = 0; i < 6; i++) {
        out[i] = fifo[rp * 6 + i];
    }

    *pos += 1;
}

/* ────────────────────────────────────────────────────────────────────
 *  rl_compute_vmc_feedforward
 *  ────────────────────────────────────────────────────────────────────
 *  VMC 重力前馈力矩 (与 compute_vmc_feedforward / _compute_vmc_torques 一致)
 *
 *  纯前馈: F = feedforward_force / cos(theta) * ramp, T = 0
 *  tau = J^T * [F, T] 通过雅可比映射到关节力矩
 *
 *  vmc_torques[6]: [tau_lf, tau_rf, tau_lr, tau_rr, 0, 0]
 * ──────────────────────────────────────────────────────────────────── */
void rl_compute_vmc_feedforward(float vmc_torques[6],
                                 const float dof_pos[4],
                                 float pitch)
{
    /* ── 左腿五连杆 ── */
    float phi1_l = dof_pos[2] + PI_5_6;
    float phi4_l = dof_pos[0] + PI_6;

    float L0_l, phi0_l, phi2_l, phi3_l, sigma1_l;
    {
        float xB = VMC_L1 * cosf(phi1_l);
        float yB = VMC_L1 * sinf(phi1_l);
        float xD = VMC_L5 + VMC_L4 * cosf(phi4_l);
        float yD = VMC_L4 * sinf(phi4_l);

        float dx = xD - xB, dy = yD - yB;
        float BD_sq = dx * dx + dy * dy;
        float A0 = 2.0f * VMC_L2 * dx;
        float b0 = 2.0f * VMC_L2 * dy;
        float C0 = VMC_L2 * VMC_L2 + BD_sq - VMC_L3 * VMC_L3;

        float disc = A0 * A0 + b0 * b0 - C0 * C0;
        if (disc < 0.0f) disc = 0.0f;
        float temp = b0 + sqrtf(disc);
        phi2_l = 2.0f * atan2f(temp, A0 + C0);

        float xC = xB + VMC_L2 * cosf(phi2_l);
        float yC = yB + VMC_L2 * sinf(phi2_l);
        L0_l = sqrtf((xC - VMC_L5 / 2.0f) * (xC - VMC_L5 / 2.0f) + yC * yC);
        phi0_l = atan2f(yC, xC - VMC_L5 / 2.0f);

        phi3_l = atan2f(yB - yD + VMC_L2 * sinf(phi2_l),
                        xB - xD + VMC_L2 * cosf(phi2_l));
        sigma1_l = sinf(phi3_l - phi2_l);
    }

    /* ── 右腿五连杆 ── */
    float phi1_r = dof_pos[3] + PI_5_6;
    float phi4_r = dof_pos[1] + PI_6;

    float L0_r, phi0_r, phi2_r, phi3_r, sigma1_r;
    {
        float xB = VMC_L1 * cosf(phi1_r);
        float yB = VMC_L1 * sinf(phi1_r);
        float xD = VMC_L5 + VMC_L4 * cosf(phi4_r);
        float yD = VMC_L4 * sinf(phi4_r);

        float dx = xD - xB, dy = yD - yB;
        float BD_sq = dx * dx + dy * dy;
        float A0 = 2.0f * VMC_L2 * dx;
        float b0 = 2.0f * VMC_L2 * dy;
        float C0 = VMC_L2 * VMC_L2 + BD_sq - VMC_L3 * VMC_L3;

        float disc = A0 * A0 + b0 * b0 - C0 * C0;
        if (disc < 0.0f) disc = 0.0f;
        float temp = b0 + sqrtf(disc);
        phi2_r = 2.0f * atan2f(temp, A0 + C0);

        float xC = xB + VMC_L2 * cosf(phi2_r);
        float yC = yB + VMC_L2 * sinf(phi2_r);
        L0_r = sqrtf((xC - VMC_L5 / 2.0f) * (xC - VMC_L5 / 2.0f) + yC * yC);
        phi0_r = atan2f(yC, xC - VMC_L5 / 2.0f);

        phi3_r = atan2f(yB - yD + VMC_L2 * sinf(phi2_r),
                        xB - xD + VMC_L2 * cosf(phi2_r));
        sigma1_r = sinf(phi3_r - phi2_r);
    }

    /* ── theta & ramp ── */
    float theta_l = PI_2 - phi0_l - pitch;
    float theta_r = PI_2 - phi0_r - pitch;

    float theta_abs_l = fabsf(theta_l);
    float theta_abs_r = fabsf(theta_r);
    float ramp_l = ((PI / 3.0f) - theta_abs_l) / (PI / 6.0f);
    float ramp_r = ((PI / 3.0f) - theta_abs_r) / (PI / 6.0f);
    if (ramp_l < 0.0f) ramp_l = 0.0f; if (ramp_l > 1.0f) ramp_l = 1.0f;
    if (ramp_r < 0.0f) ramp_r = 0.0f; if (ramp_r > 1.0f) ramp_r = 1.0f;

    float cos_theta_l = cosf(theta_l);
    float cos_theta_r = cosf(theta_r);
    if (cos_theta_l < 0.0707f) cos_theta_l = 0.0707f;  /* cos(1.5) ≈ 0.0707 */
    if (cos_theta_r < 0.0707f) cos_theta_r = 0.0707f;

    float ff_l = VMC_FEEDFORWARD_FORCE / cos_theta_l * ramp_l;
    float ff_r = VMC_FEEDFORWARD_FORCE / cos_theta_r * ramp_r;

    /* ── 雅可比转置: J^T ⨉ [F, T=0] ── */
    float sigma2_l = sinf(phi3_l - phi4_l);
    float sigma3_l = sinf(phi1_l - phi2_l);
    float sigma2_r = sinf(phi3_r - phi4_r);
    float sigma3_r = sinf(phi1_r - phi2_r);

    float eps = 1e-8f;
    float inv_s1_l = 1.0f / (sigma1_l + eps);
    float inv_s1_r = 1.0f / (sigma1_r + eps);

    float J00_l = VMC_L1 * sinf(phi0_l - phi3_l) * sigma3_l * inv_s1_l;
    float J10_l = VMC_L4 * sinf(phi0_l - phi2_l) * sigma2_l * inv_s1_l;
    float J00_r = VMC_L1 * sinf(phi0_r - phi3_r) * sigma3_r * inv_s1_r;
    float J10_r = VMC_L4 * sinf(phi0_r - phi2_r) * sigma2_r * inv_s1_r;

    float tau1_l = J00_l * ff_l;
    float tau4_l = J10_l * ff_l;
    float tau1_r = J00_r * ff_r;
    float tau4_r = J10_r * ff_r;

    /* 写到 vmc_torques: [LF=tau4_l, RF=tau4_r, LR=tau1_l, RR=tau1_r, 0, 0] */
    vmc_torques[0] = tau4_l;
    vmc_torques[1] = tau4_r;
    vmc_torques[2] = tau1_l;
    vmc_torques[3] = tau1_r;
    vmc_torques[4] = 0.0f;
    vmc_torques[5] = 0.0f;

    /* 防爆: NaN→0, ±inf→±40 */
    for (int i = 0; i < 6; i++) {
        if (isnan(vmc_torques[i])) vmc_torques[i] = 0.0f;
        if (vmc_torques[i] >  40.0f) vmc_torques[i] =  40.0f;
        if (vmc_torques[i] < -40.0f) vmc_torques[i] = -40.0f;
    }
}

/* ────────────────────────────────────────────────────────────────────
 *  rl_raw_action_to_torque
 *  ────────────────────────────────────────────────────────────────────
 *  完整转换: 经过 clip/smooth/delay 后的 action → 6D 力矩
 *
 *  leg:    target = action_i * action_scale_joint + default_dof_pos_i
 *          target = clip(target, joint_limit)
 *          torque_i = kp_i * (target - dof_pos_i) - kd_i * dof_vel_i
 *  wheel:  target_vel = action_i * action_scale_wheel
 *          torque_i = wheel_kd_i * (target_vel - dof_vel_i)
 *
 *  最后叠加 VMC 前馈力矩 (leg only, tai[:4] += vmc[:4])
 * ──────────────────────────────────────────────────────────────────── */
void rl_raw_action_to_torque(float torques[6],
                              const float action[6],
                              const float dof_pos[6],
                              const float dof_vel[6],
                              const float default_dof_pos[4],
                              const float vmc_torques[6])
{
    static const float kp[4] = {LEG_KP_LF, LEG_KP_RF, LEG_KP_LR, LEG_KP_RR};
    static const float kd[4] = {LEG_KD_LF, LEG_KD_RF, LEG_KD_LR, LEG_KD_RR};
    static const float joint_lo[4] = {JOINT_LIMIT_LF_LO, JOINT_LIMIT_RF_LO,
                                       JOINT_LIMIT_LR_LO, JOINT_LIMIT_RR_LO};
    static const float joint_hi[4] = {JOINT_LIMIT_LF_HI, JOINT_LIMIT_RF_HI,
                                       JOINT_LIMIT_LR_HI, JOINT_LIMIT_RR_HI};

    /* 腿关节 (4 维) */
    for (int i = 0; i < 4; i++) {
        float target = action[i] * ACTION_SCALE_JOINT + default_dof_pos[i];
        if (target > joint_hi[i]) target = joint_hi[i];
        if (target < joint_lo[i]) target = joint_lo[i];
        torques[i] = kp[i] * (target - dof_pos[i]) - kd[i] * dof_vel[i];
    }

    /* 轮子 (2 维): 速度目标, 纯阻尼 */
    float lw_target = action[4] * ACTION_SCALE_WHEEL;
    float rw_target = action[5] * ACTION_SCALE_WHEEL;
    torques[4] = WHEEL_KD_LW * (lw_target - dof_vel[4]);
    torques[5] = WHEEL_KD_RW * (rw_target - dof_vel[5]);

    /* 叠加 VMC 重力前馈 (仅腿) */
    if (vmc_torques != NULL) {
        for (int i = 0; i < 4; i++) {
            torques[i] += vmc_torques[i];
        }
    }
}

/* ────────────────────────────────────────────────────────────────────
 *  rl_process_action  (一站式 action 后处理)
 *  ────────────────────────────────────────────────────────────────────
 *  组合 clip → smooth → delay → torque + VMC 的完整管线。
 *
 *  @param torques          输出力矩 [6]
 *  @param raw_action       ONNX 原始输出 [6]
 *  @param filtered_action  滤波状态 [6] (调用方维护, 首次调用前清零)
 *  @param action_fifo      FIFO 缓冲 [ACTION_FIFO_SIZE*6] (调用方维护)
 *  @param fifo_pos         FIFO 写入位置的指针 (调用方维护)
 *  @param dof_pos          当前关节位置 [6]
 *  @param dof_vel          当前关节速度 [6]
 *  @param default_dof_pos  默认关节位置 [4]
 *  @param vmc_torques      VMC 前馈力矩 [6] (可为 NULL 跳过)
 *  @param smooth           EMA 平滑系数 (0=不过滤, 0.8=强平滑)
 *  @param last_actions_out 输出: 供下一帧观测使用的 last_actions [6]
 * ──────────────────────────────────────────────────────────────────── */
void rl_process_action(float torques[6],
                        const float raw_action[6],
                        float filtered_action[6],
                        float action_fifo[ACTION_FIFO_SIZE * 6],
                        int *fifo_pos,
                        const float dof_pos[6],
                        const float dof_vel[6],
                        const float default_dof_pos[4],
                        const float vmc_torques[6],
                        float smooth,
                        float last_actions_out[6])
{
    float clipped[6];
    float delayed[6];

    /* Step 1: Clip */
    for (int i = 0; i < 6; i++) clipped[i] = raw_action[i];
    rl_action_clip(clipped);

    /* Step 2: EMA Smooth */
    rl_action_smooth(filtered_action, clipped, smooth);

    /* Step 3: 保存 last_actions (stop_gradient, 即 filtered 的当前值) */
    for (int i = 0; i < 6; i++) last_actions_out[i] = filtered_action[i];

    /* Step 4: Delay FIFO */
    rl_action_delay_fifo_push(action_fifo, fifo_pos, filtered_action, delayed);

    /* Step 5: Decode + PD + VMC → torque */
    rl_raw_action_to_torque(torques, delayed, dof_pos, dof_vel,
                             default_dof_pos, vmc_torques);
}

/* =============================================================================
 *  完整的 RL 控制步 (调用示例)
 *  ---------------------------------------------------------------------------
 *
 *  float obs[OBS_DIM];
 *  float enc_input[ENC_INPUT_DIM];
 *  float history_buf[ENC_INPUT_DIM] = {0};
 *  int history_pos = 0;
 *
 *  float filtered_action[6] = {0};
 *  float action_fifo[ACTION_FIFO_SIZE * 6] = {0};
 *  int fifo_pos = 0;
 *  float last_actions[6] = {0};
 *
 *  float commands[4] = {vx, dyaw, cmd_height, standing};
 *  float target_yaw = 0.0f;
 *  float vx = 0.0f;  // 底盘实际速度 (m/s)
 *  float default_dof_pos[4] = {0, 0, 0, 0};
 *
 *  while (1) {
 *      // 1. 更新传感器
 *      get_sensor_message(bal);
 *      MotionSolve(bal);
 *
 *      // 2. 组装观测
 *      rl_build_observation(obs, bal, commands, last_actions, target_yaw);
 *
 *      // 3. 归一化 (如果模型需要)
 *      // rl_obs_normalize(obs, obs_mean, obs_std, 1e-6f);
 *
 *      // 4. 推入历史缓冲 → 155D encoder 输入
 *      rl_push_history(enc_input, history_buf, &history_pos, obs);
 *
 *      // 5. ONNX 推理: enc_input(155D) + obs(31D) → raw_action(6D)
 *      float raw_action[6];
 *      // onnx_run(enc_input, obs, raw_action);
 *
 *      // 6. 一站式 action 后处理
 *      float torques[6];
 *      float vmc[6] = {0};
 *      rl_compute_vmc_feedforward(vmc, dof_pos, pitch);
 *      rl_process_action(torques, raw_action, filtered_action,
 *                         action_fifo, &fifo_pos,
 *                         dof_pos, dof_vel, default_dof_pos, vmc,
 *                         0.8f, last_actions);
 *
 *      // 8. 发送力矩
 *      send_motor_torques(torques);
 *
 *      // 9. 更新命令 (target_yaw += dyaw * CTRL_DT)
 *      target_yaw += commands[1] * CTRL_DT;
 *  }
 * ============================================================================= */