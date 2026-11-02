// 自动生成: 控制参数 (与 trained_params.py 一致)
// 来源: trained_params.py → run=2026-10-30_09-18-48_mujoco / ckpt=model_4500.pt
// 使用: #include "ctrl_params.h"

#pragma once

/* ── 控制周期 (s) ── */
#define CTRL_DT 0.0100f
#define SIM_DT  0.0050f

/* ── PD 参数 (per-joint) ── */
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

/* ── Action 缩放 ── */
#define ACTION_SCALE_JOINT 0.5f
#define ACTION_SCALE_WHEEL 20.0f

/* ── 关节限位 (rad) ── */
#define JOINT_LIMIT_LF_LO -2.27f
#define JOINT_LIMIT_LF_HI 2.40f
#define JOINT_LIMIT_RF_LO -2.27f
#define JOINT_LIMIT_RF_HI 2.40f
#define JOINT_LIMIT_LR_LO -2.62f
#define JOINT_LIMIT_LR_HI 1.57f
#define JOINT_LIMIT_RR_LO -2.62f
#define JOINT_LIMIT_RR_HI 1.57f

/* ── VMC 参数 (与 mujoco.yaml control_config.vmc 一致) ── */
#define VMC_L1               0.215f
#define VMC_L2               0.254f
#define VMC_L3               0.254f
#define VMC_L4               0.215f
#define VMC_L5               0.0f
#define VMC_FEEDFORWARD_FORCE 0.0f
#define VMC_JUMP_FORCE       400.0f
#define JUMP_DURATION        0.30f

#define ACTION_DELAY 2
