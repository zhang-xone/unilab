#ifndef AI_POLICY_H
#define AI_POLICY_H

#include "ChassisControl.h"
#include <math.h>
#include <string.h>
#include "norm_params.h"
#include "ctrl_params.h"

#define ENC_INPUT_DIM        ENC_DIM
#define ACTION_DIM            6

#define PI_6     (0.5235987756f)
#define PI_3     (1.0471975512f)
#define PI_2_3   (2.0943951024f)
#define PI_5_6   (2.6179938780f)
#define M_PI      (3.14159265358979323846f)
#define M_PI_2    (1.57079632679489661923f)
#define DOF_OFFSET_LF   ( PI_6)
#define DOF_OFFSET_RF   ( PI_6)
#define DOF_OFFSET_LR   ( PI_5_6)
#define DOF_OFFSET_RR   ( PI_5_6)
#define DEFAULT_DOF_POS_LF  0.0f
#define DEFAULT_DOF_POS_RF  0.0f
#define DEFAULT_DOF_POS_LR  0.0f
#define DEFAULT_DOF_POS_RR  0.0f

typedef struct {
    float gyro[3];
    float euler[2];
    float commands[4];
    float last_actions[6];
    float dof_pos_err[4];
    float dof_vel[4];
    float wheel_vel[2];
    float theta[2];
    float leg_length[2];
} ObsVector;

typedef struct
{
    ObsVector obs;
    float enc_input[ENC_INPUT_DIM];
    float history_buf[ENC_INPUT_DIM];
    int   history_pos;

    float raw_action[ACTION_DIM];
    float filtered_action[ACTION_DIM];
    float last_actions[ACTION_DIM];

    float dof_pos[6];
    float dof_vel[6];
    float default_dof_pos[4];
    float target_pos[6];
    float torques[6];
    float vmc_torques[6];
    float lqr_torques[6];

    float target_yaw;
    float commands[4];
    float wheel_cmd_vel[2];
    float yaw_err_integral;
    float prev_yaw_error;

    float timing_us[4];
    uint32_t step_cnt;
    float step_dt_us;

    float pitch;
    float roll;

    uint8_t initialized;
} RLController;

void RL_Init(RLController *rl);
void RL_Step(RLController *rl, BalanceInfantry *bal);
void RL_Reset(RLController *rl);
void RL_ComputeTorque(RLController *rl, BalanceInfantry *bal);

#endif