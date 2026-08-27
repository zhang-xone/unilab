#include "ChassisControl.h"
#include "bsp_vofa.h"
#include "bsp_can.h"
#include "observe_task.h"
#include "ranging.h"
#include "power.h"
#include "referee.h"
#define CHASSIS_POWER_GIVE_CURRENT_PARAMETERS 1.23e-07

#define CHASSIS_POWER_RPM_PARAMETERS 1.453e-07

/**
 * @brief 遥控器死区限制
 *
 */
#define rc_deadband_limit(input, output, dealine)        \
    {                                                    \
        if ((input) > (dealine) || (input) < -(dealine)) \
        {                                                \
            (output) = (input);                          \
        }                                                \
        else                                             \
        {                                                \
            (output) = 0;                                \
        }                                                \
    }
BalanceInfantry balance_infantry;
// 左右轮毂电机指针
//MF9025Instance *RightMotor, *LeftMotor;
DJIMotorInstance *RightMotor, *LeftMotor;
DJIMotorInstance *gimbal_yaw, *gimbal_pitch;
uint16_t yaw_en,pitch_en;
ESO_T eso_speed_l = {.z1 = 0.0f,.z2 = 0.0f,.z3 = 0.0f,.max = 4000,.dt = 0.002,.b0 = 1/2,.w0 = 125};
ESO_T eso_speed_r = {.z1 = 0.0f,.z2 = 0.0f,.z3 = 0.0f,.max = 4000,.dt = 0.002,.b0 = 1/2,.w0 = 125};
static inline float limit_float(float value, float min_val, float max_val);
static float normalize_angle(float angle_rad);
static void ESO_Clac(ESO_T * eso , float feedback , float u);
static fp32 get_sin_theta(BalanceInfantry *balance_infantry, uint8_t leg);
static float gas_spring_compen(float L0);
void get_abs_angle(BalanceInfantry *balance_infantry);
float ramp_function(float current, float target, float step, float tolerance) ;
/**
 * @brief        平衡步兵初始化
 * @param    	 BalanceInfantry结构体
 * @retval 	  	none
 */
void BanlanceInfantry_init(BalanceInfantry *BalanceInfantry_init)
{

	/* 获取陀螺仪指针 */
//	BalanceInfantry_init->INS = &INS;
	BalanceInfantry_init->AccelData = get_accel_data_point();
	BalanceInfantry_init->GyroData = get_gyro_data_point();
	BalanceInfantry_init->Ins_Angle = get_INS_angle_point();
	BalanceInfantry_init->Motion_Accel = get_motion_accel_data_point();
	
	BalanceInfantry_init->ChassisRC = get_remote_control_point();

	/* 五连杆杆长初始化 */
	BalanceInfantry_init->L_left[1] = 0.215;
	BalanceInfantry_init->L_left[2] = 0.254;
	BalanceInfantry_init->L_left[3] = 0.254;
	BalanceInfantry_init->L_left[4] = 0.215;
	BalanceInfantry_init->L_left[5] = 0.0;

	BalanceInfantry_init->L_right[1] = 0.215;
	BalanceInfantry_init->L_right[2] = 0.254;
	BalanceInfantry_init->L_right[3] = 0.254;
	BalanceInfantry_init->L_right[4] = 0.215;
	BalanceInfantry_init->L_right[5] = 0.0;

	/* 三次多项式拟合K矩阵以及MPC相关参数 */
	const float K_params[12][4] = {
    {-182.9012, 201.8651, -107.4314, -3.4539},
    {1.8343, -2.6887, -7.2109, -0.3596},
    {-83.4532, 94.9320, -39.8080, -4.3479},
    {-15.5962, 24.3867, -15.7019, -3.5660},
    {-36.2079, 70.5137, -49.8003, 16.5069},
    {-12.4709, 15.6191, -8.2801, 2.5494},
    {404.4386, -325.2095, 68.7846, 13.4138},
    {17.6313, -15.5821, 4.0746, 1.9750},
    {26.4955, 42.8438, -60.4536, 23.5126},
    {-62.9511, 87.8646, -50.6458, 14.9699},
    {421.7204, -437.4865, 167.8944, 5.7210},
    {65.0975, -66.4720, 25.2894, -0.2708}
	};
	const float K_off_ground_params[12][4] = {
    {-265.6598, 295.1344, -136.8455, -9.4001},
    {1.7963, -0.9586, -7.5907, -0.4903},
    {18.0674, -5.2287, -4.4138, -5.1532},
    {85.0680, -69.4960, 16.8830, -7.5902},
    {-81.3390, 107.3282, -59.7791, 17.5661},
    {-28.0681, 29.7099, -12.7209, 3.0984},
    {179.0094, -76.2196, -48.2150, 41.2242},
    {20.8127, -19.2834, 5.1101, 1.8395},
    {111.8564, -66.9997, -4.8906, 10.0593},
    {18.7025, 18.0741, -30.1217, 12.1071},
    {326.0857, -356.8900, 144.7016, 8.4958},
    {47.4548, -52.1564, 21.4002, 0.3406}
	};
	const float MPC_params[6][4] = {
    {483.1682, -462.0897, 128.4944, -2.1108},
    {47.5650, -59.7721, 22.6641, -0.3722},
    {-11.0506, 9.1884, -2.5952, 0.0934},
    {17.1219, -10.3422, -0.3884, 0.1982},
    {45.4255, -46.0406, 17.1641, 0.3544},
    {33.6222, -41.3655, 19.0933, 0.6908}

	};
	for (uint8_t i = 0; i < 12; i++)
	{
		for (uint8_t j = 0; j < 4; j++)
		{
			BalanceInfantry_init->K_coef[i][j] = K_params[i][j];
		}
	}
	//离地时用的theta较大的参数
	for(uint8_t i = 0; i < 12; i++)
	{
		for (uint8_t j = 0; j < 4; j++)
		{
			BalanceInfantry_init->K_coef_off_ground[i][j] = K_off_ground_params[i][j];
		}
	}
	for (uint8_t i = 0; i < 6; i++)
	{
		for (uint8_t j = 0; j < 4; j++)
		{
			BalanceInfantry_init->K_MPC_coef[i][j] = MPC_params[i][j];
		}
	}
	
	
	/* 底盘相关PID参数，包括腿长PID（腿长给P，D可以尽量少给避免震荡），腿速PID（相当于并级PID，只给KP，相当于作为腿长PID的微分项，如果腿部出现震荡，可以降低KP），转向PID（用于小陀螺），跟随云台串级PID */
	const float leg_control_pid_l[3] = {380, 0, 0};
	const float leg_control_pid_r[3] = {380, 0, 0};
	const float leg_syn[3] = {0,0.001,0};
	const float leg_speed_pid_l[3] = {14,0,0};
	const float leg_speed_pid_r[3] = {14,0,0};
	const float roll_control_pid[3] = {400, 0, 10};
	const float leg_angle_control_pid_l[3] = {1.2, 0, 0};
	const float turn_pid[3] = {2500,0.0, 1000};
	const float chassis_track_pid[3] = {5,0.0, 1};
	const float chassis_track_speed_pid[3] = {7000,0,100};
	const float phi_l_pid[3] = {35,0,800};
	const float phi_r_pid[3] = {35,0,800};
	const float phi_l_speed_pid[3] = {3,0,0};
	const float phi_r_speed_pid[3] = {3,0,0};
	const float phi_syn[3] = {20,0,0};//同步两腿运动状态，这里选择同步位置
	const float zero_thera[3] = {70,0,20};//同步两腿运动状态
	const float sita_pid[3] = {0,0.001,0};
	PID_init(&BalanceInfantry_init->turn_pid,PID_POSITION, turn_pid, 3500, 0);
	PID_init(&BalanceInfantry_init->leg_speed_pid_l,PID_POSITION, leg_speed_pid_l, 100, 0);
	PID_init(&BalanceInfantry_init->leg_speed_pid_r,PID_POSITION, leg_speed_pid_r, 100, 0);
	PID_init(&BalanceInfantry_init->leg_angle_control_pid_l, PID_POSITION, leg_angle_control_pid_l, 1, 0);
	PID_init(&BalanceInfantry_init->leg_control_pid_l, PID_POSITION, leg_control_pid_l, 60, 0);
	PID_init(&BalanceInfantry_init->leg_control_pid_r, PID_POSITION, leg_control_pid_r, 60, 0);
	PID_init(&BalanceInfantry_init->roll_control_pid, PID_POSITION, roll_control_pid, 50, 0);
	PID_init(&BalanceInfantry_init->track_pid,PID_POSITION,chassis_track_pid,6,0);
	PID_init(&BalanceInfantry_init->track_speed_pid,PID_POSITION,chassis_track_speed_pid,6000,0);
	PID_init(&BalanceInfantry_init->phi_L_pid,PID_POSITION,phi_l_pid,14,0);
	PID_init(&BalanceInfantry_init->phi_R_pid,PID_POSITION,phi_r_pid,14,0);
	PID_init(&BalanceInfantry_init->phi_L_speed_pid,PID_POSITION,phi_l_speed_pid,8,0);
	PID_init(&BalanceInfantry_init->phi_R_speed_pid,PID_POSITION,phi_r_speed_pid,8,0);
	PID_init(&BalanceInfantry_init->phi_syn_pid,PID_POSITION,phi_syn,10,0);
	PID_init(&BalanceInfantry_init->zero_thera_pid_l,PID_POSITION,zero_thera,15,10);
	PID_init(&BalanceInfantry_init->zero_thera_pid_r,PID_POSITION,zero_thera,15,10);
	PID_init(&BalanceInfantry_init->sita_pid,PID_POSITION,sita_pid,0.08,0.08);
	PID_init(&BalanceInfantry_init->leg_syn_pid,PID_POSITION,leg_syn,0.5,0.5);
	/* 配置3508轮毂电机 */
	Motor_Init_Config_s MotorInitConfig =
		{
			.can_init_config.CanHandle = &hcan2,
			.controller_param_init_config =
				{
					.angle_PID =
						{
							.Kp = 1.5f,
							.Ki = 0.0f,
							.Kd = 0.0f,
							.IntegralLimit = 1.3f,
							.Improve = PID_Trapezoid_Intergral | PID_Integral_Limit | PID_Derivative_On_Measurement,
							.MaxOut = 5.0f,
						},
					.speed_PID =
						{
							.Kp = 0.0f,
							.Ki = 0.05f,
							.Kd = 0.0f,
							.IntegralLimit = 0.0f,
							.Improve = PID_Trapezoid_Intergral | PID_Integral_Limit | PID_Derivative_On_Measurement,
							.MaxOut = 1.0f,
						},
					.current_PID = // 这里先暂用电流环pid做转向pid
					{
						.Kp = 50.0f,
						.Ki = 0.0F,
						.Kd = 0.0f,
						.IntegralLimit = 0.0f,
						.Improve = PID_Trapezoid_Intergral | PID_Integral_Limit | PID_Derivative_On_Measurement,
						.MaxOut = 4000.0f,
					}},
			.controller_setting_init_config =
				{
					.angle_feedback_source = MOTOR_FEED_,
					.speed_feedback_source = MOTOR_FEED_,
					.outer_loop_type = TORQUE_LOOP_,
					.close_loop_type = TORQUE_LOOP_,
				},
			.motor_type = M3508_,
		};
	MotorInitConfig.can_init_config.TxID = 1;
	MotorInitConfig.controller_setting_init_config.motor_reverse_flag = MOTOR_DIRECTION_NORMAL_;
	LeftMotor = DJIMotorInit(&MotorInitConfig);
	MotorInitConfig.can_init_config.CanHandle = &hcan2;
	MotorInitConfig.can_init_config.TxID = 2;
	MotorInitConfig.controller_setting_init_config.motor_reverse_flag = MOTOR_DIRECTION_REVERSE_;
	RightMotor = DJIMotorInit(&MotorInitConfig);
	/* 初始化并使能8009关节电机 */
	joint_motor_init(&balance_infantry.DM8009_Motor[0], 1, MIT_MODE);
	joint_motor_init(&balance_infantry.DM8009_Motor[1], 2, MIT_MODE);
	joint_motor_init(&balance_infantry.DM8009_Motor[2], 4, MIT_MODE);
	joint_motor_init(&balance_infantry.DM8009_Motor[3], 6, MIT_MODE);
	for (uint8_t i = 0; i < 2; i++)
	{
		for (uint8_t j = 0; j < 20; j++)
		{
			enable_motor_mode(&hcan1, balance_infantry.DM8009_Motor[i].para.id, MIT_MODE);
			vTaskDelay(2);
		}
	}
	for (uint8_t i = 2; i < 4; i++)
	{
		for (uint8_t j = 0; j < 10; j++)
		{
			enable_motor_mode(&hcan2, balance_infantry.DM8009_Motor[i].para.id, MIT_MODE);
			vTaskDelay(2);
		}
	}
	//初始化腿长
	balance_infantry.target_L_left = 0.17;
	balance_infantry.target_L_right = 0.17;
	//将跳跃标志位置0
	balance_infantry.jump_flag=0;
	balance_infantry.jump_done_flag =0;
	balance_infantry.jump_status = 0;
	//将上台阶标志位与状态置0
	balance_infantry.stair_flag=0;
	balance_infantry.stair_status=0;
	//初始化重力前馈
	balance_infantry.qiankui= 0;
	balance_infantry.step_shirink_flag = 0;
	//将起立完成标志位置1
	balance_infantry.stand_done_flag = 0;
	balance_infantry.jump_shirink_flag = 0;
	balance_infantry.rotation_flag=0;
	//将离地标志位置0，即初始化不离地
	balance_infantry.is_get_enough_force=0;
	balance_infantry.is_get_enough_force_l=0;
	balance_infantry.is_get_enough_force_r=0;
	//初始化机械零点
	balance_infantry.target_vector[4] = 0.0f;
	//初始化腿长
	 balance_infantry.init_L_leg = 0.15;
	 balance_infantry.leg_flag = 0;
	//初始化功率控制
  chassis_power_init(BalanceInfantry_init);
	//翻倒自救标志位
	balance_infantry.Self_resuce_flag_X = 0;
	//yaw中值
	yaw_en = 173;
}
const float R[3][3] = 
{
	{1.0 , 0.0 , 0.0},
	{0.0 , 0.0 , 1.0},
	{0.0 , -1.0 , 0.0}
};
float start[3] = {0.0},angle_result[3] = {0.0};
void rotate_vector_left_multiply(float *vec, const float rot[3][3], float *result) {
    for (int j = 0; j < 3; j++) {
        result[j] = 0.0;
        for (int i = 0; i < 3; i++) {
            result[j] += vec[i] * rot[j][i];
        }
    }
}

/**
 * @brief        获取传感器信息
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
void get_sensor_message(BalanceInfantry *balance_infantry)
{
	/*
	theta,theta_dot,x,x_dot,phi,phi_dot
	*/
	for(int i = 0;i<3;i++)
	{
		start[i] = balance_infantry->Ins_Angle[i];
	}
	rotate_vector_left_multiply(start,R,angle_result);
	
	/* IMU数据获取 */
	balance_infantry->state_vector_l[4] = normalize_angle(balance_infantry->Ins_Angle[2]);
	balance_infantry->state_vector_l[5] = balance_infantry->GyroData[0];
	balance_infantry->state_vector_r[4] = normalize_angle(balance_infantry->Ins_Angle[2]);
	balance_infantry->state_vector_r[5] = balance_infantry->GyroData[0];

	/* 获取x,x_dot */
	balance_infantry->state_vector_l[3] = balance_infantry->v_filter;
	balance_infantry->state_vector_r[3] = balance_infantry->v_filter;
	balance_infantry->state_vector_l[2] = balance_infantry->x_filter;
	balance_infantry->state_vector_r[2] = balance_infantry->x_filter;
	
	/* 五连杆phi角获取 */
	balance_infantry->phi_left[1] = balance_infantry->DM8009_Motor[0].para.pos + PI;//
	balance_infantry->phi_left[4] = balance_infantry->DM8009_Motor[1].para.pos;//
	balance_infantry->phi_right[1] = -balance_infantry->DM8009_Motor[3].para.pos + PI;
	balance_infantry->phi_right[4] = -balance_infantry->DM8009_Motor[2].para.pos;
	/* 关节电机反馈力矩 */
	for (uint8_t i = 0; i < 4; i++)
	{
		balance_infantry->sensors_info.torque_feed_back[i] = balance_infantry->DM8009_Motor[i].para.tor;
	}

	/* 反馈轮毂电机力矩  */
	balance_infantry->sensors_info.wheel_torque[0] = LeftMotor->measure.real_torque;
	balance_infantry->sensors_info.wheel_torque[1] = RightMotor->measure.real_torque;

	
	if(balance_infantry->ChassisMode != balance_infantry->Last_ChassisMode)
	{
		balance_infantry->yaw_set = balance_infantry->Ins_Angle[0];
	}
  if (balance_infantry->Ins_Angle[0] -balance_infantry->yaw_last_angle > 3.1415926f)
	{
		balance_infantry->yaw_count--;
	}
	else if (balance_infantry->Ins_Angle[0] -balance_infantry->yaw_last_angle < -3.1415926f)
	{
		balance_infantry->yaw_count++;
	}
	balance_infantry->yaw_total_angle = 6.283f* balance_infantry->yaw_count + balance_infantry->Ins_Angle[0];
	balance_infantry->yaw_last_angle = balance_infantry->Ins_Angle[0];          
	balance_infantry->yaw_gyro = balance_infantry->Ins_Angle[1];
	balance_infantry->chassis_relative_angle = rad_format(motor_ecd_to_angle_change(yaw_ecd,yaw_en));
	
	//计算气弹簧
	balance_infantry->F_Gas_left = gas_spring_compen(balance_infantry->L_left[0]);
	balance_infantry->F_Gas_right = gas_spring_compen(balance_infantry->L_right[0]);


	
//		balance_infantry->Fx_left = 37.5f*get_sin_theta(balance_infantry,LEG_LEFT) * arm_cos_f32(balance_infantry->phi_left[2] -PI/2);
//		balance_infantry->Fx_right = 37.5f*get_sin_theta(balance_infantry,LEG_RIGHT) * arm_cos_f32(balance_infantry->phi_right[2] - PI/2);
//		balance_infantry->Fy_left = 37.5f*get_sin_theta(balance_infantry,LEG_LEFT) * arm_sin_f32(balance_infantry->phi_left[2] - PI/2);
//		balance_infantry->Fy_right = 37.5f*get_sin_theta(balance_infantry,LEG_RIGHT) * arm_sin_f32(balance_infantry->phi_right[2]- PI/2);
//	//??????????????????????????
//	if(balance_infantry->stand_done_flag != 2){
//		//此时水平的力往左
//		balance_infantry->F_Gas_left = balance_infantry->Fx_left;// * arm_cos_f32(balance_infantry->phi_left[0]) ;// * balance_infantry->L_left[1] * arm_sin_f32(PI - balance_infantry->phi_left[1] + balance_infantry->phi_left[2]) / balance_infantry->L_left[0];
////		balance_infantry->extra_torque_l += 37.5f*get_sin_theta(balance_infantry,LEG_LEFT) * arm_sin_f32(balance_infantry->phi_left[2] - PI/2) * balance_infantry->L_left[0] * arm_sin_f32(balance_infantry->phi_left[0]);
//		balance_infantry->F_Gas_right = balance_infantry->Fx_right;//* arm_cos_f32(balance_infantry->phi_right[0]) ;//* balance_infantry->L_right[1] * arm_sin_f32(PI - balance_infantry->phi_right[1] + balance_infantry->phi_right[2]) / balance_infantry->L_right[0];
////		balance_infantry->extra_torque_r += 37.5f*get_sin_theta(balance_infantry,LEG_RIGHT) * arm_sin_f32(balance_infantry->phi_right[2] - PI/2) * balance_infantry->L_right[0] * arm_sin_f32(balance_infantry->phi_right[0]);
//	}else{
//		//此时竖直的力往下
//		balance_infantry->F_Gas_left = -150.0 * 0.25*get_sin_theta(balance_infantry,LEG_LEFT) * arm_sin_f32(balance_infantry->phi_left[2] - PI/2);// * arm_sin_f32(balance_infantry->phi_left[0]);// * balance_infantry->L_left[1] * arm_sin_f32(PI - balance_infantry->phi_left[1] + balance_infantry->phi_left[2]) / balance_infantry->L_left[0];
//		balance_infantry->F_Gas_right = -150.0 * 0.25*get_sin_theta(balance_infantry,LEG_RIGHT) * arm_sin_f32(balance_infantry->phi_right[2]- PI/2);// * arm_sin_f32(balance_infantry->phi_right[0]);//* balance_infantry->L_right[1] * arm_sin_f32(PI - balance_infantry->phi_right[1] + balance_infantry->phi_right[2]) / balance_infantry->L_right[0];
//////		balance_infantry->extra_torque_l += 0.5f*arm_cos_f32(balance_infantry->phi_left[0]) * balance_infantry->L_left[0] * 37.5f*get_sin_theta(balance_infantry,LEG_LEFT) * arm_cos_f32(balance_infantry->phi_left[2] - PI/2);
//////		balance_infantry->extra_torque_r += 0.5f*arm_cos_f32(balance_infantry->phi_right[0]) * balance_infantry->L_right[0] * 37.5f*get_sin_theta(balance_infantry,LEG_RIGHT) * arm_cos_f32(balance_infantry->phi_right[2] - PI/2);
//	}
}
/**
 * @brief        底盘模式选择
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
void ChassisModeSet(BalanceInfantry *balance_infantry)
{
	switch (balance_infantry->ChassisRC->rc.switch_mid)
	{
		case RC_SW_DOWN:
			balance_infantry->ChassisMode = CHASSIS_ZERO_FORCE;
			balance_infantry->x_filter=0;
			balance_infantry->target_vector[2]=0;
			break;
		case RC_SW_MID:
			balance_infantry->ChassisMode = CHASSIS_NO_FOLLOW_YAW;
			break;
		case RC_SW_UP:
			balance_infantry->ChassisMode = CHASSIS_FOLLOW_GIMBAL_YAW;
			break;
		default:
//			balance_infantry->ChassisMode = CHASSIS_ZERO_FORCE;
			break;
	}
//	balance_infantry->lastmode = balance_infantry->ChassisRC->rc.s[1];
	if(balance_infantry->ChassisMode == CHASSIS_NO_FOLLOW_YAW && balance_infantry->Last_ChassisMode != CHASSIS_NO_FOLLOW_YAW)
	{
		balance_infantry->step_shirink_flag = 1;
		balance_infantry->stand_state = 0;
	}
	if(balance_infantry->ChassisMode == CHASSIS_ZERO_FORCE && balance_infantry->Last_ChassisMode != CHASSIS_ZERO_FORCE)
	{
			balance_infantry->x_filter=0;
			balance_infantry->target_vector[2]=0;
	}
	
	if(balance_infantry->ChassisMode != CHASSIS_ZERO_FORCE && !balance_infantry->Self_resuce_flag_X && !balance_infantry->stand_or_not)
	{
			balance_infantry->ChassisMode = CHASSIS_ZERO_FORCE;
	}
	
	if(balance_infantry->ChassisMode != CHASSIS_NO_FOLLOW_YAW && balance_infantry->key & KEY_C)
	{
			balance_infantry->ChassisMode = CHASSIS_NO_FOLLOW_YAW;
	}
	
	balance_infantry->Last_ChassisMode = balance_infantry->ChassisMode;

}

/**
 * @brief        根据遥控器的值计算速度
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
fp32 count=0;
fp32 Vxoffset = 0;
fp32 Vyoffset = 0.13;
float LeftSteerTorque,RightSteerTorque;
extern RC_ctrl_t rc_ctrl;
void ChassisDataSet(BalanceInfantry *balance_infantry)
{
	fp32 VxSetChannel,VySetChannel;
	
	fp32 target_wz = 0,angle_set;
	
	rc_deadband_limit(balance_infantry->ChassisRC->rc.rocker_r1, VxSetChannel, CHASSIS_RC_DEADLINE);
	// 没有值直接给零
	if (VxSetChannel < CHASSIS_RC_DEADLINE * CHASSIS_V_SEN && VxSetChannel > -CHASSIS_RC_DEADLINE * CHASSIS_V_SEN)
	{
		VxSetChannel = 0.0f;
	}
//					vx_set_channel = ramp_function(chassis_move_rc_to_vector->vx,chassis_move_rc_to_vector->vx_max_speed,1.5,0.1);
	VxSetChannel = CHASSIS_V_SEN * balance_infantry->ChassisRC->rc.rocker_r1;
//	if(VxSetChannel > 1.0f){
//		VxSetChannel = 1.0f;
//	}else if(VxSetChannel < -1.0f){
//		VxSetChannel = -1.0f;
//	}else{
//		VxSetChannel = 0.0f;
//	}
	
	
//	slope_following(&VxSetChannel,&balance_infantry->target_vector[3],0.05);
//	if(balance_infantry->rotation_flag == 0){
		balance_infantry->target_vector[3] = ramp_function(balance_infantry->state_vector_l[3], VxSetChannel, 1.0, 0.8);//LIMIT_MAX_MIN(VxSetChannel,balance_infantry->state_vector_l[3] + 1.0f,balance_infantry->state_vector_l[3]-1.0f);//balance_infantry->vx_set
		if(rc_ctrl.key.v & KEY_PRESSED_OFFSET_S)
		{
			balance_infantry->target_vector[3] = ramp_function(balance_infantry->state_vector_l[3], -MAX_V_SPEED, 0.8, 0.5);
		}
		if(rc_ctrl.key.v & KEY_PRESSED_OFFSET_W)
		{
				if(balance_infantry->stair_flag != 0)
				{
					balance_infantry->target_vector[3] = LIMIT_MAX_MIN(MAX_V_SPEED_STAIR,balance_infantry->state_vector_l[3] + 0.9f,balance_infantry->state_vector_l[3] - 0.9f);
				}
				else
				{
					balance_infantry->target_vector[3] = ramp_function(balance_infantry->state_vector_l[3], MAX_V_SPEED, 0.8, 0.5);
				}

		}
		balance_infantry->target_vector[2] = balance_infantry->state_vector_l[2];
//		balance_infantry->target_vector[2] += balance_infantry->target_vector[3] * 0.0018f;
		balance_infantry->target_vector[2] = limit_float(balance_infantry->target_vector[2],balance_infantry->state_vector_l[2] - 0.35,balance_infantry->state_vector_l[2] +0.2);
//	}
//	else{
//			VySetChannel = CHASSIS_V_SEN * balance_infantry->ChassisRC->rc.ch[0];
//			balance_infantry->target_vector[3] = (VySetChannel/4.0f+Vyoffset) * arm_cos_f32(normalize_angle(balance_infantry->chassis_relative_angle + 0.40f))+ (VxSetChannel/4.0f+Vxoffset) * arm_sin_f32(normalize_angle(balance_infantry->chassis_relative_angle+0.40f));
//			balance_infantry->target_vector[2] += balance_infantry->target_vector[3] * 0.0015f;
//			balance_infantry->target_vector[2] = limit_float(balance_infantry->target_vector[2],balance_infantry->state_vector_l[2] - 2,balance_infantry->state_vector_l[2] +2);
//	}
	//不跟随云台部分
	if(balance_infantry->ChassisMode == CHASSIS_NO_FOLLOW_YAW)
	{

		//控制yaw角度
		//balance_infantry->yaw_set  += balance_infantry->ChassisRC->rc.ch[0] * -0.00000857575757575f;
//		balance_infantry->Left_current = PID_calc(&balance_infantry->turn_pid,balance_infantry->yaw_total_angle , balance_infantry->yaw_set);
//		balance_infantry->Right_current = PID_calc(&balance_infantry->turn_pid,balance_infantry->yaw_total_angle , balance_infantry->yaw_set);
		//控制角速度
//		balance_infantry->w_set = balance_infantry->ChassisRC->rc.ch[2] * -0.015f;//转向控制
		balance_infantry->Left_current = PID_calc(&balance_infantry->turn_pid,balance_infantry->GyroData[2] * 1.08f+balance_infantry->GyroData[0]*arm_sin_f32(balance_infantry->state_vector_l[4]) * 0.29f, balance_infantry->w_set);
		balance_infantry->Right_current = PID_calc(&balance_infantry->turn_pid,balance_infantry->GyroData[2] * 0.92f+balance_infantry->GyroData[0]*arm_sin_f32(balance_infantry->state_vector_r[4]) * 0.13f, balance_infantry->w_set);
		count ++;
		count = count>= 1001 ? 1001 : count;
		if(balance_infantry->ChassisRC->rc.ch[4] < -500)//小陀螺
		{
			target_wz = rotation_WZ;
			balance_infantry->Left_current = PID_calc(&balance_infantry->turn_pid,balance_infantry->GyroData[2]+balance_infantry->GyroData[0]*arm_sin_f32(balance_infantry->state_vector_l[4]) * 0.21f , target_wz);
			balance_infantry->Right_current = PID_calc(&balance_infantry->turn_pid, balance_infantry->GyroData[2]+balance_infantry->GyroData[0]*arm_sin_f32(balance_infantry->state_vector_r[4]) * 0.21f, target_wz);
		}

	}
	//跟随云台部分
	if(balance_infantry->ChassisMode == CHASSIS_FOLLOW_GIMBAL_YAW)
	{
		count ++;
		count = count>= 1001 ? 1001 : count;

		if(balance_infantry->ChassisRC->rc.ch[4] < -500 || rc_ctrl.key.v & KEY_PRESSED_OFFSET_SHIFT)//小陀螺
		{
			balance_infantry->rotation_flag = 1;
			target_wz = rotation_WZ;
			balance_infantry->Left_current = -PID_calc(&balance_infantry->turn_pid,-(balance_infantry->GyroData[2]+balance_infantry->GyroData[0]*arm_sin_f32(balance_infantry->state_vector_l[4])) * 0.21f , target_wz);
			balance_infantry->Right_current = -PID_calc(&balance_infantry->turn_pid,(-balance_infantry->GyroData[2]+balance_infantry->GyroData[0]*arm_sin_f32(balance_infantry->state_vector_r[4])) * 0.21f, target_wz);
		}
		else//跟随云台
		{	
			balance_infantry->rotation_flag =0;
			balance_infantry->chassis_relative_angle_set = 0.0f;
			angle_set = PID_calc(&balance_infantry->track_pid,-rad_format(balance_infantry->chassis_relative_angle - balance_infantry->chassis_relative_angle_set) , 0);	
			//
			balance_infantry->Left_current = PID_calc(&balance_infantry->track_speed_pid,(balance_infantry->GyroData[2]+balance_infantry->GyroData[0]*arm_sin_f32(balance_infantry->state_vector_l[4])),angle_set);
			balance_infantry->Right_current = PID_calc(&balance_infantry->track_speed_pid,(balance_infantry->GyroData[2]+balance_infantry->GyroData[0]*arm_sin_f32(balance_infantry->state_vector_r[4])),angle_set);
		}
	}
	if(balance_infantry->ChassisMode == CHASSIS_ZERO_FORCE)
	{
		count=0;
	}
	//roll轴部分
	if(balance_infantry->ChassisMode == CHASSIS_IKUN)
	{
		balance_infantry->roll_set = balance_infantry->ChassisRC->rc.ch[2] * 0.00052666f;
	}
	else
	{
		balance_infantry->roll_set = 0;//LIMIT_MAX_MIN(arm_atan2_f32(balance_infantry->state_vector_l[3] * balance_infantry->GyroData[2],9.81),0.26f,-0.26f);
	}

	//功能部分
	
	//自动跳跃按键
	if((balance_infantry->key & KEY_F) && !(balance_infantry->last_key & KEY_F)) {
    balance_infantry->jump_flag=~balance_infantry->jump_flag;
		balance_infantry->jump_shirink_flag = ~balance_infantry->jump_shirink_flag;
		balance_infantry->jump_done_flag = 0;
		balance_infantry->target_vector[2] = -0.0f;
		balance_infantry->x_filter = 0;
	}
	if(balance_infantry->jump_done_flag){
		balance_infantry->jump_flag=0;
		balance_infantry->jump_shirink_flag = 0;
		balance_infantry->jump_done_flag = 0;
	}
	
	//磕台阶
	if((balance_infantry->key & KEY_B) && !(balance_infantry->last_key & KEY_B))
	{
		balance_infantry->stair_flag = ~balance_infantry->stair_flag;
	}
	
	//抬腿
	if(balance_infantry->stand_done_flag == 2 && balance_infantry->stair_flag == 0 && balance_infantry->jump_flag == 0){//跳跃和磕台阶情况下单独控制腿长
		if(balance_infantry->key & KEY_R && !(balance_infantry->last_key & KEY_R)) {
			balance_infantry->leg_flag = ~balance_infantry->leg_flag;
			balance_infantry->target_vector[2] = 0.0f;
		}
			if(balance_infantry->leg_flag == 0){
			balance_infantry->target_L_left = 0.18f;
			balance_infantry->target_L_right = 0.18f;
			}else{
					balance_infantry->target_L_left = 0.27f;
					balance_infantry->target_L_right = 0.27f;
			}
	}
	
	//翻倒自救
	if((balance_infantry->key & KEY_X) && !(balance_infantry->last_key & KEY_X))
	{
		balance_infantry->Self_resuce_flag_X = ~balance_infantry->Self_resuce_flag_X;
		balance_infantry->self_resuce_state = 0;
	}
	
	//小陀螺
	if(balance_infantry->key & KEY_PRESSED_OFFSET_SHIFT)
	{
		balance_infantry->target_vector[2] = balance_infantry->state_vector_l[2];
		balance_infantry->target_vector[3] = balance_infantry->state_vector_l[3];
	}
	balance_infantry->last_key = balance_infantry->key;
//	if(balance_infantry->ChassisRC->rc.s[1]==RC_SW_UP && balance_infantry->lastmode == RC_SW_UP)
//	{
//		balance_infantry->jump_flag=1;
//		balance_infantry->jump_shirink_flag = 1;
//	}
//	else if(balance_infantry->ChassisRC->rc.s[1] != RC_SW_UP)
//	{
//		balance_infantry->jump_flag=0;
//		balance_infantry->jump_shirink_flag = 0;
//		balance_infantry->jump_done_flag = 0;
//	}
//	if( balance_infantry->ChassisRC->rc.s[1]==RC_SW_MID && balance_infantry->lastmode!=RC_SW_MID)
//	{
//		balance_infantry->stair_flag = 1;
//	}
//	积分
		balance_infantry->target_vector[0]=-0.01f;
		if(!balance_infantry->stair_flag){
			balance_infantry->target_vector[0] -= PID_calc(&balance_infantry->sita_pid,
																								 balance_infantry->state_vector_l[0],
																								 balance_infantry->target_vector[0]);
		}
}

/**
 * @brief        ??????
 * @param    	  BalanceInfantry???
 * @retval 	  none
 */
void set_target_vector(BalanceInfantry *balance_infantry)
{
	/* 控速度还是控位移待测试 */
	// theta,theta_dot,x,x_dot,phi,phi_dot
	ChassisModeSet(balance_infantry);
	ChassisDataSet(balance_infantry);
	switch (balance_infantry->ChassisMode)
	{
	case CHASSIS_NO_FOLLOW_YAW:
	case CHASSIS_IKUN:
	case CHASSIS_FOLLOW_GIMBAL_YAW:
		for (uint8_t i = 0; i < 10; i++)
		{
			for (uint8_t j = 0; j < 4; j++)
			{
				enable_motor_mode(&JOINT_HCAN, MIT_MODE, balance_infantry->DM8009_Motor[j].para.id);
			}
		}
		balance_infantry->target_vector[0] = -0.13f;
		//轮腿模型应为0
//		balance_infantry->target_vector[4] = -0.04f;
		balance_infantry->target_vector[5] = 0.0f;
		
		break;
	case CHASSIS_ZERO_FORCE:
		/* 失能关节电机和轮毂电机 */
		for (uint8_t i = 0; i < 10; i++)
		{
			for (uint8_t j = 0; j < 4; j++)
			{
				enable_motor_mode(&JOINT_HCAN, MIT_MODE, balance_infantry->DM8009_Motor[j].para.id);
			}
		}
		RightMotor->stop_flag = MOTOR_STOP_;
		LeftMotor->stop_flag = MOTOR_STOP_;
		break;
	default:
		break;
	}
}
/**
 * @brief        逆运动学解算
 * @param    	  BalanceInfantry结构体，轮子x轴，轮子y轴
 * @retval 	  none
 */
void InvMotionSolve(BalanceInfantry *balance_infantry,float L_0,float phi_0)
{
    float x_C= L_0 * arm_cos_f32(phi_0) + balance_infantry->L_left[5]/2;
    float y_C= L_0 * arm_sin_f32(phi_0);
    float A= balance_infantry->L_left[1] + x_C;
    float B= balance_infantry->L_left[1] * balance_infantry->L_left[1] - x_C * x_C;
    float C= balance_infantry->L_left[2] * balance_infantry->L_left[2] - y_C * y_C;
    float D= balance_infantry->L_left[2] * balance_infantry->L_left[2] + y_C * y_C;

    float fai1 = 2* arm_atan2_f32( (2*balance_infantry->L_left[1]*y_C + sqrt( 2*balance_infantry->L_left[1]*balance_infantry->L_left[1] * D + 2*x_C*x_C*C - B*B-C*C) ), ( A* A-C ) );
    if(fai1>2.0f*PI)fai1-=2.0f*PI;
    if(fai1<0)fai1+=2.0f*PI;

    A= balance_infantry->L_left[3] + balance_infantry->L_left[4];
    B=balance_infantry->L_left[5]- x_C;
    C=balance_infantry->L_left[3]-balance_infantry->L_left[4];
    D=x_C+balance_infantry->L_left[4]-balance_infantry->L_left[5];
    float E=A*A - B*B - y_C*y_C;
    float F=B*B - C*C + y_C*y_C;
    float fai4 = 2* arm_atan2_f32( (2*balance_infantry->L_left[4]*y_C - sqrt( E*F) ), ( D*D+y_C*y_C-balance_infantry->L_left[3]*balance_infantry->L_left[3] ) );
    if(fai4>PI)fai4-=2.0f*PI;
    if(fai4<-PI)fai4+=2.0f*PI;
                      
    balance_infantry->fai1 = fai1;
    balance_infantry->fai4 = fai4;
}
/**
 * @brief        正运动学解算
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
void MotionSolve(BalanceInfantry *balance_infantry)
{
	static float B_D_distance_2_l, A0_l, C0_l, temp_l;
	static float b0_l;
	static float B_D_distance_2_r, A0_r, C0_r, temp_r;
	static float b0_r;

	/* 左腿正运动学解算 */
	balance_infantry->SolverLeft.x_B =
		balance_infantry->L_left[1] * arm_cos_f32(balance_infantry->phi_left[1]);
	balance_infantry->SolverLeft.y_B =
		balance_infantry->L_left[1] * arm_sin_f32(balance_infantry->phi_left[1]);
	balance_infantry->SolverLeft.x_D =
		balance_infantry->L_left[5] + balance_infantry->L_left[4] * arm_cos_f32(balance_infantry->phi_left[4]);
	balance_infantry->SolverLeft.y_D =
		balance_infantry->L_left[4] * arm_sin_f32(balance_infantry->phi_left[4]);
	B_D_distance_2_l =
		(balance_infantry->SolverLeft.x_D - balance_infantry->SolverLeft.x_B) * (balance_infantry->SolverLeft.x_D - balance_infantry->SolverLeft.x_B) + (balance_infantry->SolverLeft.y_D - balance_infantry->SolverLeft.y_B) * (balance_infantry->SolverLeft.y_D - balance_infantry->SolverLeft.y_B);
	A0_l = 2 * balance_infantry->L_left[2] * (balance_infantry->SolverLeft.x_D - balance_infantry->SolverLeft.x_B);
	b0_l = 2 * balance_infantry->L_left[2] * (balance_infantry->SolverLeft.y_D - balance_infantry->SolverLeft.y_B);
	C0_l = balance_infantry->L_left[2] * balance_infantry->L_left[2] + B_D_distance_2_l - balance_infantry->L_left[3] * balance_infantry->L_left[3];
	temp_l = b0_l + sqrt(A0_l * A0_l + b0_l * b0_l - C0_l * C0_l);
	balance_infantry->phi_left[2] = 2 * atan2f(temp_l, A0_l + C0_l);

	balance_infantry->SolverLeft.x_C =
		balance_infantry->SolverLeft.x_B + balance_infantry->L_left[2] * arm_cos_f32(balance_infantry->phi_left[2]);
	balance_infantry->SolverLeft.y_C =
		balance_infantry->SolverLeft.y_B + balance_infantry->L_left[2] * arm_sin_f32(balance_infantry->phi_left[2]);
	balance_infantry->SolverLeft.x_C -=
		balance_infantry->L_left[5] / 2;

	balance_infantry->L_left[0] =
		sqrt(balance_infantry->SolverLeft.x_C * balance_infantry->SolverLeft.x_C + balance_infantry->SolverLeft.y_C * balance_infantry->SolverLeft.y_C);
	balance_infantry->phi_left[0] =
		atan2f(balance_infantry->SolverLeft.y_C, balance_infantry->SolverLeft.x_C);
//	balance_infantry->SolverLeft.x_C +=
//		balance_infantry->L_left[5] / 2;
	balance_infantry->phi_left[3] = atan2f(balance_infantry->SolverLeft.y_B - balance_infantry->SolverLeft.y_D + balance_infantry->L_left[2] * arm_sin_f32(balance_infantry->phi_left[2]),balance_infantry->SolverLeft.x_B - balance_infantry->SolverLeft.x_D + balance_infantry->L_left[2] * arm_cos_f32(balance_infantry->phi_left[2]));
//	balance_infantry->phi_left[3] = PI - arm_atan2_f32(balance_infantry->SolverLeft.y_C - balance_infantry->SolverLeft.y_D,balance_infantry->SolverLeft.x_D - balance_infantry->SolverLeft.x_B);
	
	/* 右腿正运动学解算 */
	balance_infantry->SolverRight.x_B =
		balance_infantry->L_right[1] * arm_cos_f32(balance_infantry->phi_right[1]);
	balance_infantry->SolverRight.y_B =
		balance_infantry->L_right[1] * arm_sin_f32(balance_infantry->phi_right[1]);
	balance_infantry->SolverRight.x_D =
		balance_infantry->L_right[5] + balance_infantry->L_right[4] * arm_cos_f32(balance_infantry->phi_right[4]);
	balance_infantry->SolverRight.y_D =
		balance_infantry->L_right[4] * arm_sin_f32(balance_infantry->phi_right[4]);
	B_D_distance_2_r =
		(balance_infantry->SolverRight.x_D - balance_infantry->SolverRight.x_B) * (balance_infantry->SolverRight.x_D - balance_infantry->SolverRight.x_B) + (balance_infantry->SolverRight.y_D - balance_infantry->SolverRight.y_B) * (balance_infantry->SolverRight.y_D - balance_infantry->SolverRight.y_B);
	A0_r = 2 * balance_infantry->L_right[2] * (balance_infantry->SolverRight.x_D - balance_infantry->SolverRight.x_B);
	b0_r = 2 * balance_infantry->L_right[2] * (balance_infantry->SolverRight.y_D - balance_infantry->SolverRight.y_B);
	C0_r = balance_infantry->L_right[2] * balance_infantry->L_right[2] + B_D_distance_2_r - balance_infantry->L_right[3] * balance_infantry->L_right[3];
	temp_r = b0_r + sqrt(A0_r * A0_r + b0_r * b0_r - C0_r * C0_r);
	balance_infantry->phi_right[2] = 2 * arm_atan2_f32(temp_r, A0_r + C0_r);

	balance_infantry->SolverRight.x_C =
		balance_infantry->SolverRight.x_B + balance_infantry->L_right[2] * arm_cos_f32(balance_infantry->phi_right[2]);
	balance_infantry->SolverRight.y_C =
		balance_infantry->SolverRight.y_B + balance_infantry->L_right[2] * arm_sin_f32(balance_infantry->phi_right[2]);
	balance_infantry->SolverRight.x_C -=
		balance_infantry->L_right[5] / 2;

	balance_infantry->L_right[0] =
		sqrt(balance_infantry->SolverRight.x_C * balance_infantry->SolverRight.x_C + balance_infantry->SolverRight.y_C * balance_infantry->SolverRight.y_C);
	balance_infantry->phi_right[0] =
		arm_atan2_f32(balance_infantry->SolverRight.y_C, balance_infantry->SolverRight.x_C);
//	balance_infantry->SolverRight.x_C +=
//		balance_infantry->L_right[5] / 2;
	balance_infantry->phi_right[3] = arm_atan2_f32(balance_infantry->SolverRight.y_B - balance_infantry->SolverRight.y_D + balance_infantry->L_right[2] * arm_sin_f32(balance_infantry->phi_right[2]),balance_infantry->SolverRight.x_B - balance_infantry->SolverRight.x_D + balance_infantry->L_right[2] * arm_cos_f32(balance_infantry->phi_right[2]));


	float sigma_1_l, sigma_2_l, sigma_3_l;
	float sigma_1_r, sigma_2_r, sigma_3_r;

	/* 定义sigma角 */
	sigma_1_l = arm_sin_f32(balance_infantry->phi_left[3] - balance_infantry->phi_left[2]);
	sigma_2_l = arm_sin_f32(balance_infantry->phi_left[3] - balance_infantry->phi_left[4]);
	sigma_3_l = arm_sin_f32(balance_infantry->phi_left[1] - balance_infantry->phi_left[2]);

	/* 雅可比矩阵的计算 */
	balance_infantry->J_l[0][0] =
		balance_infantry->L_left[1] * arm_cos_f32(balance_infantry->phi_left[0] - balance_infantry->phi_left[3]) * sigma_3_l / balance_infantry->L_left[0] / sigma_1_l;
	balance_infantry->J_l[0][1] =
		balance_infantry->L_left[1] * arm_sin_f32(balance_infantry->phi_left[0] - balance_infantry->phi_left[3]) * sigma_3_l / sigma_1_l;
	balance_infantry->J_l[1][0] =
		balance_infantry->L_left[4] * arm_cos_f32(balance_infantry->phi_left[0] - balance_infantry->phi_left[2]) * sigma_2_l / balance_infantry->L_left[0] / sigma_1_l;
	balance_infantry->J_l[1][1] =
	 balance_infantry->L_left[4] * arm_sin_f32(balance_infantry->phi_left[0] - balance_infantry->phi_left[2]) * sigma_2_l / sigma_1_l;
    
		/* 求解雅可比矩阵的逆矩阵，用于离地检测 */
    balance_infantry->J_inv_left[0][0] = balance_infantry->L_left[0] * arm_sin_f32(balance_infantry->phi_left[0] - balance_infantry->phi_left[2]) / balance_infantry->L_left[1] * sigma_3_l;
    balance_infantry->J_inv_left[0][1] = -balance_infantry->L_left[0] * arm_sin_f32(balance_infantry->phi_left[0] - balance_infantry->phi_left[3]) / balance_infantry->L_left[4] * sigma_2_l;
    balance_infantry->J_inv_left[1][0] = -arm_cos_f32(balance_infantry->phi_left[0] - balance_infantry->phi_left[2]) / balance_infantry->L_left[1] * sigma_3_l;
    balance_infantry->J_inv_left[1][1] = arm_cos_f32(balance_infantry->phi_left[0] - balance_infantry->phi_left[3]) / balance_infantry->L_left[4] * sigma_2_l;

	/* 右腿的VMC解算 */
	sigma_1_r = arm_sin_f32(balance_infantry->phi_right[3] - balance_infantry->phi_right[2]);
	sigma_2_r = arm_sin_f32(balance_infantry->phi_right[3] - balance_infantry->phi_right[4]);
	sigma_3_r = arm_sin_f32(balance_infantry->phi_right[1] - balance_infantry->phi_right[2]);

	/* 雅可比矩阵的计算 */
	balance_infantry->J_r[0][0] =
		balance_infantry->L_right[1] * arm_cos_f32(balance_infantry->phi_right[0] - balance_infantry->phi_right[3]) * sigma_3_r / balance_infantry->L_right[0] / sigma_1_r;
	balance_infantry->J_r[0][1] =
		balance_infantry->L_right[1] * arm_sin_f32(balance_infantry->phi_right[0] - balance_infantry->phi_right[3]) * sigma_3_r / sigma_1_r;
	balance_infantry->J_r[1][0] =
		balance_infantry->L_right[4] * arm_cos_f32(balance_infantry->phi_right[0] - balance_infantry->phi_right[2]) * sigma_2_r / balance_infantry->L_right[0] / sigma_1_r;
	balance_infantry->J_r[1][1] =
		balance_infantry->L_right[4] * arm_sin_f32(balance_infantry->phi_right[0] - balance_infantry->phi_right[2]) * sigma_2_r / sigma_1_r;
	   /* 求解雅可比矩阵的逆矩阵，用于离地检测 */
    balance_infantry->J_inv_right[0][0] = balance_infantry->L_right[0] * arm_sin_f32(balance_infantry->phi_right[0] - balance_infantry->phi_right[2]) / balance_infantry->L_right[1] * sigma_3_r;
    balance_infantry->J_inv_right[0][1] = -balance_infantry->L_right[0] * arm_sin_f32(balance_infantry->phi_right[0] - balance_infantry->phi_right[3]) / balance_infantry->L_right[4] * sigma_2_r;
    balance_infantry->J_inv_right[1][0] = -arm_cos_f32(balance_infantry->phi_right[0] - balance_infantry->phi_right[2]) / balance_infantry->L_right[1] * sigma_3_r;
    balance_infantry->J_inv_right[1][1] = arm_cos_f32(balance_infantry->phi_right[0] - balance_infantry->phi_right[3]) / balance_infantry->L_right[4] * sigma_2_r;

//		if(balance_infantry->ChassisMode != CHASSIS_ZERO_FORCE){
			if(balance_infantry->phi_left[0] - balance_infantry->last_phi_l<-PI)
			{
				balance_infantry->count_phi_l--;
			}
			else if(balance_infantry->phi_left[0] - balance_infantry->last_phi_l>PI)
			{
				balance_infantry->count_phi_l++;
			}
			if(balance_infantry->phi_right[0] - balance_infantry->last_phi_r<-PI)
			{
				balance_infantry->count_phi_r--;
			}
			else if(balance_infantry->phi_right[0] - balance_infantry->last_phi_r>PI)
			{
				balance_infantry->count_phi_r++;
			}
			//获取绝对角度
			get_abs_angle(balance_infantry);
//		}
//		else{
//				balance_infantry->count_phi_l=0;
//				balance_infantry->count_phi_r=0;
//		}
		balance_infantry->last_phi_l = balance_infantry->phi_left[0];
		balance_infantry->last_phi_r = balance_infantry->phi_right[0];
	}

/**
 * @brief        获取腿部thera角
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
void getTheta(BalanceInfantry *balance_infantry)
{

	/* 左腿腿部thera角 */
	balance_infantry->theta_left = (PI/2 - balance_infantry->phi_left[0] - balance_infantry->state_vector_l[4]);
	/* 通过vmc矩阵解算theta_dot */
	balance_infantry->theta_left_w = balance_infantry->J_l[0][0] * balance_infantry->DM8009_Motor[0].para.vel + balance_infantry->J_l[1][0] * balance_infantry->DM8009_Motor[1].para.vel;
	balance_infantry->state_vector_l[0] = balance_infantry->theta_left;
	balance_infantry->state_vector_l[1] = -balance_infantry->theta_left_w ;

	/* 右腿腿部thera角 */
	balance_infantry->theta_right = (PI/2 - balance_infantry->phi_right[0] - balance_infantry->state_vector_r[4]);
	//通过vmc矩阵解算theta_dot
	balance_infantry->theta_right_w = -balance_infantry->J_r[0][0] * balance_infantry->DM8009_Motor[3].para.vel - balance_infantry->J_r[1][0] * balance_infantry->DM8009_Motor[2].para.vel;
	balance_infantry->state_vector_r[0] = balance_infantry->theta_right;
	balance_infantry->state_vector_r[1] = -balance_infantry->theta_right_w ;
	if(balance_infantry->stand_done_flag == 2)
	{
	balance_infantry->state_vector_l[0] = normalize_angle( balance_infantry->theta_left);
	balance_infantry->state_vector_r[0] = normalize_angle( balance_infantry->theta_right);
	}
	static uint16_t zero_theta_count;
//	//计算归零的pid 
	if(balance_infantry->stand_done_flag && balance_infantry->ChassisMode != CHASSIS_ZERO_FORCE)
	{

		if(zero_theta_count>500)
		{
			zero_theta_count = 501;
			balance_infantry->extra_torque_l = PID_calc(&balance_infantry->zero_thera_pid_l,balance_infantry->state_vector_l[0],balance_infantry->state_vector_r[0]);
//			balance_infantry->extra_torque_r = PID_calc(&balance_infantry->zero_thera_pid_r,balance_infantry->state_vector_r[0],balance_infantry->state_vector_l[0]);
		}
		else
		{
			zero_theta_count++;
		}
	}
	else
	{
		zero_theta_count = 0;
		balance_infantry->zero_thera_pid_l.out = 0.0;
		balance_infantry->zero_thera_pid_r.out = 0.0;
		balance_infantry->extra_torque_l = 0.0;
	}
	if(balance_infantry->ChassisMode != CHASSIS_ZERO_FORCE){
		if(balance_infantry->state_vector_l[0] - balance_infantry->last_theta_l<-PI)
		{
			balance_infantry->count_theta_l++;
		}
		else if(balance_infantry->state_vector_l[0] - balance_infantry->last_theta_l>PI)
		{
			balance_infantry->count_theta_l--;
		}
		if(balance_infantry->state_vector_r[0] - balance_infantry->last_theta_r<-PI)
		{
			balance_infantry->count_theta_r++;
		}
		else if(balance_infantry->state_vector_r[0] - balance_infantry->last_theta_r>PI)
		{
			balance_infantry->count_theta_r--;
		}
	}
	else{
			balance_infantry->count_theta_r=0;
			balance_infantry->count_theta_l=0;
	}
	balance_infantry->last_theta_l = balance_infantry->state_vector_l[0];
	balance_infantry->last_theta_r = balance_infantry->state_vector_r[0];
}

/**
 * @brief        Roll轴控制
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
void roll_control(BalanceInfantry *balance_infantry)
{
	/* ROll轴补偿，用于保持云台Roll轴水平，两腿分别叠加PID的输出到F即可 */
	float output;
//	if(balance_infantry->FN_l < 30.0f && balance_infantry->FN_r < 30.0f){
//		balance_infantry->roll_control_pid.max_out = 20;
//	}else{
//		balance_infantry->roll_control_pid.max_out = 50;
//	}
	if(balance_infantry->jump_flag == 0 && balance_infantry->stand_done_flag == 2)// && balance_infantry->is_get_enough_force == FALSE
	{
		output = PID_calc(&balance_infantry->roll_control_pid, balance_infantry->Ins_Angle[1]+0.03f, balance_infantry->roll_set);
		balance_infantry->extra_leg_force_l -= output;
		balance_infantry->extra_leg_force_r += output;
		balance_infantry->extra_leg_force_l += PID_calc(&balance_infantry->leg_syn_pid,balance_infantry->L_left[0],balance_infantry->L_right[0]);
		balance_infantry->extra_leg_force_r += PID_calc(&balance_infantry->leg_syn_pid,balance_infantry->L_right[0],balance_infantry->L_left[0]);
	}
}
/**
 * @brief        轮腿跳跃
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
//static float last_filtered_val,filtered_val;
//	filtered_val = 0.01f * (*(balance_infantry->distance)) + 0.99f * last_filtered_val;
//	if(filtered_val < 0)
//	{
//		filtered_val = -filtered_val;
//	}
float distance_jump;
void jump(BalanceInfantry *balance_infantry)
{
	const float *current_distance;
	distance_jump = Get_distance();
//  int16_t current_distance = *(balance_infantry->distance);
	//距离触发条件：距离≤阈值 + 未处于跳跃状态（避免重复触发） + 跳跃标志位未被占用
	if (distance_jump <= AUTO_JUMP_DISTANCE_THRESHOLD && 
			balance_infantry->jump_status == 0 && 
			balance_infantry->jump_flag != 0 &&
			balance_infantry->jump_done_flag ==0)
	{
			balance_infantry->jump_status = 1;  // 置位跳跃触发
			balance_infantry->jump_flag = 0;
		}

	if(balance_infantry->jump_flag && balance_infantry->jump_shirink_flag)
	{
		
		balance_infantry->target_L_left = 0.14f;
		balance_infantry->target_L_right = 0.14f;
		balance_infantry->jump_shirink_flag = ~balance_infantry->jump_shirink_flag;
	}

			if(balance_infantry->ChassisRC->rc.ch[4]>=400)
			{
//				balance_infantry->jump_status=1;
//				balance_infantry->jump_flag = 0;
			}
			else if(balance_infantry->jump_status == 1)
			{
				balance_infantry->target_L_left = 0.37f;
				balance_infantry->target_L_right = 0.37f;
				balance_infantry->qiankui = 210;//1.24-200 
				balance_infantry->jump_time++;
				if(balance_infantry->jump_time >= 90)
				{
					balance_infantry->jump_status=2;
					balance_infantry->jump_time=0;
					//清除位移
					balance_infantry->x_filter = 0;
					balance_infantry->target_vector[2] = 0;
				}
			}
			else if(balance_infantry->jump_status == 2)
			{
				balance_infantry->target_L_left = 0.1f;
				balance_infantry->target_L_right = 0.1f;
				balance_infantry->qiankui = -75;//1.24- -60
				balance_infantry->jump_time++;
				if(balance_infantry->jump_time >= 85)
				{
					balance_infantry->jump_status=3;
					balance_infantry->jump_time=0;
					balance_infantry->target_vector[2] = 0;
					balance_infantry->x_filter = 0;
				}
			}
			else if(balance_infantry->jump_status == 3)
			{
				balance_infantry->target_L_left = 0.27f;
				balance_infantry->target_L_right = 0.27f;
//				balance_infantry->leg_speed_pid_l.Kp=60;
//				balance_infantry->leg_speed_pid_r.Kp=60;
				balance_infantry->qiankui=60;//25
				balance_infantry->jump_time=0;
				balance_infantry->jump_status=0;
				balance_infantry->jump_flag=0;
				//设置往前的位移
//				balance_infantry->target_vector[2] = -0.5;
				balance_infantry->jump_done_flag = 1;
			}
}
/**
 * @brief        LQR计算得到K矩阵
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
void LQR_Solve(BalanceInfantry *balance_infantry)
{
	if(balance_infantry->stand_done_flag != 0){
		//LQR计算左腿的虚拟力和虚拟力矩
//		balance_infantry->virtual_torque_l[1] = 0;
//		balance_infantry->virtual_torque_r[1] = 0;
		for (uint8_t i = 0; i < 2; i++)
		{
			balance_infantry->virtual_torque_l[i] = 0;
			for (uint8_t j = 0; j < 6; j++)
			{
					balance_infantry->virtual_torque_l[i] += balance_infantry->K_l[i][j] * (balance_infantry->target_vector[j] - balance_infantry->state_vector_l[j]);
			}
		}
//		balance_infantry->virtual_torque_l[1] += balance_infantry->K_l[1][0] * (balance_infantry->target_vector[0] - balance_infantry->state_vector_l[0]);
//		balance_infantry->virtual_torque_l[1] += balance_infantry->K_l[1][1] * (balance_infantry->target_vector[1] - balance_infantry->state_vector_l[1]);
//		balance_infantry->virtual_torque_l[1] += balance_infantry->K_l[1][2] * (-balance_infantry->target_vector[2] + balance_infantry->state_vector_l[2]);
//		balance_infantry->virtual_torque_l[1] += balance_infantry->K_l[1][3] * (-balance_infantry->target_vector[3] + balance_infantry->state_vector_l[3]);
//		balance_infantry->virtual_torque_l[1] += balance_infantry->K_l[1][4] * (-balance_infantry->state_vector_l[4] + balance_infantry->target_vector[4]);
//		balance_infantry->virtual_torque_l[1] += balance_infantry->K_l[1][5] * (-balance_infantry->state_vector_l[5] + balance_infantry->target_vector[5]);
		//LQR计算右腿的虚拟力和虚拟力矩
		for (uint8_t i = 0; i < 2; i++)
		{
			balance_infantry->virtual_torque_r[i] = 0;
			for (uint8_t j = 0; j < 6; j++)
			{
					balance_infantry->virtual_torque_r[i] += balance_infantry->K_r[i][j] * (balance_infantry->target_vector[j] - balance_infantry->state_vector_r[j]);
			}
		}
//		balance_infantry->virtual_torque_r[1] += balance_infantry->K_r[1][0] * (balance_infantry->target_vector[0] - balance_infantry->state_vector_r[0]);
//		balance_infantry->virtual_torque_r[1] += balance_infantry->K_r[1][1] * (balance_infantry->target_vector[1] - balance_infantry->state_vector_r[1]);
//		balance_infantry->virtual_torque_r[1] += balance_infantry->K_r[1][2] * (-balance_infantry->target_vector[2] + balance_infantry->state_vector_r[2]);
//		balance_infantry->virtual_torque_r[1] += balance_infantry->K_r[1][3] * (-balance_infantry->target_vector[3] + balance_infantry->state_vector_r[3]);
//		balance_infantry->virtual_torque_r[1] += balance_infantry->K_r[1][4] * (-balance_infantry->state_vector_r[4] + balance_infantry->target_vector[4]);
//		balance_infantry->virtual_torque_r[1] += balance_infantry->K_r[1][5] * (-balance_infantry->state_vector_r[5] + balance_infantry->target_vector[5]);

	}
}

/**
 * @brief        VMC解算腿长速度
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
void get_leg_speed(BalanceInfantry *balance_infantry)
{
	/* 往上 phi1变大 phi4变小 leg_speed为负值 */
	/* 左腿速度求解 */
	float sigma, phi0_v;//leg_speed, 
	sigma = arm_sin_f32(balance_infantry->phi_left[2] - balance_infantry->phi_left[3]);
	phi0_v = balance_infantry->L_left[1] * arm_sin_f32(balance_infantry->phi_left[1] - balance_infantry->phi_left[2]) * arm_sin_f32(balance_infantry->phi_left[3]) / sigma * balance_infantry->DM8009_Motor[1].para.vel 
				+ balance_infantry->L_left[4] * arm_sin_f32(balance_infantry->phi_left[3] - balance_infantry->phi_left[4]) * arm_sin_f32(balance_infantry->phi_left[2]) / sigma * balance_infantry->DM8009_Motor[0].para.vel;
	balance_infantry->leg_speed_l = -balance_infantry->L_left[1] * arm_sin_f32(balance_infantry->phi_left[1] - balance_infantry->phi_left[2]) * arm_cos_f32(balance_infantry->phi_left[3]) / sigma * balance_infantry->DM8009_Motor[1].para.vel 
				- balance_infantry->L_left[4] * arm_sin_f32(balance_infantry->phi_left[3] - balance_infantry->phi_left[4]) * arm_cos_f32(balance_infantry->phi_left[2]) / sigma * balance_infantry->DM8009_Motor[0].para.vel;
	
	PID_calc(&balance_infantry->leg_speed_pid_l, balance_infantry->leg_speed_l, 0);
	balance_infantry->extra_leg_speed_force_l = balance_infantry->leg_speed_pid_l.out;
	
	/* 右腿速度求解 */
	sigma = arm_sin_f32(balance_infantry->phi_right[2] - balance_infantry->phi_right[3]);
	phi0_v = -balance_infantry->L_right[1] * arm_sin_f32(balance_infantry->phi_right[1] - balance_infantry->phi_right[2]) * arm_sin_f32(balance_infantry->phi_right[3]) / sigma * balance_infantry->DM8009_Motor[2].para.vel 
				- balance_infantry->L_right[4] * arm_sin_f32(balance_infantry->phi_right[3] - balance_infantry->phi_right[4]) * arm_sin_f32(balance_infantry->phi_right[2]) / sigma * balance_infantry->DM8009_Motor[3].para.vel;
	balance_infantry->leg_speed_r = balance_infantry->L_right[1] * arm_sin_f32(balance_infantry->phi_right[1] - balance_infantry->phi_right[2]) * arm_cos_f32(balance_infantry->phi_right[3]) / sigma * balance_infantry->DM8009_Motor[2].para.vel 
				+ balance_infantry->L_right[4] * arm_sin_f32(balance_infantry->phi_right[3] - balance_infantry->phi_right[4]) * arm_cos_f32(balance_infantry->phi_right[2]) / sigma * balance_infantry->DM8009_Motor[3].para.vel;
	
	PID_calc(&balance_infantry->leg_speed_pid_r, balance_infantry->leg_speed_r, 0);
	balance_infantry->extra_leg_speed_force_r = balance_infantry->leg_speed_pid_r.out;

}
/**
 * @brief        VMC解算
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
void VMC_Solve(BalanceInfantry *balance_infantry)
{
	/* 计算左腿长速度 */
	balance_infantry->leg_speed_l = balance_infantry->J_l[0][1] * balance_infantry->DM8009_Motor[1].para.vel + balance_infantry->J_l[1][1] * balance_infantry->DM8009_Motor[0].para.vel;
	PID_calc(&balance_infantry->leg_speed_pid_l, balance_infantry->leg_speed_l, 0);
	balance_infantry->extra_leg_speed_force_l = balance_infantry->leg_speed_pid_l.out;

	/* 轮毂电机的力矩输出及虚拟力矩的限幅 (符号待确定) */
		if(balance_infantry->is_get_enough_force_l == TRUE && count >=200)
	{
//		balance_infantry->extra_leg_force_l -= balance_infantry->F_Gas_left;
		balance_infantry->output_torque_l[0]=0;
		balance_infantry->Left_current=0;
		balance_infantry->x_filter = 0;
		balance_infantry->target_vector[0] = 0.0f;
		if(balance_infantry->state_vector_l[3] > 0){
			balance_infantry->target_vector[0] = 0.0f;//往后走的补偿
			balance_infantry->target_vector[2] = 0.325f;
		}else{
			balance_infantry->target_vector[0] = 0.0f;//往前走的补偿0.08
			balance_infantry->target_vector[2] = -1.25f;
		}
		
		balance_infantry->virtual_torque_l[1]=balance_infantry->K_l[1][0] * (balance_infantry->target_vector[0] - balance_infantry->state_vector_l[0])+balance_infantry->K_l[1][1] * (balance_infantry->target_vector[1] - balance_infantry->state_vector_l[1]);
		balance_infantry->virtual_torque_l[2] = LIMIT_MAX_MIN((-balance_infantry->extra_leg_speed_force_l+balance_infantry->extra_leg_force_l), MAX_VMC_FORCE, MIN_VMC_FORCE);
	}
	//起立与上台阶需要pd角度，所以不使用lqr计算的髋虚拟力矩
	else if(balance_infantry->stand_done_flag == 0 || balance_infantry->stair_status >2|| balance_infantry->Self_resuce_flag_X == 1)// 
	{
		balance_infantry->extra_leg_force_l -= balance_infantry->F_Gas_left;
		balance_infantry->virtual_torque_l[2] = LIMIT_MAX_MIN((-balance_infantry->extra_leg_speed_force_l+balance_infantry->extra_leg_force_l), MAX_VMC_FORCE, MIN_VMC_FORCE);
	}
	else{
		if(balance_infantry->jump_status != 1){
			balance_infantry->extra_leg_force_l -= balance_infantry->F_Gas_left;
		}
		//balance_infantry->target_vector[0] = 0.0f;
		balance_infantry->output_torque_l[0] = balance_infantry->virtual_torque_l[0];
		balance_infantry->virtual_torque_l[1] = LIMIT_MAX_MIN((balance_infantry->virtual_torque_l[1] + balance_infantry->extra_torque_l ), MAX_VMC_TORQUE, MIN_VMC_TORQUE);//- balance_infantry->MPC_delta_l
		balance_infantry->virtual_torque_l[2] = LIMIT_MAX_MIN((-balance_infantry->extra_leg_speed_force_l+balance_infantry->extra_leg_force_l), MAX_VMC_FORCE, MIN_VMC_FORCE);
	}

	//J00*Tp+J01*F0 0.1
	balance_infantry->output_torque_l[1] = balance_infantry->J_l[0][0] * balance_infantry->virtual_torque_l[1] + balance_infantry->J_l[0][1] * balance_infantry->virtual_torque_l[2];
	balance_infantry->output_torque_l[2] = balance_infantry->J_l[1][0] * balance_infantry->virtual_torque_l[1] + balance_infantry->J_l[1][1] * balance_infantry->virtual_torque_l[2];

	/* ???VMC?? */

	/* 计算右腿长速度 */
	balance_infantry->leg_speed_r = -balance_infantry->J_r[0][1] * balance_infantry->DM8009_Motor[2].para.vel - balance_infantry->J_r[1][1] * balance_infantry->DM8009_Motor[3].para.vel;
	PID_calc(&balance_infantry->leg_speed_pid_r, balance_infantry->leg_speed_r, 0);
	balance_infantry->extra_leg_speed_force_r = balance_infantry->leg_speed_pid_r.out;

	/* 轮毂电机的力矩输出及虚拟力矩的限幅 */
	if(balance_infantry->is_get_enough_force_r == TRUE && count >=200)
	{
		balance_infantry->output_torque_r[0]=0;
		balance_infantry->Right_current=0;
//		balance_infantry->extra_leg_force_r -= balance_infantry->F_Gas_right;
		balance_infantry->virtual_torque_r[1]=balance_infantry->K_r[1][0] * (balance_infantry->target_vector[0] - balance_infantry->state_vector_r[0])+balance_infantry->K_r[1][1] * (balance_infantry->target_vector[1] - balance_infantry->state_vector_r[1]);
		balance_infantry->virtual_torque_r[2] = LIMIT_MAX_MIN((-balance_infantry->extra_leg_speed_force_r +balance_infantry->extra_leg_force_r), MAX_VMC_FORCE, MIN_VMC_FORCE);
	}
	//起立与上台阶需要pd角度，所以不使用lqr计算的髋虚拟力矩
	else if(balance_infantry->stand_done_flag == 0 || balance_infantry->stair_status >2|| balance_infantry->Self_resuce_flag_X == 1)// 
	{
		balance_infantry->extra_leg_force_r -= balance_infantry->F_Gas_right;
		balance_infantry->virtual_torque_r[2] = LIMIT_MAX_MIN((-balance_infantry->extra_leg_speed_force_r +balance_infantry->extra_leg_force_r), MAX_VMC_FORCE, MIN_VMC_FORCE);
	}
	else{//正常
		if(balance_infantry->jump_status != 1 && balance_infantry->jump_flag == 0){
			balance_infantry->extra_leg_force_r -= balance_infantry->F_Gas_right;
		}
		balance_infantry->output_torque_r[0] = balance_infantry->virtual_torque_r[0];
		balance_infantry->virtual_torque_r[1] = LIMIT_MAX_MIN((balance_infantry->virtual_torque_r[1] - balance_infantry->extra_torque_l ), MAX_VMC_TORQUE, MIN_VMC_TORQUE);//- balance_infantry->MPC_delta_r
		balance_infantry->virtual_torque_r[2] = LIMIT_MAX_MIN((-balance_infantry->extra_leg_speed_force_r +balance_infantry->extra_leg_force_r), MAX_VMC_FORCE, MIN_VMC_FORCE);
	}

	balance_infantry->output_torque_r[1] = balance_infantry->J_r[0][0] * balance_infantry->virtual_torque_r[1] + balance_infantry->J_r[0][1] * balance_infantry->virtual_torque_r[2];
	balance_infantry->output_torque_r[2] = balance_infantry->J_r[1][0] * balance_infantry->virtual_torque_r[1] + balance_infantry->J_r[1][1] * balance_infantry->virtual_torque_r[2];

}
/**
 * @brief        实时更新增益矩阵K
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
void updataK(BalanceInfantry *balance_infantry)
{
	/* 左腿K */
	float pow2 = balance_infantry->L_left[0] * balance_infantry->L_left[0];
	float pow3 = balance_infantry->L_left[0] * balance_infantry->L_left[0] * balance_infantry->L_left[0];
	for (uint8_t i = 0; i < 2; i++)
	{
		for (uint8_t j = 0; j < 6; j++)
		{
			balance_infantry->K_l[i][j] =
				balance_infantry->K_coef[i * 6 + j][3] + balance_infantry->K_coef[i * 6 + j][2] * balance_infantry->L_left[0] + balance_infantry->K_coef[i * 6 + j][1] * pow2 + balance_infantry->K_coef[i * 6 + j][0] * pow3;
		}
	}
	//离地时
	if(balance_infantry->is_get_enough_force_l)
	{
		for (uint8_t i = 0; i < 2; i++)
		{
			for (uint8_t j = 0; j < 6; j++)
			{
				balance_infantry->K_l[i][j] =
					balance_infantry->K_coef_off_ground[i * 6 + j][3] + balance_infantry->K_coef_off_ground[i * 6 + j][2] * balance_infantry->L_left[0] + balance_infantry->K_coef_off_ground[i * 6 + j][1] * pow2 + balance_infantry->K_coef_off_ground[i * 6 + j][0] * pow3;
			}
		}
	}
	for (uint8_t i = 0; i < 6; i++)
	{
		/* code */
		balance_infantry->K_MPC_l[i] =
			balance_infantry->K_MPC_coef[i][3] + balance_infantry->K_MPC_coef[i][2] * balance_infantry->L_left[0] + balance_infantry->K_MPC_coef[i][1] * pow2 + balance_infantry->K_MPC_coef[i][0] * pow3;
	}
	
	/* 右腿K */
	pow2 = balance_infantry->L_right[0] * balance_infantry->L_right[0];
	pow3 = balance_infantry->L_right[0] * balance_infantry->L_right[0] * balance_infantry->L_right[0];
	for (uint8_t i = 0; i < 2; i++)
	{
		for (uint8_t j = 0; j < 6; j++)
		{
			balance_infantry->K_r[i][j] =
				balance_infantry->K_coef[i * 6 + j][3] + balance_infantry->K_coef[i * 6 + j][2] * balance_infantry->L_right[0] + balance_infantry->K_coef[i * 6 + j][1] * pow2 + balance_infantry->K_coef[i * 6 + j][0] * pow3;
		}
	}
	//离地时
	if(balance_infantry->is_get_enough_force_r)
	{
		for (uint8_t i = 0; i < 2; i++)
		{
			for (uint8_t j = 0; j < 6; j++)
			{
				balance_infantry->K_r[i][j] =
					balance_infantry->K_coef_off_ground[i * 6 + j][3] + balance_infantry->K_coef_off_ground[i * 6 + j][2] * balance_infantry->L_right[0] + balance_infantry->K_coef_off_ground[i * 6 + j][1] * pow2 + balance_infantry->K_coef_off_ground[i * 6 + j][0] * pow3;
			}
		}
	}
//
	for(uint8_t i = 0; i < 6; i++)
	{
		/* code */
		balance_infantry->K_MPC_r[i] =
			balance_infantry->K_MPC_coef[i][3] + balance_infantry->K_MPC_coef[i][2] * balance_infantry->L_right[0] + balance_infantry->K_MPC_coef[i][1] * pow2 + balance_infantry->K_MPC_coef[i][0] * pow3;
	}
}

/**
 * @brief        实时更新MPC增量部分
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
float delta_l;
float delta_r;
void updataMPC(BalanceInfantry *balance_infantry){
	
	// 对状态变化量进行滤波
	float alpha = 0.3f; // 可调整

	for (uint8_t i = 0; i < 6; i++) {
    delta_l = balance_infantry->state_vector_l[i] - balance_infantry->last_state_vector_l[i];
    delta_r = balance_infantry->state_vector_r[i] - balance_infantry->last_state_vector_r[i];
    
    // 滤波状态变化量
    balance_infantry->filtered_delta_l[i] = 
        alpha * delta_l + (1 - alpha) * balance_infantry->last_filtered_delta_l[i];
    balance_infantry->filtered_delta_r[i] = 
        alpha * delta_r + (1 - alpha) * balance_infantry->last_filtered_delta_r[i];        
    // 更新历史滤波值
    balance_infantry->last_filtered_delta_l[i] = balance_infantry->filtered_delta_l[i];
    balance_infantry->last_filtered_delta_r[i] = balance_infantry->filtered_delta_r[i];
	}
	/* 左腿MPC增量部分 */
	for (uint8_t i = 0; i < 6; i+=2)
	{
		/* code */
		balance_infantry->MPC_delta_l += 
			(balance_infantry->K_MPC_l[i] *balance_infantry->filtered_delta_l[i]);
	}
//	balance_infantry->MPC_delta_l += balance_infantry->K_MPC_l[3] * balance_infantry->last_filtered_delta_l[3];
	/* 右腿MPC增量部分 */
	for (uint8_t i = 0; i < 6; i+=2)
	{
		/* code */
		balance_infantry->MPC_delta_r +=
			(balance_infantry->K_MPC_r[i] *balance_infantry->filtered_delta_r[i]);
	}
//	balance_infantry->MPC_delta_r += balance_infantry->K_MPC_r[3] * balance_infantry->last_filtered_delta_r[3];
	for(uint8_t i=0;i<6;i++){
		balance_infantry->last_state_vector_l[i] = balance_infantry->state_vector_l[i];
	}
	for(uint8_t i=0;i<6;i++){
		balance_infantry->last_state_vector_r[i] = balance_infantry->state_vector_r[i];
	}
}

/**
 * @brief  梯形控制
 * @param  反馈；设定；加速度
 * @retval 输出值
 */
float ramp_control(float ref, float set, float accel)
{
    float ramp = LIMIT_MAX_MIN(accel, 1, 0) * (set - ref);
    return ref + ramp;
}

/**
 * @brief        腿长控制
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
void leg_control(BalanceInfantry *balance_infantry)
{	
	//stand_done以后加前馈
	if(balance_infantry->jump_flag == 0 && balance_infantry->stair_flag ==0 && balance_infantry->ChassisMode != CHASSIS_ZERO_FORCE && balance_infantry->stand_done_flag)
	{
		static uint16_t qiankui_count;
		if(fabs(balance_infantry->state_vector_l[0]) < 0.3 && fabs(balance_infantry->state_vector_r[0]) < 0.3 && balance_infantry->qiankui != G_Frontback)
		{
			qiankui_count++;
			if(qiankui_count > 100)//150
			{
				balance_infantry->qiankui = G_Frontback;
				balance_infantry->leg_control_pid_l.Kp = 300;//710
				balance_infantry->leg_control_pid_r.Kp = 300;
				balance_infantry->leg_control_pid_l.Kd = 250;
				balance_infantry->leg_control_pid_r.Kd = 250;
				balance_infantry->leg_speed_pid_l.Kp = 50;
				balance_infantry->leg_speed_pid_r.Kp = 50;
				balance_infantry->stand_done_flag = 2;
				qiankui_count = 0;
			}

		}
		else
		{
			qiankui_count = 0;					
		}

//		balance_infantry->target_L_left += balance_infantry->ChassisRC->rc.ch[3] * LEG_CONTROL_PROPORTION;
//		balance_infantry->target_L_right += balance_infantry->ChassisRC->rc.ch[3] * LEG_CONTROL_PROPORTION;
		
		if(balance_infantry->target_L_left <= LOWEST)
		{
				balance_infantry->target_L_left = LOWEST;
		}
		if(balance_infantry->target_L_left >= HIGAHEST)
		{
				balance_infantry->target_L_left = HIGAHEST;
		}
		if(balance_infantry->target_L_right <= LOWEST)
		{
				balance_infantry->target_L_right = LOWEST;
		}
		if(balance_infantry->target_L_right >= HIGAHEST)
		{
				balance_infantry->target_L_right = HIGAHEST;
		}
//		LIMIT_MAX_MIN(balance_infantry->target_L_left,0.24f,0.12f);
//		LIMIT_MAX_MIN(balance_infantry->target_L_right,0.24f,0.12f);
	}
	/* 腿长控制pid计算 */
	balance_infantry->extra_leg_force_l = PID_calc(&balance_infantry->leg_control_pid_l, balance_infantry->L_left[0], balance_infantry->target_L_left)+balance_infantry->qiankui/arm_cos_f32(balance_infantry->theta_left);
	balance_infantry->extra_leg_force_r = PID_calc(&balance_infantry->leg_control_pid_r, balance_infantry->L_right[0], balance_infantry->target_L_right)+balance_infantry->qiankui/arm_cos_f32(balance_infantry->theta_right);
}

/**
 * @brief        离地检测
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
	static float a[6];
extern ext_power_heat_data_t power_heat_data_t;
int ground_flag,ground_count;
void off_ground_detect(BalanceInfantry *balance_infantry)
{
	
//	update_leg_sensors(LEG_LEFT,balance_infantry->state_vector_l[0],balance_infantry->state_vector_l[1],balance_infantry->state_vector_l[2],balance_infantry->state_vector_l[3],balance_infantry->state_vector_l[4],balance_infantry->state_vector_l[5],balance_infantry->virtual_torque_l[2],balance_infantry->virtual_torque_l[1]);
//	update_leg_sensors(LEG_RIGHT,balance_infantry->state_vector_r[0],balance_infantry->state_vector_r[1],balance_infantry->state_vector_r[2],balance_infantry->state_vector_r[3],balance_infantry->state_vector_r[4],balance_infantry->state_vector_r[5],balance_infantry->virtual_torque_r[2],balance_infantry->virtual_torque_r[1]);
//	dual_leg_detection_task();
//	balance_infantry->FN_l = get_leg_ground_state(LEG_LEFT);
//	balance_infantry->FN_r = get_leg_ground_state(LEG_RIGHT);
	static float leg_speed_l_prev,leg_accel_l_filtered,leg_speed_r_prev,leg_accel_r_filtered;
	
		 // 2. 一阶差分求原始加速度（仅用当前帧和上一帧速度）
    float leg_accel_raw_l = (balance_infantry->leg_speed_l - leg_speed_l_prev) / 0.02f;
    
    // 4. 一阶低通滤波（平滑信号）
    leg_accel_l_filtered = 0.3f * leg_accel_raw_l 
                         + (1 - 0.3f) * leg_accel_l_filtered;
    
    // 5. 更新上一帧速度（为下一帧准备）
    leg_speed_l_prev = balance_infantry->leg_speed_l;
		
				 // 2. 一阶差分求原始加速度（仅用当前帧和上一帧速度）
    float leg_accel_raw_r = (balance_infantry->leg_speed_r - leg_speed_r_prev) / 0.02f;
    
    // 4. 一阶低通滤波（平滑信号）
    leg_accel_r_filtered = 0.3f * leg_accel_raw_r 
                         + (1 - 0.3f) * leg_accel_r_filtered;
    
    // 5. 更新上一帧速度（为下一帧准备）
    leg_speed_r_prev = balance_infantry->leg_speed_r;
		
		static float phi0_speed_l_prev,phi0_accel_l_filtered,phi0_speed_r_prev,phi0_accel_r_filtered;
	
		 // 2. 一阶差分求原始加速度（仅用当前帧和上一帧速度）
    float phi0_accel_raw_l = (balance_infantry->state_vector_l[1] - phi0_speed_l_prev) / 0.02f;
    
    // 4. 一阶低通滤波（平滑信号）
    phi0_accel_l_filtered = 0.3f * phi0_accel_raw_l 
                         + (1 - 0.3f) * phi0_accel_l_filtered;
    
    // 5. 更新上一帧速度（为下一帧准备）
    phi0_speed_l_prev = balance_infantry->state_vector_l[1];
		
				 // 2. 一阶差分求原始加速度（仅用当前帧和上一帧速度）
    float phi0_accel_raw_r = (balance_infantry->state_vector_r[1] - leg_speed_r_prev) / 0.02f;
    
    // 4. 一阶低通滤波（平滑信号）
    phi0_accel_r_filtered = 0.3f * phi0_accel_raw_r 
                         + (1 - 0.3f) * phi0_accel_r_filtered;
    
    // 5. 更新上一帧速度（为下一帧准备）
    phi0_speed_r_prev = balance_infantry->state_vector_r[1];

    //[0]元素为Tp，[1]元素为F
    balance_infantry->wheel_f_torque_feedback_l[0] = 
        -balance_infantry->J_inv_left[0][0] * balance_infantry->DM8009_Motor[0].para.tor - balance_infantry->J_inv_left[0][1] * balance_infantry->DM8009_Motor[1].para.tor;
    balance_infantry->wheel_f_torque_feedback_l[1] = //--
        balance_infantry->J_inv_left[1][0] * balance_infantry->DM8009_Motor[0].para.tor + balance_infantry->J_inv_left[1][1] * balance_infantry->DM8009_Motor[1].para.tor;

    balance_infantry->wheel_f_torque_feedback_r[0] = 
        balance_infantry->J_inv_right[0][0] * balance_infantry->DM8009_Motor[3].para.tor + balance_infantry->J_inv_right[0][1] * balance_infantry->DM8009_Motor[2].para.tor;
    balance_infantry->wheel_f_torque_feedback_r[1] = 
				balance_infantry->J_inv_right[1][0] * balance_infantry->DM8009_Motor[3].para.tor + balance_infantry->J_inv_right[1][1] * balance_infantry->DM8009_Motor[2].para.tor;
			//??????????
//		balance_infantry->wheel_f_torque_feedback_l[1] += balance_infantry->F_Gas_left;
//		balance_infantry->wheel_f_torque_feedback_r[1] -= balance_infantry->F_Gas_right;
    /* 先不加输出一阶低通滤波(待加入机体加速度) */

//	

//		float F = -balance_infantry->virtual_torque_l[1] -balance_infantry->virtual_torque_r[1];
//		float Tp = balance_infantry->virtual_torque_l[2] + balance_infantry->virtual_torque_r[2];
//	
//	
//		float P = F * arm_cos_f32(balance_infantry->theta_left) + Tp * arm_sin_f32(balance_infantry->theta_right) / ((balance_infantry->L_right[0] + balance_infantry->L_left[0])/2);
//	
//		float ddz = balance_infantry->Motion_Accel[2] - (leg_accel_l_filtered + leg_accel_r_filtered)* arm_cos_f32(balance_infantry->theta_left)/2.0 + (balance_infantry->leg_speed_r + balance_infantry->leg_speed_l)*(balance_infantry->state_vector_l[1]+balance_infantry->state_vector_r[1])*arm_sin_f32(balance_infantry->theta_left)/2;
//	
//		balance_infantry->FN_l = P +0.5*9.81+0.5 * ddz;
    static float FN_l_row,FN_l_prev,FN_r_row,FN_r_prev;
		FN_l_row	=
        (balance_infantry->extra_leg_speed_force_l+balance_infantry->extra_leg_force_l) * arm_cos_f32(balance_infantry->theta_left)
				+ balance_infantry->wheel_f_torque_feedback_l[0] * arm_sin_f32(balance_infantry->theta_left)/ balance_infantry->L_left[0] + WHEEL_WEIGHT + 0.5f * (balance_infantry->L_left[0]*balance_infantry->state_vector_l[1] * balance_infantry->state_vector_l[1] * arm_cos_f32(balance_infantry->state_vector_l[0]) //-leg_accel_l_filtered * arm_cos_f32(balance_infantry->state_vector_l[0])
				 + 2 * balance_infantry->state_vector_l[1] * arm_sin_f32(balance_infantry->state_vector_l[0]) * balance_infantry->leg_speed_l + balance_infantry->L_left[0] * phi0_accel_l_filtered * arm_sin_f32(balance_infantry->state_vector_l[0]));
	    // 4. 一阶低通滤波（平滑信号）
    balance_infantry->FN_l = 0.1f * FN_l_row 
                         + (1 - 0.1f) * FN_l_prev;
    
		FN_l_prev = FN_l_row;
		
    FN_r_row =
        (-balance_infantry->extra_leg_speed_force_r +balance_infantry->extra_leg_force_r) * arm_cos_f32(balance_infantry->theta_right)
				+ balance_infantry->wheel_f_torque_feedback_r[0] * arm_sin_f32(balance_infantry->theta_right)/ balance_infantry->L_right[0] + WHEEL_WEIGHT + 0.5f * (balance_infantry->L_right[0]*balance_infantry->state_vector_r[1] * balance_infantry->state_vector_r[1] * arm_cos_f32(balance_infantry->state_vector_r[0]) //- leg_accel_r_filtered * arm_cos_f32(balance_infantry->state_vector_r[0])
				 + 2 * balance_infantry->state_vector_r[1] * arm_sin_f32(balance_infantry->state_vector_r[0]) * balance_infantry->leg_speed_r + balance_infantry->L_right[0] * phi0_accel_r_filtered * arm_sin_f32(balance_infantry->state_vector_r[0]));
		balance_infantry->FN_r = 0.1f * FN_r_row 
                         + (1 - 0.1f) * FN_r_prev;
		FN_r_prev = FN_r_row;

//未加+ balance_infantry->extra_torque_l+ balance_infantry->extra_torque_r
//balance_infantry->FN_l = 
//		(balance_infantry->extra_leg_speed_force_l+balance_infantry->extra_leg_force_l) * arm_cos_f32(balance_infantry->theta_left) +
//		(balance_infantry->virtual_torque_l[1]) * arm_sin_f32(balance_infantry->theta_left)/ balance_infantry->L_left[0] + WHEEL_WEIGHT;
//	balance_infantry->FN_r = 
//		(balance_infantry->extra_leg_speed_force_r+balance_infantry->extra_leg_force_r) * arm_cos_f32(balance_infantry->theta_right) +
//		(balance_infantry->virtual_torque_r[1]) * arm_sin_f32(balance_infantry->theta_right)/ balance_infantry->L_right[0] + WHEEL_WEIGHT;
	
	
//	balance_infantry->FN_l = 
//		(balance_infantry->wheel_f_torque_feedback_l[0]) * arm_cos_f32(balance_infantry->theta_left) +
//		(balance_infantry->wheel_f_torque_feedback_l[1]) * arm_sin_f32(balance_infantry->theta_left)/ balance_infantry->L_left[0] + WHEEL_WEIGHT;
//	balance_infantry->FN_r = 
//		(balance_infantry->wheel_f_torque_feedback_r[0]) * arm_cos_f32(balance_infantry->theta_right) -
//		(balance_infantry->wheel_f_torque_feedback_l[1]) * arm_sin_f32(balance_infantry->theta_right)/ balance_infantry->L_right[0] + WHEEL_WEIGHT;


//	balance_infantry->FN_l =
//		balance_infantry->wheel_f_torque_feedback_l[1] * arm_cos_f32(balance_infantry->theta_left) -
//		balance_infantry->wheel_f_torque_feedback_l[0] * arm_cos_f32(balance_infantry->theta_left) / balance_infantry->L_left[0] +
//		WHEEL_WEIGHT * (GRAVITY - balance_infantry->L_theta_a_l);
//	balance_infantry->FN_r =
//		-balance_infantry->wheel_f_torque_feedback_r[1] * arm_cos_f32(balance_infantry->theta_right) +
//		balance_infantry->wheel_f_torque_feedback_r[0] * arm_cos_f32(balance_infantry->theta_right) / balance_infantry->L_right[0] +
//		WHEEL_WEIGHT * (GRAVITY - balance_infantry->L_theta_a_r);
	/* 后续可加入低通滤波 */
//	if(balance_infantry->FN_l > 10.0f)
//	{
//		balance_infantry->is_get_enough_force_l = FALSE;
//	}else{
//		balance_infantry->is_get_enough_force_l = TRUE;
//	}
//	if(balance_infantry->FN_r > 10.0f){
//		balance_infantry->is_get_enough_force_r = FALSE;
//	}else{
//		balance_infantry->is_get_enough_force_r = TRUE;
//	}
////////	if (balance_infantry->FN_l>10.0f && balance_infantry->FN_r>10.0f)
////////	{
////////		balance_infantry->is_get_enough_force = FALSE;
////////		count_off_ground=0;
////////	}
////////	else if ((balance_infantry->FN_l < 10.0f )&& (balance_infantry->FN_r < 10.0f) && balance_infantry->ChassisMode != CHASSIS_ZERO_FORCE && balance_infantry->stand_done_flag == 2)
////////	{
////////			balance_infantry->is_get_enough_force = TRUE;
////////			balance_infantry->x_filter = 0;
////////			balance_infantry->target_vector[2] = 0;
////////	}

	if (balance_infantry->FN_l > 15.0f && balance_infantry->FN_r > 15.0f)
	{
		balance_infantry->is_get_enough_force_l = FALSE;
		balance_infantry->is_get_enough_force_r = FALSE;
	}
	else if (balance_infantry->ChassisMode != CHASSIS_ZERO_FORCE && balance_infantry->stand_done_flag == 2)//(balance_infantry->FN_l < 15.0f )&& 
	{
			balance_infantry->is_get_enough_force_l = TRUE;
			balance_infantry->is_get_enough_force_r = TRUE;
//			balance_infantry->x_filter = 0;
//			balance_infantry->target_vector[2] = 0;
	}
//	if (balance_infantry->FN_r>15.0f)
//	{
//		balance_infantry->is_get_enough_force_r = FALSE;
//		count_off_ground=0;
//	}
//	else if ((balance_infantry->FN_r < 15.0f) && balance_infantry->ChassisMode != CHASSIS_ZERO_FORCE && balance_infantry->stand_done_flag == 2)
//	{
//			balance_infantry->is_get_enough_force_r = TRUE;
//			balance_infantry->x_filter = 0;
//			balance_infantry->target_vector[2] = 0;
//	}

	if(balance_infantry->jump_flag !=0 && balance_infantry->jump_status!=3)
	{
		balance_infantry->is_get_enough_force_l = FALSE;
		balance_infantry->is_get_enough_force_r = FALSE;
	}
	
	static uint16_t stair_count;
	//结束磕台阶时避免进入离地
	if(balance_infantry->last_stair_flag == 1 && balance_infantry->stair_flag == 0){//flag为1时为磕台阶
		stair_count=300;
			if(stair_count >= 0){
			stair_count--;
			balance_infantry->is_get_enough_force_l = FALSE;
			balance_infantry->is_get_enough_force_r = FALSE;
		}
	}

	// 磕台阶时
	if(balance_infantry->stair_flag){
		balance_infantry->is_get_enough_force_l = FALSE;
		balance_infantry->is_get_enough_force_r = FALSE;
	}
	
	if(balance_infantry->last_is_get_enough_force_state == TRUE && balance_infantry->is_get_enough_force_l == FALSE && balance_infantry->is_get_enough_force_r == FALSE)
	{
		ground_flag = 1;
//		balance_infantry->target_L_left = 0.17;
//		balance_infantry->target_L_right = 0.17;
	}
//	if(ground_flag == 1){
//		ground_count++;
//		balance_infantry->qiankui = 95;
//		balance_infantry->phi_L_pid.Kd = 500;
//		balance_infantry->phi_R_pid.Kd = 500;
//	}else{
//		balance_infantry->qiankui = G_Frontback;
//		balance_infantry->phi_L_pid.Kd = 0;
//		balance_infantry->phi_R_pid.Kd = 0;
//		ground_count = 0;
//	}
//	if(ground_count > 300){
//		ground_count = 0;
//		ground_flag = 0;
//	}
//	balance_infantry->last_is_get_enough_force_state = balance_infantry->is_get_enough_force;
}

/**
 * @brief        输出整理
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
extern float aver_v;	
extern float initial_total_power;
extern float k_yaw;
extern float k_speed;

extern 	float initial_give_power[2]; // initial power from PID calculation
void motor_output(BalanceInfantry *balance_infantry)
{
//	balance_infantry->output_current_l = balance_infantry->output_torque_l[0] * M3508TORQUEPROPORTION - balance_infantry->Left_current;//
//	
//	balance_infantry->output_current_r = balance_infantry->output_torque_r[0] * M3508TORQUEPROPORTION + balance_infantry->Right_current;//
//	chassis_power_control(balance_infantry);
//	ESO_Clac(&eso_speed_l,balance_infantry->state_vector_l[3],balance_infantry->output_current_l);
//	ESO_Clac(&eso_speed_r,balance_infantry->state_vector_r[3],balance_infantry->output_current_r);
	/* 左腿电机输出 计算重力矩 7*0.04* */
	if(balance_infantry->ChassisMode != CHASSIS_ZERO_FORCE && balance_infantry->stand_done_flag)// 
	{
		balance_infantry->output_current_l = ((balance_infantry->output_torque_l[0]) * M3508TORQUEPROPORTION+ balance_infantry->Left_current);// - eso_speed_l.z3 ;// 
	
		balance_infantry->output_current_r = ((balance_infantry->output_torque_r[0]) * M3508TORQUEPROPORTION- balance_infantry->Right_current);// - eso_speed_r.z3;// 
	}
	//	else if(balance_infantry->ChassisMode != CHASSIS_ZERO_FORCE && balance_infantry->stand_done_flag == 2)
//	{
//		balance_infantry->output_current_l = (balance_infantry->output_torque_l[0] - 0.25* arm_sin_f32(balance_infantry->state_vector_l[0])) * M3508TORQUEPROPORTION ;//+ balance_infantry->Left_current
//	
//		balance_infantry->output_current_r = (balance_infantry->output_torque_r[0] - 0.25* arm_sin_f32(balance_infantry->state_vector_r[0])) * M3508TORQUEPROPORTION ;//- balance_infantry->Right_current
//	}
	else
	{
		balance_infantry->output_current_l = 0;
		balance_infantry->output_current_r = 0;
	}
	if((balance_infantry->stair_status > 2  || balance_infantry->Self_resuce_flag_X == 1))//
	{
		balance_infantry->output_current_l = 0;
	
		balance_infantry->output_current_r = 0;
	}
//		a[0] = balance_infantry->self_resuce_state;
//		a[1] = balance_infantry->jump_shirink_flag;
//		a[2] = balance_infantry->target_L_left;
//		a[3] = balance_infantry->target_L_right;
//		a[4] = balance_infantry->stair_count;
//		a[5] = balance_infantry->state_vector_r[4];
	
		a[0] = balance_infantry->L_left[0];
		a[1] = balance_infantry->L_right[0];
		a[2] = balance_infantry->target_L_left;
		a[3] = balance_infantry->Ins_Angle[2];
		a[4] = balance_infantry->Ins_Angle[0];
		a[5] = balance_infantry->stair_status;
	
	vofa_justfloat_output(a,6,&huart1);

//	balance_infantry->output_current_l = 0;
//	
//	balance_infantry->output_current_r = 0;
	if(balance_infantry->stand_done_flag != 2){
		balance_infantry->output_current_l = limit_float(balance_infantry->output_current_l,8000,-8000);
		balance_infantry->output_current_r = limit_float(balance_infantry->output_current_r,8000,-8000);
	}
	else{
		balance_infantry->output_current_l = limit_float(balance_infantry->output_current_l,13000,-13000);
		balance_infantry->output_current_r = limit_float(balance_infantry->output_current_r,13000,-13000);
	}
	LIMIT_MAX_MIN(balance_infantry->output_current_l,12000,-12000);
	LIMIT_MAX_MIN(balance_infantry->output_current_r,12000,-12000);
}

/**
 * @brief 上台阶状态机：
 * 1) 外部通过设置 stair_flag=1 触发；
 * 2) 状态0: 保存原始腿长，抬高腿长到 stair_raise_len；
 * 3) 状态1: 监测俯仰角，当 Ins_Angle[1] 超过阈值(stair_pitch_threshold)认为已接到台阶，进入收腿状态；
 * 4) 状态2: 将腿长设为 stair_retract_len，等待稳定后恢复原始腿长并退出。
 */
void  stair_climb(BalanceInfantry *balance_infantry)
{
	balance_infantry->last_stair_flag = balance_infantry->stair_flag;
	if (balance_infantry->stair_flag == 0)
	{
		balance_infantry->target_vector[4] = -0.03f;
		balance_infantry->stair_status = 0;
		return;
	}
	if(balance_infantry->stair_flag != 0)
	{
		switch (balance_infantry->stair_status)
		{
		case 0: /* 准备：保存原始值并抬高腿长 */
			balance_infantry->stair_original_L_left = 0.14f;
			balance_infantry->stair_original_L_right = 0.14f;
			balance_infantry->target_L_left = 0.28f;
			balance_infantry->target_L_right =0.28f;
			balance_infantry->stair_time = 0;
			balance_infantry->target_vector[0] = 0.0f; // 后倾
			balance_infantry->target_vector[4] = 0.0f;//机体//0.15
			balance_infantry->count_theta_l = 0;//清零多圈
			balance_infantry->count_theta_r = 0;
			if(fabs(balance_infantry->target_L_left - balance_infantry->L_left[0])<0.05f && fabs(balance_infantry->target_L_right - balance_infantry->L_right[0])<0.05f)
			{
				balance_infantry->stair_count ++;
				if(balance_infantry->stair_count > 30)
				{
					balance_infantry->stair_count = 0;
					balance_infantry->stair_status ++;				
				}
			}
			else
			{
					balance_infantry->stair_count = 0;
			}
			break;

		case 1:
			balance_infantry->target_L_left = 0.37f;
			balance_infantry->target_L_right =0.37f;
			if(fabs(balance_infantry->target_L_left - balance_infantry->L_left[0])<0.05f && fabs(balance_infantry->target_L_right - balance_infantry->L_right[0])<0.05f)
			{
				balance_infantry->stair_count ++;
				if(balance_infantry->stair_count > 60)
				{
					balance_infantry->stair_count = 0;
					balance_infantry->stair_status ++;				
				}
			}
			else
			{
					balance_infantry->stair_count = 0;
			}
			break;
			
		case 2: /* 等待抬高完成并检测俯仰触发收腿 */
			 /* 若到位且检测到机体俯仰增加，触发收腿 */
			if (balance_infantry->state_vector_l[0] < STAIR_PITCH_THRESHOLD && balance_infantry->state_vector_r[0] < STAIR_PITCH_THRESHOLD)
			{
				balance_infantry->target_vector[0]=-0.5f* PI;
				balance_infantry->target_vector[1] = 0.0f;
				balance_infantry->phi_L_pid.Kp = 40;
				balance_infantry->phi_R_pid.Kp = 40;
				balance_infantry->phi_L_pid.Kd = 200;
				balance_infantry->phi_R_pid.Kd = 200;
				balance_infantry->phi_L_speed_pid.Kp = 10;
				balance_infantry->phi_R_speed_pid.Kp = 10;
				balance_infantry->stair_status ++;
			}
			break;

		case 3: /* 收腿中，等待稳定后恢复原腿长 */
	//					balance_infantry->stair_time++;

								balance_infantry->virtual_torque_r[1] = -PID_calc(&balance_infantry->phi_R_pid,
						balance_infantry->state_vector_r[0] + balance_infantry->count_theta_r * 2 * PI,balance_infantry->target_vector[0])
					- PID_calc(&balance_infantry->phi_R_speed_pid,balance_infantry->state_vector_r[1],balance_infantry->target_vector[1]);

		            balance_infantry->virtual_torque_l[1] = -PID_calc(&balance_infantry->phi_L_pid,
						balance_infantry->state_vector_l[0] + balance_infantry->count_theta_l * 2 * PI,balance_infantry->target_vector[0])
					- PID_calc(&balance_infantry->phi_L_speed_pid,balance_infantry->state_vector_l[1],balance_infantry->target_vector[1]);

		
//										balance_infantry->virtual_torque_r[1] = PID_calc(&balance_infantry->phi_R_pid,
//					normalize_angle(balance_infantry->state_vector_r[0]),balance_infantry->target_vector[0])
					
					
//							balance_infantry->virtual_torque_l[1] = PID_calc(&balance_infantry->phi_L_pid,
//					normalize_angle(balance_infantry->state_vector_l[0]),balance_infantry->target_vector[0]) 
			balance_infantry->stair_count ++;
			if (fabs(normalize_angle( balance_infantry->state_vector_l[0]) - 0.5f*PI) < 0.1f)
			{
				/* 恢复原始腿长 */
				balance_infantry->stair_status++;
				balance_infantry->stair_count = 0;
				balance_infantry->qiankui = 0;
			}
//			if(balance_infantry->stair_count > 200)
//			{
//				balance_infantry->stair_status = 3;
//				balance_infantry->stair_count = 0;
//				balance_infantry->phi_L_pid.max_out = 7;
//				balance_infantry->phi_R_pid.max_out = 7;
//				balance_infantry->leg_control_pid_l.Kp = 720;
//				balance_infantry->leg_control_pid_r.Kp = 720;
//				balance_infantry->leg_speed_pid_l.Kp = 50;
//				balance_infantry->leg_speed_pid_r.Kp = 50;
//			}
			break;
			
		case 4:
			
//										balance_infantry->virtual_torque_r[1] = PID_calc(&balance_infantry->phi_R_pid,
//						balance_infantry->state_vector_r[0] + balance_infantry->count_theta_r * 2 * PI,balance_infantry->target_vector[0])
//					+ PID_calc(&balance_infantry->phi_R_speed_pid,balance_infantry->state_vector_r[1],balance_infantry->target_vector[1]);

//		            balance_infantry->virtual_torque_l[1] =PID_calc(&balance_infantry->phi_L_pid,
//						balance_infantry->state_vector_l[0] + balance_infantry->count_theta_l * 2 * PI,balance_infantry->target_vector[0])
//					+ PID_calc(&balance_infantry->phi_L_speed_pid,balance_infantry->state_vector_l[1],balance_infantry->target_vector[1]);

//							balance_infantry->virtual_torque_r[1] = 0.00000030545f * fabs(PID_calc(&balance_infantry->phi_R_pid, 0.00000030545f * 
//					normalize_angle(balance_infantry->state_vector_r[0]),balance_infantry->target_vector[0]))0.00000030545f * 
//					+ fabs(PID_calc(&balance_infantry->phi_R_speed_pid,balance_infantry->state_vector_r[1],balance_infantry->target_vector[1]));
//	//				-PID_calc(&balance_infantry->phi_syn_pid,
//	//						normalize_angle(balance_infantry->state_vector_l[0] - balance_infantry->state_vector_r[0]),0.0);
//					
//					
//							balance_infantry->virtual_torque_l[1] = 0.00000030545f * fabs(PID_calc(&balance_infantry->phi_L_pid,
//					normalize_angle(balance_infantry->state_vector_l[0]),balance_infantry->target_vector[0])) 
//					+ fabs(PID_calc(&balance_infantry->phi_L_speed_pid,balance_infantry->state_vector_l[1],balance_infantry->target_vector[1]));
	//				+PID_calc(&balance_infantry->phi_syn_pid,
	//						normalize_angle(balance_infantry->state_vector_l[0] - balance_infantry->state_vector_r[0]),0.0);
				
				balance_infantry->virtual_torque_l[1] = 0.0f;
				balance_infantry->virtual_torque_r[1] = 0.0f;
				balance_infantry->target_L_left =0.15f;
				balance_infantry->target_L_right =0.15f;
				balance_infantry->stair_count ++;
				balance_infantry->x_filter = 0;
				balance_infantry->target_vector[2] = 0;
	//			balance_infantry->stair_time++;
			if( (fabs(balance_infantry->target_L_left - balance_infantry->L_left[0])<0.2f && fabs(balance_infantry->target_L_right - balance_infantry->L_right[0])<0.2f) || balance_infantry->stair_count > 600)
			{
				balance_infantry->stair_status++;
				balance_infantry->stair_count = 0;
//				balance_infantry->target_vector[2] = -1.2;
//				balance_infantry->qiankui = 60;
			}
			break;
			
		case 5:
				balance_infantry->stair_count++;
				if(balance_infantry->stair_count > 30){
						balance_infantry->stair_count = 0;
						balance_infantry->stair_time = 0;
						balance_infantry->stair_status++;
						balance_infantry->target_vector[0] = 0.0f;
						balance_infantry->target_vector[4] = 0.0f; 
						balance_infantry->x_filter = 0.0f;
						balance_infantry->target_vector[2] = -0.0f;
						balance_infantry->stair_flag = 0; /* 完成 */
//						balance_infantry->qiankui = G_Frontback;
						balance_infantry->stair_status = 0;
				}
				break;
				
		default:
			//balance_infantry->target_vector[0] = 0.0f;
			balance_infantry->target_vector[4] = 0.0f;
			balance_infantry->stair_status = 0;
			balance_infantry->stair_flag = 0;
			balance_infantry->stair_count = 0;
			break;
		}
	}
}

/**
 * @brief        任意姿态站立
 * @param          BalanceInfantry结构体
 * @retval       none
 */
void StandRobot(BalanceInfantry *balance_infantry)
{
//		balance_infantry->virtual_torque_l[1] = PID_calc(&balance_infantry->phi_L_speed_pid,balance_infantry->state_vector_l[1],balance_infantry->target_vector[1]);
//		balance_infantry->virtual_torque_r[1] = PID_calc(&balance_infantry->phi_R_speed_pid,balance_infantry->state_vector_r[1],balance_infantry->target_vector[1]);
	

    float err_left = balance_infantry->phi_left[0] - PI / 2.0f;
    float err_right = balance_infantry->phi_right[0] - PI / 2.0f;
//    const float THRESH = 0.26f; /* 原阈值保持不变 */
    static int count_state1,count_state0,count_state2 = 0;
    //判断是否需要自救
			//balance_infantry->stand_or_not = 1;//避免执行起立
    if(balance_infantry->stand_done_flag != 0 || balance_infantry->stand_or_not == 0)// && (fabsf(err_left) < THRESH || fabsf(err_right) < THRESH)) (balance_infantry->step_shirink_flag) || 
        return;
    balance_infantry->stand_done_flag = 0;
    switch(balance_infantry->stand_state){
        case 0://左右腿不在一边的情况下
					
            if(fabs(err_left - err_right) < 0.3f){
                balance_infantry->stand_state++;
                count_state1 = 0;
            }
						else
						{
								balance_infantry->target_L_left = 0.35f;
							  balance_infantry->target_L_right = 0.35f;
						}
						if(fabs(1.5f * PI - balance_infantry->state_vector_l[0]) < 0.3f)
						{
							balance_infantry->target_vector[0] = 1.8f * PI;
						}else
						{
							balance_infantry->target_vector[0] = 1.5f * PI;
						}
						balance_infantry->target_vector[1] = 0.0f;
						//更新pid参数
						balance_infantry->phi_L_pid.Kp = 4;
						balance_infantry->phi_L_pid.Kd = 120;
						balance_infantry->phi_R_pid.Kp = 4;
						balance_infantry->phi_R_pid.Kd = 120;
						balance_infantry->phi_L_speed_pid.Kp = 4;
						balance_infantry->phi_R_speed_pid.Kp = 4;
						balance_infantry->phi_L_pid.max_out = 8;
						balance_infantry->phi_R_pid.max_out = 8;
						balance_infantry->phi_L_speed_pid.max_out = 2;
						balance_infantry->phi_R_speed_pid.max_out = 2;

            if(err_left * err_right < 0){//不同号
                if(err_left > 0){//左边腿在下面，转动右边腿
									//normal一下误差就有多圈的效果了
								balance_infantry->virtual_torque_r[1] = PID_calc(&balance_infantry->phi_R_pid,
								normalize_angle(balance_infantry->state_vector_r[0] - balance_infantry->target_vector[0]),0.0)
								+ PID_calc(&balance_infantry->phi_R_speed_pid,balance_infantry->state_vector_r[1],balance_infantry->target_vector[1]);
								
								balance_infantry->virtual_torque_l[1] = 0.0f;
                }
                else{//右边腿在下面，转动左边腿
								balance_infantry->virtual_torque_l[1] = PID_calc(&balance_infantry->phi_L_pid,
								normalize_angle(balance_infantry->state_vector_l[0] - balance_infantry->target_vector[0]),0.0) 
								+ PID_calc(&balance_infantry->phi_L_speed_pid,balance_infantry->state_vector_l[1],balance_infantry->target_vector[1]);
								
								balance_infantry->virtual_torque_r[1] = 0.0f;
                }
            }
                
            break;
        case 1://
					if((balance_infantry->phi_left[0]) > 0 && (balance_infantry->phi_left[0]) < PI)
            balance_infantry->target_L_left = 0.14f;
					else
						balance_infantry->target_L_left = 0.35f;
					
					if(balance_infantry->phi_right[0] < PI && balance_infantry->phi_right[0] > 0)
            balance_infantry->target_L_right = 0.14f;
					else
						balance_infantry->target_L_right = 0.35f;
					
						balance_infantry->virtual_torque_l[1] = 0.0f;
						balance_infantry->virtual_torque_r[1] = 0.0f;
						balance_infantry->qiankui = 0.0f;
					
            if(fabs(balance_infantry->target_L_left - balance_infantry->L_left[0]) < 0.05 && fabs(balance_infantry->target_L_right - balance_infantry->L_right[0]) < 0.05 )
						{
							count_state0++;
							if(count_state0 > 550){
								if(balance_infantry->target_L_left == 0.14f && balance_infantry->target_L_right == 0.14f){
									balance_infantry->stand_state=3;
								}else{
									balance_infantry->stand_state = 2;
								}
								
								count_state0 = 0;
							}
						}
                 
            else{
							balance_infantry->stand_state = 1;
							count_state0 = 0;
						}
            break;
        case 2:
						balance_infantry->target_vector[0] = 1.5*PI;
						balance_infantry->target_vector[1] = 0.0f;
						balance_infantry->virtual_torque_r[1] = PID_calc(&balance_infantry->phi_R_pid,
						balance_infantry->state_vector_r[0] + balance_infantry->count_theta_r * 2 * PI,balance_infantry->target_vector[0])
				-PID_calc(&balance_infantry->phi_syn_pid,normalize_angle(balance_infantry->state_vector_l[0] - balance_infantry->state_vector_r[0]),0.0);
			
            balance_infantry->virtual_torque_l[1] = PID_calc(&balance_infantry->phi_L_pid,
						balance_infantry->state_vector_l[0] + balance_infantry->count_theta_l * 2 * PI,balance_infantry->target_vector[0])
				+PID_calc(&balance_infantry->phi_syn_pid,normalize_angle(balance_infantry->state_vector_l[0] - balance_infantry->state_vector_r[0]),0.0);

//							balance_infantry->target_vector[0] = 1.8f * PI;
//						balance_infantry->target_vector[1] = 0.0f;
//				
//						//normal一下误差就有多圈的效果了
//						balance_infantry->virtual_torque_r[1] = fabs(PID_calc(&balance_infantry->phi_R_pid,
//				normalize_angle(balance_infantry->state_vector_r[0] - balance_infantry->target_vector[0]),0.0))
//				+ (PID_calc(&balance_infantry->phi_R_speed_pid,balance_infantry->state_vector_r[1],balance_infantry->target_vector[1]));//正值往后抡
//				//-PID_calc(&balance_infantry->phi_syn_pid,
//					//	normalize_angle(balance_infantry->state_vector_l[0] - balance_infantry->state_vector_r[0]),0.0);
//				
//				
//            balance_infantry->virtual_torque_l[1] = fabs(PID_calc(&balance_infantry->phi_L_pid,
//				normalize_angle(balance_infantry->state_vector_l[0] - balance_infantry->target_vector[0]),0.0)) 
//				+ (PID_calc(&balance_infantry->phi_L_speed_pid,balance_infantry->state_vector_l[1],balance_infantry->target_vector[1]));
				//+PID_calc(&balance_infantry->phi_syn_pid,
						//normalize_angle(balance_infantry->state_vector_l[0] - balance_infantry->state_vector_r[0]),0.0);
//            balance_infantry->virtual_torque_r[1] += PID_calc(&balance_infantry->phi_R_speed_pid,balance_infantry->state_vector_r[1],balance_infantry->target_vector[1]);
//            balance_infantry->virtual_torque_l[1] += PID_calc(&balance_infantry->phi_L_speed_pid,balance_infantry->state_vector_l[1],balance_infantry->target_vector[1]);
				
            if((balance_infantry->phi_left[0]) > 0 && (balance_infantry->phi_left[0]) < PI && balance_infantry->phi_right[0] < PI && balance_infantry->phi_right[0] > 0){
//							if(fabs(balance_infantry->state_vector_l[0] - 1.5*PI) < 0.2f && fabs(balance_infantry->state_vector_r[0] - 1.5*PI) < 0.2f){
								balance_infantry->count_theta_r = 0;
								balance_infantry->count_theta_l = 0;
                count_state1++;
                if(count_state1 > 10){
										
                    balance_infantry->stand_state = 3;
										count_state1 = 0;
										balance_infantry->target_vector[0] = 0;
										balance_infantry->phi_L_pid.Kp = 100;
										balance_infantry->phi_R_pid.Kp = 100;
										balance_infantry->phi_L_speed_pid.Kp = 7.5;
										balance_infantry->phi_R_speed_pid.Kp = 7.5;
										balance_infantry->leg_control_pid_l.Kp = 720;
										balance_infantry->leg_control_pid_r.Kp = 720;
										balance_infantry->leg_speed_pid_l.Kp = 50;
										balance_infantry->leg_speed_pid_r.Kp = 50;
//										balance_infantry->qiankui = 8;
                }
            }
            else{
                count_state1 = 0;
                balance_infantry->stand_state = 2;
            }
            break;
        case 3:
            //收腿
						balance_infantry->target_vector[1] = 0.0f;
            balance_infantry->target_L_left = 0.14f;
            balance_infantry->target_L_right = 0.14f;
						if(fabs(balance_infantry->target_L_left - balance_infantry->L_left[0]) < 0.1f && fabs(balance_infantry->target_L_right - balance_infantry->L_right[0]) < 0.1f)
						{
								count++;
                if(count > 50){//250 6
										count = 0;
                    balance_infantry->stand_done_flag = 1;
                    balance_infantry->stand_state = 4;
										balance_infantry->phi_L_pid.Kp = 70;
										balance_infantry->phi_R_pid.Kp = 70;
										balance_infantry->phi_L_speed_pid.Kp = 5;
										balance_infantry->phi_R_speed_pid.Kp = 5;
										balance_infantry->leg_control_pid_l.Kp = 500;
										balance_infantry->leg_control_pid_r.Kp = 500;
										balance_infantry->leg_speed_pid_l.Kp = 38;
										balance_infantry->leg_speed_pid_r.Kp = 38;
										balance_infantry->target_vector[0] = 0.07;
										balance_infantry->target_vector[2] = -0.6;//-0.65
                }
            }
            else{
                count = 0;
                balance_infantry->stand_state = 3;
            }
//						balance_infantry->virtual_torque_l[1] = 0;
//						balance_infantry->virtual_torque_r[1] = 0; 
//						balance_infantry->virtual_torque_r[1] = PID_calc(&balance_infantry->phi_R_pid,
//				normalize_angle(balance_infantry->state_vector_r[0]),-0.1)
//				+ PID_calc(&balance_infantry->phi_R_speed_pid,balance_infantry->state_vector_r[1],balance_infantry->target_vector[1])
//				-PID_calc(&balance_infantry->phi_syn_pid,
//						normalize_angle(balance_infantry->state_vector_l[0] - balance_infantry->state_vector_r[0]),0.0);
//				
//				
//            balance_infantry->virtual_torque_l[1] = PID_calc(&balance_infantry->phi_L_pid,
//				normalize_angle(balance_infantry->state_vector_l[0]),-0.1) 
//				+ PID_calc(&balance_infantry->phi_L_speed_pid,balance_infantry->state_vector_l[1],balance_infantry->target_vector[1])
//				+PID_calc(&balance_infantry->phi_syn_pid,
//						normalize_angle(balance_infantry->state_vector_l[0] - balance_infantry->state_vector_r[0]),0.0);
//            if(fabs(balance_infantry->state_vector_l[0] + 0.26f) < 0.58f && fabs(balance_infantry->state_vector_r[0] + 0.26f) < 0.58f){
//							count++;
//                if(count > 3){//250 6
//										count = 0;
//                    balance_infantry->stand_done_flag = 1;
//                    balance_infantry->stand_state = 4;
//										balance_infantry->phi_L_pid.Kp = 60;
//										balance_infantry->phi_R_pid.Kp = 60;
//										balance_infantry->phi_L_speed_pid.Kp = 5;
//										balance_infantry->phi_R_speed_pid.Kp = 5;
//										balance_infantry->leg_control_pid_l.Kp = 500;
//										balance_infantry->leg_control_pid_r.Kp = 500;
//										balance_infantry->leg_speed_pid_l.Kp = 38;
//										balance_infantry->leg_speed_pid_r.Kp = 38;
//										balance_infantry->target_vector[0] = 0.07;
//										balance_infantry->target_vector[2] = -0.65;
//                }
//            }
//            else{
//                count = 0;
//                balance_infantry->stand_state = 3;
//            }
            break;
        default:
						
            break;

    }
}

/**
 * @brief        自救
 * @param          BalanceInfantry结构体
 * @retval       none
 */
void SelfResuce(BalanceInfantry *balance_infantry)
{
    //判断是否需要自救
    if(fabs(balance_infantry->state_vector_l[4]) < PI/2 && !balance_infantry->Self_resuce_flag_X ){
        balance_infantry->Self_resuce_flag_X = 0;
				balance_infantry->stand_or_not = 1;
        return;
    }
		static int rescue_count;
    //将起立标志位置0
		balance_infantry->stand_or_not = 0;//避免执行起立
    balance_infantry->stand_done_flag = 0;
//		balance_infantry->self_resuce_state = 0;
    balance_infantry->target_L_left = 0.35;
    balance_infantry->target_L_right = 0.35;
    balance_infantry->phi_L_pid.Kp = 4;
    balance_infantry->phi_L_pid.Kd = 100;
    balance_infantry->phi_R_pid.Kp = 4;
    balance_infantry->phi_R_pid.Kd = 100;
    balance_infantry->phi_L_speed_pid.Kp = 3;
    balance_infantry->phi_R_speed_pid.Kp = 3;
    balance_infantry->phi_L_pid.max_out = 5;
    balance_infantry->phi_R_pid.max_out = 5;
    balance_infantry->phi_L_speed_pid.max_out = 10;
    balance_infantry->phi_R_speed_pid.max_out = 10;
		balance_infantry->target_vector[1] = 0.0f;
    switch(balance_infantry->self_resuce_state){
//				case 0:
//					if(balance_infantry->state_vector_l[0] < PI && balance_infantry->state_vector_r[0] > PI)//(左)前(右)后 
//					{
//						    float target_speed_L = PID_calc(&balance_infantry->phi_L_pid,
//                                               balance_infantry->abs_phi_l,
//                                               -6.0f - balance_infantry->state_vector_l[4]/2);  // 角度外环：目标角度-实际角度
//                balance_infantry->virtual_torque_l[1] = PID_calc(&balance_infantry->phi_L_speed_pid,
//                                                                balance_infantry->state_vector_l[1],
//                                                                target_speed_L);  // 速度内环：目标速度-实际速度
//					}
//					else if(balance_infantry->state_vector_l[0] > PI && balance_infantry->state_vector_r[0] < PI)//(右)前(左)后
//					{
//						    float target_speed_R = PID_calc(&balance_infantry->phi_R_pid,
//                                               balance_infantry->abs_phi_r,
//                                               -6.0f - balance_infantry->state_vector_r[4]/2);  // 角度外环：目标角度-实际角度
//                balance_infantry->virtual_torque_r[1] = PID_calc(&balance_infantry->phi_R_speed_pid,
//                                                                balance_infantry->state_vector_r[1],
//                                                                target_speed_R);  // 速度内环：目标速度-实际速度
//					}
//					else 
//					if(fabsf(balance_infantry->state_vector_l[0] - balance_infantry->state_vector_l[0]) <0.2)
//					{
//								rescue_count++;
//								if(rescue_count >500)
//								{
//									balance_infantry->self_resuce_state++;
//									rescue_count = 0;
//								}					
//					} 
        case 0:
					if(balance_infantry->ChassisMode != CHASSIS_ZERO_FORCE && balance_infantry->Self_resuce_flag_X)
					{
					  if(balance_infantry->state_vector_l[4] > PI/3)//根据底盘的不同方向		两条腿在一边
            {
                //往一个方向转
                    balance_infantry->qiankui = 0.0f;
//                    balance_infantry->target_vector[0] = -PI/2;
                    balance_infantry->target_vector[1] = 0;

                                // 左电机串级PID计算：角度外环输出作为速度内环目标
                float target_speed_L = PID_calc(&balance_infantry->phi_L_pid,
                                               balance_infantry->abs_phi_l,
                                               -6.0f - balance_infantry->state_vector_l[4]/2);  // 角度外环：目标角度-实际角度
                balance_infantry->virtual_torque_l[1] = PID_calc(&balance_infantry->phi_L_speed_pid,
                                                                balance_infantry->state_vector_l[1],
                                                                target_speed_L)
																											+ PID_calc(&balance_infantry->phi_L_speed_pid,balance_infantry->state_vector_l[1],balance_infantry->target_vector[1]);  // 速度内环：目标速度-实际速度
                
                // 右电机串级PID计算（同左电机逻辑）
                float target_speed_R = PID_calc(&balance_infantry->phi_R_pid,
                                               balance_infantry->abs_phi_r,
                                               -6.0f - balance_infantry->state_vector_r[4]/2);
                balance_infantry->virtual_torque_r[1] = PID_calc(&balance_infantry->phi_R_speed_pid,
                                                                balance_infantry->state_vector_r[1],
                                                                target_speed_R) 						
																											+ PID_calc(&balance_infantry->phi_R_speed_pid,balance_infantry->state_vector_r[1],balance_infantry->target_vector[1]);
								if(balance_infantry->phi_left[0] + balance_infantry->state_vector_l[4] < 0.7 && balance_infantry->phi_right[0] + balance_infantry->state_vector_r[4] < 0.7)
								{
								  rescue_count++;
								  if(rescue_count >500)
									{
										balance_infantry->self_resuce_state++;
										rescue_count = 0;
									}
								}
								else 
							  {
									rescue_count = 0;
								}

            }
						else 
						{
                //往一个方向转
                balance_infantry->qiankui = 0.0f;
//                    balance_infantry->target_vector[0] = -PI/2;
                balance_infantry->target_vector[1] = 0;

                                // 左电机串级PID计算：角度外环输出作为速度内环目标
                float target_speed_L = PID_calc(&balance_infantry->phi_L_pid,
                                               balance_infantry->abs_phi_l,
                                               6.0f + balance_infantry->state_vector_l[4]/2);  // 角度外环：目标角度-实际角度
                balance_infantry->virtual_torque_l[1] = PID_calc(&balance_infantry->phi_L_speed_pid,
                                                                balance_infantry->state_vector_l[1],
                                                                target_speed_L);  // 速度内环：目标速度-实际速度
                
                // 右电机串级PID计算（同左电机逻辑）
                float target_speed_R = PID_calc(&balance_infantry->phi_R_pid,
                                               balance_infantry->abs_phi_r,
                                               6.0f + balance_infantry->state_vector_r[4]/2);
                balance_infantry->virtual_torque_r[1] = PID_calc(&balance_infantry->phi_R_speed_pid,
                                                                balance_infantry->state_vector_r[1],
                                                                target_speed_R);
								if((fabsf(balance_infantry->phi_left[0]) - fabsf(balance_infantry->state_vector_l[4])) < 0.5 && (fabsf(balance_infantry->phi_right[0]) - fabsf(balance_infantry->state_vector_r[4])) < 0.5)
								{
								  rescue_count++;
								  if(rescue_count >500)
									{
										balance_infantry->self_resuce_state++;
										rescue_count = 0;
									}
								}
								else 
							  {
									rescue_count = 0;
								}
            }
					}
					if(!balance_infantry->Self_resuce_flag_X)//再按一次按键退出
						return;
					
        break;
				case 1:
					  if(balance_infantry->state_vector_l[4] > PI/3)//根据底盘的不同方向
            {
								balance_infantry->phi_L_pid.Kp = 2;
								balance_infantry->phi_L_pid.Kd = 80;
								balance_infantry->phi_R_pid.Kp = 2;
								balance_infantry->phi_R_pid.Kd = 80;
								balance_infantry->phi_L_speed_pid.Kp = 2;
								balance_infantry->phi_R_speed_pid.Kp = 2;
								balance_infantry->phi_L_pid.max_out = 10;
								balance_infantry->phi_R_pid.max_out = 10;
								balance_infantry->phi_L_speed_pid.max_out = 10;
								balance_infantry->phi_R_speed_pid.max_out = 20;
				
                float target_speed_L = PID_calc(&balance_infantry->phi_L_pid,
                                               balance_infantry->abs_phi_l,
                                               -6.0f - balance_infantry->state_vector_l[4]/2);  // 角度外环：目标角度-实际角度
                balance_infantry->virtual_torque_l[1] = PID_calc(&balance_infantry->phi_L_speed_pid,
                                                                balance_infantry->state_vector_l[1],
                                                                target_speed_L)	+ PID_calc(&balance_infantry->phi_L_speed_pid,balance_infantry->state_vector_l[1],balance_infantry->target_vector[1]);  // 速度内环：目标速度-实际速度
  // 速度内环：目标速度-实际速度
                
                // 右电机串级PID计算（同左电机逻辑）
                float target_speed_R = PID_calc(&balance_infantry->phi_R_pid,
                                               balance_infantry->abs_phi_r,
                                               -6.0f - balance_infantry->state_vector_r[4]/2);
                balance_infantry->virtual_torque_r[1] = PID_calc(&balance_infantry->phi_R_speed_pid,
                                                                balance_infantry->state_vector_r[1],
                                                                target_speed_R)	+ PID_calc(&balance_infantry->phi_R_speed_pid,balance_infantry->state_vector_r[1],balance_infantry->target_vector[1]);

							}
						else
						{//轮子顶地的情况
								balance_infantry->phi_L_pid.Kp = 2;
								balance_infantry->phi_L_pid.Kd = 80;
								balance_infantry->phi_R_pid.Kp = 2;
								balance_infantry->phi_R_pid.Kd = 80;
								balance_infantry->phi_L_speed_pid.Kp = 2;
								balance_infantry->phi_R_speed_pid.Kp = 2;
								balance_infantry->phi_L_pid.max_out = 10;
								balance_infantry->phi_R_pid.max_out = 10;
								balance_infantry->phi_L_speed_pid.max_out = 10;
								balance_infantry->phi_R_speed_pid.max_out = 20;
				
                float target_speed_L = PID_calc(&balance_infantry->phi_L_pid,
                                               balance_infantry->abs_phi_l,
                                               6.0f + balance_infantry->state_vector_l[4]/2);  // 角度外环：目标角度-实际角度
                balance_infantry->virtual_torque_l[1] = PID_calc(&balance_infantry->phi_L_speed_pid,
                                                                balance_infantry->state_vector_l[1],
                                                                target_speed_L) + PID_calc(&balance_infantry->phi_L_speed_pid,balance_infantry->state_vector_l[1],balance_infantry->target_vector[1]);;  // 速度内环：目标速度-实际速度
                
                // 右电机串级PID计算（同左电机逻辑）
                float target_speed_R = PID_calc(&balance_infantry->phi_R_pid,
                                               balance_infantry->abs_phi_r,
                                               6.0f + balance_infantry->state_vector_r[4]/2);
                balance_infantry->virtual_torque_r[1] = PID_calc(&balance_infantry->phi_R_speed_pid,
                                                                balance_infantry->state_vector_r[1],
                                                                target_speed_R) + PID_calc(&balance_infantry->phi_R_speed_pid,balance_infantry->state_vector_r[1],balance_infantry->target_vector[1]);;
						}
						if(fabs(balance_infantry->state_vector_l[4])<0.8 && fabs(balance_infantry->state_vector_r[4])<0.8)
						{
								balance_infantry->self_resuce_state++;
						}
						
						if(!balance_infantry->Self_resuce_flag_X)//再按一次按键退出
							return;
						
					break;
        default:
					  balance_infantry->phi_L_pid.Kp = 2.4;//15;
						balance_infantry->phi_L_pid.Kd = 100;//500;
						balance_infantry->phi_R_pid.Kp = 2.4;//15;
						balance_infantry->phi_R_pid.Kd = 100;//500;
						balance_infantry->phi_L_speed_pid.Kp = 1.5;//15;
						balance_infantry->phi_R_speed_pid.Kp = 1.5;//15;
						balance_infantry->phi_L_pid.max_out = 5;//13;
						balance_infantry->phi_R_pid.max_out = 5;//13;
						balance_infantry->phi_L_speed_pid.max_out = 10;//8;
						balance_infantry->phi_R_speed_pid.max_out = 10;//8;
            balance_infantry->self_resuce_state = 0;
				    balance_infantry->Self_resuce_flag_X = 0;
						balance_infantry->stand_or_not = 0;
            break;
                
    }
}
static float gas_spring_compen(float L0)
{
    float l1 = 0.215,l2 = 0.254,s2 = 0.059,s3 = 0.185,alpha = 0.266;//单位弧度与米
    float Fs = 150,Fv = 0.0;
    float cos_theta3 = 0 ,theta3 = 0,ls = 0.0;
    cos_theta3 = (l1*l1 + l2*l2 - L0*L0)/(2 * l1 * l2);
    theta3 = acosf(cos_theta3);
    ls = sqrt(s3*s3 + s2*s2 - 2*s2*s3*arm_cos_f32(theta3 - alpha));
    Fv = Fs*(s2*s3*arm_sin_f32(theta3 - alpha)*L0)/(l1*l2*arm_sin_f32(theta3)*ls);
    if(balance_infantry.ChassisMode != CHASSIS_ZERO_FORCE)
        return Fv;
    else
        return 0;
}

/**
 * @brief        ???????????С??????н???????
 * @param          BalanceInfantry????
 * @retval       none
 */
fp32 BCD;
fp32 AC;
fp32 ACB;
fp32 AD;
static fp32 get_sin_theta(BalanceInfantry *balance_infantry, uint8_t leg)
{
    // 1. ?????????У??
    if (balance_infantry == NULL || (leg != LEG_LEFT && leg != LEG_RIGHT)) {
        return 0.0f;
    }

    // 2. ?????????????????????
    fp32 *phi_array = (leg == LEG_LEFT) ? balance_infantry->phi_left : balance_infantry->phi_right;

    // 3. ???????????
    const fp32 LEN_1 = 59.0f;                // ??????1
    const fp32 LEN_2 = 215.0f;               // ??????2
    const fp32 ANGLE_59_DEG = 59.0f * RAD_PER_DEG;  // 59???????
    const fp32 LEN_1_SQ = LEN_1 * LEN_1;     // ???????1?????
    const fp32 LEN_2_SQ = LEN_2 * LEN_2;     // ???????2?????
    const fp32 TWO_LEN1_LEN2 = 2.0f * LEN_1 * LEN_2;  // ?????2*LEN1*LEN2
    const fp32 TWO_LEN1_AC = 2.0f * LEN_1;   // ?????2*LEN1

    // 4. ????????????????????????????????Ч???
    // ????4.1??????BCD??????1?μ????????????
    if(leg == LEG_LEFT){
        phi_array[1] = balance_infantry->phi_left[1];
        phi_array[2] = balance_infantry->phi_left[2];
    }
    else{
        phi_array[1] = balance_infantry->phi_right[1];
        phi_array[2] = balance_infantry->phi_right[2];
    }
		BCD = PI - phi_array[1] + phi_array[2];

    // ????4.2???????????????
    fp32 cos_59 = arm_cos_f32(ANGLE_59_DEG);  // ?????59????????
    fp32 sin_59 = arm_sin_f32(ANGLE_59_DEG);  // ?????59????????
    fp32 cos_BCD = arm_cos_f32(BCD);          // ?????BCD?????????

    // ????4.3??????AC
    AC = 180;
    // ?????????????AC?0?????????????????/????????
    if (AC <= 1e-6f) {
        return 0.0f;
    }

    // ????4.4??????ACB
    ACB = (LEN_1 * sin_59) / AC;

    // ????4.5??????AD???????????
    AD = (AC * AC) + LEN_1_SQ - (TWO_LEN1_AC * AC * arm_cos_f32(BCD-ACB));
    // ?????????????AD?0????????????????????
    if (AD <= 1e-6f) {
        return 0.0f;
    }

    // ????4.6????????????
    fp32 sin_BCD_ACB = arm_sin_f32(BCD - ACB);
    fp32 result = (AC * sin_BCD_ACB) / sqrt(AD);
				
    // 5. ??????
    return (result > 1.0f) ? 1.0f : (result < -1.0f) ? -1.0f : result;
}

/**
 * @brief        整体控制
 * @param    	  BalanceInfantry结构体
 * @retval 	  none
 */
//float num[1] = LeftMotor->measure.total_angle;
void main_control(BalanceInfantry *balance_infantry)
{
	get_sensor_message(balance_infantry);

	set_target_vector(balance_infantry);
	MotionSolve(balance_infantry);
	getTheta(balance_infantry);

	/* 力矩计算 */
	updataK(balance_infantry);
	LQR_Solve(balance_infantry);

	leg_control(balance_infantry);

	VMC_Solve(balance_infantry);

	off_ground_detect(balance_infantry);

	/* 输出整理 */
	motor_output(balance_infantry);

	/* 限幅 */
}

/**
  * @brief          计算ecd与offset_ecd之间的相对角度
  * @param[in]      ecd: 电机当前编码
  * @param[in]      offset_ecd: 电机中值编码
  * @retval         相对角度，单位rad
  */
static fp32 motor_ecd_to_angle_change(uint16_t ecd, uint16_t offset_ecd)
{
    int32_t relative_ecd = ecd - offset_ecd;
    if (relative_ecd > HALF_ECD_RANGE)
    {
        relative_ecd -= ECD_RANGE;
    }
    else if (relative_ecd < -HALF_ECD_RANGE)
    {
        relative_ecd += ECD_RANGE;
    }

    return relative_ecd * MOTOR_ECD_TO_RAD;
}
// 角度标准化函数：将任意角度映射到 [-π, π] 范围内
static float normalize_angle(float angle_rad) {
    // 若结果大于 π，则减去 2π；若小于 -π，则加上 2π，最终落在 [-π, π]
    if (angle_rad > PI) {
        angle_rad -= 2 * PI;
    } else if (angle_rad < -PI) {
        angle_rad += 2 * PI;
    }
    return angle_rad;
}

static inline float limit_float(float value, float min_val, float max_val) {
    float real_min = (min_val < max_val) ? min_val : max_val;
    float real_max = (min_val < max_val) ? max_val : min_val;
    
    if (value <= real_min) return real_min;
    if (value >= real_max) return real_max;
    return value;
}


static void ESO_Clac(ESO_T * eso , float feedback , float u)
{
	float beta1 = 3*eso->w0;
	float beta2 = eso->w0 * eso->w0;
	float beta3 = eso->w0 * eso->w0 * eso->w0;
	float err = eso->z1 - feedback;
	eso->z1 += (eso->z2 - beta1 * err )* eso->dt;
	eso->z2 += (eso->b0 * u + eso->z3 - beta2* err)*eso->dt;
	eso->z3 += -beta3 * err * eso->dt;
	
	eso->z3 = eso->z3 > eso->max ? eso->max : eso->z3;
	eso->z3 = eso->z3 < -eso->max ? -eso->max : eso->z3;
}

float get_chassis_wz(void)
{
	return balance_infantry.GyroData[2];
}
void get_abs_angle(BalanceInfantry *balance_infantry)
{
			//计算单圈角度差值
			double raw_diff_l = balance_infantry->phi_left[0] - balance_infantry->last_phi_l;
			double raw_diff_r = balance_infantry->phi_right[0] - balance_infantry->last_phi_r;
	
			//归一化差值到[-PI, PI] → 得到“物理上真实旋转的小角度”
			double real_rot_diff_l = fmod(raw_diff_l, 2 * PI);
			if (real_rot_diff_l > PI) {
					real_rot_diff_l -= 2 * PI;
			} else if (real_rot_diff_l < -PI) {
					real_rot_diff_l += 2 * PI;
			}
			double real_rot_diff_r = fmod(raw_diff_r, 2 * PI);
			if (real_rot_diff_r > PI) {
					real_rot_diff_r -= 2 * PI;
			} else if (real_rot_diff_r < -PI) {
					real_rot_diff_r += 2 * PI;
			}
			
			//更新连续绝对角度
			balance_infantry->abs_phi_l += real_rot_diff_l;
			balance_infantry->abs_phi_r += real_rot_diff_r;
}

float ramp_function(float current, float target, float step, float tolerance) 
{
    if (fabsf(current - target) <= tolerance) 
		{
        return target;  // 已达到目标
    }
    
    if (current < target) 
		{
        current += step;
        return (current > target) ? target : current;
    } 
		else 
		{
        current -= step;
        return (current < target) ? target : current;
    }
}
