#include "AI_policy.h"
#include "app_x-cube-ai.h"
#include "network.h"
#include "norm_params.h"
#include "bsp_vofa.h"
#include "bsp_dwt.h"
#include "bsp_can.h"
#include <string.h>
#include "FreeRTOS.h"
#include "task.h"

extern RC_ctrl_t rc_ctrl;

/* ══════════════════════════════════════════════════════════════════════
 *  诊断开关: 冻结策略输出 (打开闭环)
 *
 *  用来区分 20~25 Hz 抖动到底是:
 *    (a) 策略闭环极限环   → 冻结后应该消失
 *    (b) 机械共振/PD 层   → 冻结后仍然存在 (与策略、CPU、延迟无关)
 *
 *  冻结后关节 PD (kp=40, kd=2) 仍然以 500 Hz 把 target_pos 拉向固定值,
 *  但策略不再参与。观测里 I39~I44 (filtered_action)、I45~I50 (target_pos)
 *  会变成固定值, 可以用这个确认开关生效。
 *
 *  ⚠ 轮子 target 也会被冻结, 机器人失去自平衡能力。
 *    请把机器人吊起来 / 架上支架做这个测试。
 *
 *  改完记得改回 0。
 * ══════════════════════════════════════════════════════════════════════ */
#define RL_DEBUG_FREEZE_ACTION   0
/*  0 = 正常
 *  1 = 策略输出强制为 0 (关节回标称姿态, 轮子只剩速度阻尼)
 *  2 = 冻结在当前帧 (约 1 s 后抓一帧, 尽量保持当前姿态, 减小瞬态)
 */

/* ══════════════════════════════════════════════════════════════════════
 *  VOFA 日志开关 (A/B 对照实验用)
 *
 *  【实测结果】
 *    VOFA_LOG_ENABLE = 1  -> 轻微抖动
 *    VOFA_LOG_ENABLE = 0  -> 直接乱动   (与"日志无关"的预期相反!)
 *
 *  这说明环路本身裕度极小: VOFA 只改变几十微秒的时序, 却能把行为从
 *  "微抖"翻成"乱动"。所以 VOFA 不是根因, 但可以当成"扰动敏感度探针"。
 *
 *  ── 开关 1: VOFA_LOG_ENABLE ─────────────────────────────────────────
 *    1 = 正常: 组 67 通道帧(272 字节)并发 UART DMA, ~220 Hz
 *    0 = 完全关闭: 不组帧、不发 DMA、不刷新 CPU 统计
 *
 *  ── 开关 2: VOFA_DMA_SEND (仅 LOG=1 时有意义) ───────────────────────
 *    1 = 正常发送
 *    0 = 仍然组帧、做全部 memcpy 和 DWT 读取, 但跳过
 *        vofa_justfloat_output() —— 不发 DMA, 不产生 TxCplt 中断
 *
 *  ── 三组对照 (只改这两个数字, 其他一律不动) ─────────────────────────
 *      LOG=1  SEND=1   ->  基准, 已知"微抖"
 *      LOG=1  SEND=0   ->  有组帧开销, 无 UART/DMA 中断   <== 下一步测这个
 *      LOG=0  SEND=-   ->  两者都没有, 实测"乱动"
 *
 *  判读:
 *      SEND=0 就乱动   -> 起镇定作用的是 UART DMA / 中断
 *      SEND=0 仍微抖   -> 起镇定作用的是组帧占用的那点 CPU 时间
 *
 *  注意: 日志块不修改任何控制状态, PD 力矩计算路径完全不变,
 *        所以三组之间唯一的差别就是时序和中断负载。
 *        关闭后 I51~I56(CPU 占比) 不再刷新, 属预期。
 * ══════════════════════════════════════════════════════════════════════ */
#define VOFA_LOG_ENABLE   1
#define VOFA_DMA_SEND     1

/* ══════════════════════════════════════════════════════════════════════
 *  站立高度档位 (由右手自定义按钮 / 键盘 R 切换)
 *
 *  信号链 (底盘拿不到"自定义按钮"和"键盘R"的区别, 两者完全等价):
 *    gimbal  remote_control.c:503    解析出 rc.Custom_button_left
 *    gimbal  communicate_task.c:129  FLag 的 bit6 = Custom_button_left || 键盘R
 *    gimbal  communicate_task.c:162  main[1] = FLag  →  CAN id rc1_id
 *    chassis bsp_can.c:161           balance_infantry.key = main[1]
 *    chassis ChassisControl.c:804    KEY_R(=0x40=bit6) 边沿 → leg_flag = ~leg_flag
 *
 *  为什么可以安全地拿 leg_flag 当高度开关:
 *    老腿长 PID 的输出 extra_leg_force_l/r, 在 application/ChassisControl.c
 *    里的引用全部被注释掉了(1208/1211/1447/1448 行), 也就是说
 *    target_L_left/right 现在不接任何输出 —— leg_flag 本来是个空转的标志位。
 *
 *  为什么这两档高度是安全的:
 *    训练配置 height_limit = [0.20, 0.38] (mujoco.yaml 第 96~98 行),
 *    高度指令本来就在随机化, 所以两档都落在训练分布内部。
 *
 *  更正: 这台机器到目前为止所有测试都是在 commands[2] = 0.28 下做的,
 *        0.37 从来没有实测过 —— 所以绝不能把 0.37 当成"已知更好"。
 *        按钮的价值就在这里: 不用重新烧录, 站着就能 A/B 两个站高。
 *
 *    leg_flag = 0 (上电默认, 还没按过)  ->  HEIGHT_CMD_DEFAULT = 0.28  (你一直在测的)
 *    leg_flag = 1 (按一次)              ->  HEIGHT_CMD_SWITCH  = 0.37  (未测过, 待验证)
 *
 *  预期差别 (蹲得低就得多要力矩):
 *    L0 = 0.37 -> 气弹簧推力 59.2 N ;  L0 = 0.28 -> 46.0 N (少 22%)
 *    rl->torques 在 0.28 下本来就长期贴着 ±40 钳位, 抬高站姿应该能减轻饱和。
 * ══════════════════════════════════════════════════════════════════════ */
#define HEIGHT_CMD_DEFAULT 0.28f
#define HEIGHT_CMD_SWITCH  0.37f
/* 高度指令斜坡速率 (m/s): 防止按按钮时直接丢给策略一个阶跃指令。
 * 0.10 m/s 下, 0.28 <-> 0.37 (9 cm) 约需 0.9 s。 */
#define HEIGHT_CMD_SLEW    0.10f

/* ══════════════════════════════════════════════════════════════════════
 *  ────────  测 试 区 : real2sim 轨迹辨识  ────────
 *
 *  目的: 不逐个测减速器刚度/齿隙/阻尼(单关节测不准), 而是给一条完全已知的
 *        激励, 在 MuJoCo 里跑同一条, 比 dof_pos 曲线, 反推 armature/friction
 *        和力矩响应滞后。这就是华南虎文章推荐的关节辨识流程。
 *
 *  做法: 开启后策略输出被完全旁路, target_pos 由测试目标生成:
 *        摆角(A): 绕起始姿态摆动 φ0, 腿长 L 不变
 *        腿长(B): 改 L, φ0 不变
 *        两者都是 (φ0, L) 目标, 经「精确逆解」算成关节目标角, 再走原有关节 PD。
 *
 *  为什么用精确逆解而不是 J_inv 线性化: 那份雅可比和正解 MotionSolve 对不上 ——
 *  在标称位形 (φ1=5π/6, φ4=π/6) 处数值差分得到 ∂L/∂φ1 ≈ -0.38、∂L/∂φ4 ≈ -0.10,
 *  而 J_l[0][1] / J_l[1][1] 都给出 ≈ -0.15, 符号和倍数都不一致。实测拿它做逆解
 *  会把腿挪到错误位形。这里改用 chassis_algorithm.c:11-36 InvMotionSolve 的代数闭式解
 *  (泛化成左右腿共用一个函数), 没有线性化误差, 也不依赖雅可比。
 *  ！关键: 仿真里必须用同一个映射, 否则两边指令不同, 比曲线没有意义。
 *
 *  指令定义(仿真照抄):
 *    p = (x_C, y_C)  轮心位置;   L = |p|,  φ0 = atan2(y_C, x_C)
 *    L  = TRAJ_BASE_HEIGHT + (腿长模式) A_L *sin(phase)     ← 直接给绝对腿长
 *    φ0 = PI/2 - pitch     + (摆角模式) A_th*sin(phase)     ← theta=0
 *    (φ1, φ4) = traj_ik_exact(L, φ0)          代数闭式, 不做线性化
 *    target_pos = normalize(φ_target - 关节零位偏移)   偏移是常数(PI_6 / PI_5_6)
 *
 *  关节通道对应(已核对 chassis_algorithm.c:296-297 / 327-328 与 AI_policy.c:863-866):
 *    左腿: φ1 = phi_left[1]  -> LR -> target_pos[2]
 *          φ4 = phi_left[4]  -> LF -> target_pos[0]
 *    右腿: φ1 = phi_right[1] -> RR -> target_pos[3]
 *          φ4 = phi_right[4] -> RF -> target_pos[1]
 *
 *  ！必须把机器人吊起来。轮子目标置 0(只剩阻尼)。
 *  ！现成的 VOFA 通道就够, 不用改:
 *     I45~I48 = target_pos(指令)   I19~I22 = dof_pos_err(实际)
 *     I23~I26 = dof_vel            I61~I64 = rl->torques
 *     I35~I38 = output_torque(含气弹簧项)
 * ══════════════════════════════════════════════════════════════════════ */
#define TRAJ_TEST_ENABLE   1        /* 1 = 开启轨迹辨识测试 */
#define TRAJ_MODE          1        /* 0 = 摆角  1 = 腿长  2 = 两者  3 = 轮速辨识 */
#define TRAJ_WAVE          0        /* 轮速模式用: 0 = 正弦(测带宽/阻尼)  1 = 斜坡(测静摩擦) */
#define TRAJ_SWEEP         0        /* 0 = 定频正弦  1 = 对数扫频 */
#define TRAJ_FREQ_HZ       0.05f    /* 准静态: 一个周期 20 s */
#define TRAJ_F0_HZ         1.0f     /* 扫频起始频率 */
#define TRAJ_F1_HZ         20.0f    /* 扫频终止频率 */
#define TRAJ_DURATION_S    40.0f    /* 两个完整周期, 顺便看迟滞 */
#define TRAJ_DELAY_S       1.0f     /* 退出无力后先在默认位形站 1 s, 再开始扫频。
                                     * 这 1 s 里 t 被钳在 0, phase = 0, s = 0,
                                     * 所以 target_pos 恰好就是
                                     * (TRAJ_BASE_HEIGHT, TRAJ_BASE_PHI0) 的默认位形 ——
                                     * 顺带是一次从自由姿态到默认位形的阶跃响应。
                                     * ！分析时 analyze_sweep.py 的 D 必须传同一个值。 */
/* ── 测试台直接给"腿长 + 摆角", 关节位置由精确逆解自动算 ──
 * TRAJ_BASE_HEIGHT 就是目标腿长 (m, 绝对值, 不是偏置), 改它就能换腿长。
 * TRAJ_BASE_PHI0 是摆角基准 (rad), 默认 PI/2 = 腿竖直。
 *
 * 标称站姿对应 L = 0.2804, phi0 = 1.5711 (即 dof_pos = 0):
 *   phi1 = 5π/6, phi4 = π/6  ->  x_C = -0.000082, y_C = 0.280355
 * 所以 BASE_HEIGHT = 0.28 时逆解结果 ≈ 当前位姿, dphi ≈ 0, 不会有阶跃。 */
#define TRAJ_BASE_PHI0     1.5707963f   /* 摆角基准 (rad) */
/* 前/后倾测试: 实际 φ0 = TRAJ_BASE_PHI0 + TRAJ_PHI0_TILT。
 * 坐标: +y 由髋指向轮(向下), x_C = L*cos(phi0)。
 *   TILT > 0 -> x_C < 0 (轮心往 -x)
 *   TILT < 0 -> x_C > 0 (轮心往 +x)
 * 0.15 rad ≈ 8.6°, L=0.20 时轮心水平位移约 30 mm。
 * 注意: φ0 变化对所有四个关节近似是共模旋转 —— 两个关节角一起加/减同一个量,
 * 所以 LF 和 LR 的 dof_pos 会一个变大一个变小, 不是对称的。 */
#define TRAJ_PHI0_TILT     0.0f
#define TRAJ_BASE_HEIGHT   0.28f        /* 目标腿长 (m, 绝对值) */
#define TRAJ_PITCH_COMP    0            /* 1 = φ0 里扣掉车身俯仰: phi0 = PI/2 - pitch,
                                         *     等价于在世界系保持 theta=0
                                         * 0 = 纯机体相对, phi0 恒为 PI/2 (推荐)
                                         *
                                         * 为什么默认 0: 扣 pitch 会把 IMU 拉进指令环,
                                         *   车身俯仰 -> 关节目标 -> 腿动 -> 反作用 -> 车身再俯仰
                                         * 也是一条闭环。腿部辨识期间没有姿态环去压它,
                                         * 实测会让整个机体缓慢来回摆。辨识只需要一个
                                         * 可重复的「机体相对」位形, 不需要世界系竖直。 */
#define TRAJ_AMP_ANGLE     0.0f     /* 本轮不用摆角, 纯腿长扫 */
#define TRAJ_AMP_LEN       0.08f    /* 0.08 叠加在 BASE_HEIGHT 0.28 上 -> 0.20 ~ 0.36。
                                     * 跨度 0.16 m, 补偿量从约 3.5 变到 9.5 N·m,
                                     * 斜率信噪比比 ±0.03 那版高一倍。
                                     * 上端不用 0.37: 接近伸直奇异, ∂L/∂φ 趋零,
                                     * 力矩信息少, 而且可能顶到机械限位把 dof_pos 削平。 */
/* 激励形状: 0 = 正弦, 1 = 三角波。
 * 为什么这次要用三角波: 正弦从 BASE_HEIGHT 出发往上走, 上行只覆盖 0.28~0.36,
 * 下行才覆盖 0.36~0.20 —— 下半段只有单方向的点, 迟滞配对不出来。
 * 实测上行平均偏短 6.5 mm、下行平均偏长 31 mm, 差 37 mm 腿长,
 * 折成关节死区 ±0.065 rad / 摩擦约 2.6 N·m, 比要测的补偿缺口还大。
 * 三角波上下行都匀速、都覆盖全量程, 同一位形能取上行/下行两点:
 *     中线 = (res_up + res_dn)/2  -> 真实补偿残余 (摩擦消掉)
 *     半差 = (res_dn - res_up)/2  -> 库仑摩擦
 * 只有中线能用来改 gas_spring_compen。 */
#define TRAJ_TRIANGLE      1
#define TRAJ_AMP_WHEEL     5.0f    /* 轮速辨识: 幅度(正弦)或终值(斜坡), rad/s */
#define TRAJ_MASK          0x0Fu    /* 驱动哪些腿: bit0=LF bit1=RF bit2=LR bit3=RR */

/* 基准位姿固定取策略 action=0 的站姿 (DEFAULT_DOF_POS = 0), 没有开关:
 * 那就是 (theta=0, L=0.2804) 的标称位姿, 也是 14 Hz 抖动被观察到时的姿态。
 * 切出无力后腿会从软塌姿态"站"到这个位姿, 本身是一个阶跃, 可以顺带看阶跃响应。 */

/* 辨识日志降频: 每 N 拍才发一帧 VOFA。
 * 任务本身约 500 Hz, 500 Hz × 8 通道会让 VOFA+ 的曲线控件很卡(它要重绘
 * 几万个点), 而静摩擦斜坡是秒级的慢过程, 100 Hz 都绰绰有余。
 *   1 = 500 Hz   2 = 250 Hz   4 = 125 Hz   10 = 50 Hz
 * 注意: 腿模式的扫频/正弦要测到 20 Hz, 那时必须设 1, 否则每周期点太少,
 * 幅值和相位都会算不准。 */
#define TRAJ_VOFA_DECIM    4        /* 本轮是 0.05 Hz 准静态, 125 Hz 绰绰有余。
                                     * (若是 20 Hz 扫频, 这里必须是 1 = 500 Hz) */

/* 退出无力后的力矩软启动时间 (ms)。
 * 无力模式下关节完全自由, 气弹簧会把腿顶到机械极限, dof_pos 离 target_pos
 * 可能有 0.3~0.5 rad。退出无力那一拍的 PD 力矩 = kp*误差 直接就是 12~20 N·m,
 * 气弹簧补偿再从 0 跳到约 6 N·m, 全部在一个 2 ms 周期内加上去 —— 机身就"蹬"
 * 起来了。这里把最终输出力矩从 0 线性拉到 1, 顺带也限制了腿走回标称位形的速度。
 * 0 = 关闭软启动 (退化成原来的行为)。 */
#define SOFT_START_MS      300.0f

/* 力矩整体缩放系数 (原来是散落在 5 处的魔数 0.6)。
 * 训练配置 torque_scale_range = [0.8, 1.05], 名义是 1.0 —— 0.6 低于下限 25%。
 * 它是用来压增益的临时手段, 不是模型的参数。
 * 测试模式必须取 1.0, 否则「真机 PD×0.6」对「仿真 PD×1.0」, 曲线必然对不上,
 * 而且对不上的原因和要找的 gap 无关。 */
#if TRAJ_TEST_ENABLE
#define TORQUE_SCALE       1.0f
#else
#define TORQUE_SCALE       0.6f
#endif

/* ══════════════════════════════════════════════════════════════════════
 *  CPU 占用率统计
 *
 *  FreeRTOS 的 configGENERATE_RUN_TIME_STATS 需要一对宏:
 *    portCONFIGURE_TIMER_FOR_RUN_TIME_STATS()  调度器启动前初始化计数器
 *    portGET_RUN_TIME_COUNTER_VALUE()          每次任务切换时读一次
 *  这两个宏在 FreeRTOSConfig.h 里指向下面这两个函数。
 *
 *  计时源用 DWT->CYCCNT (168 MHz, 32 位, 25.6 s 回绕一次)。
 *  每 500 ms 用 uxTaskGetSystemState() 采一次, 算出各任务占比,
 *  塞进 vofa 的 I51~I56 通道 (原来是恒为 0 的 vmc_torques)。
 * ══════════════════════════════════════════════════════════════════════ */

void DWT_CpuStatInit(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t DWT_GetCyccnt(void)
{
    return DWT->CYCCNT;
}

#define CPU_STAT_MAX_TASKS  16

float cpu_pct_imu     = 0.0f;   /* I51: imuTask     (INS_task, 优先级 6) */
float cpu_pct_chassis = 0.0f;   /* I52: ChassisTask (chassis_task, 优先级 4) */
float cpu_pct_rl      = 0.0f;   /* I53: RLTask      (RL_Task, 优先级 1) */
float cpu_pct_dm8009  = 0.0f;   /* I54: dm8009Task  (优先级 5) */
float cpu_pct_idle    = 0.0f;   /* I55: IDLE        (CPU 余量) */
float cpu_pct_other   = 0.0f;   /* I56: 其它任务 + 中断 (ISR 不记进任何任务) */

static TaskStatus_t cpu_st[CPU_STAT_MAX_TASKS];
static uint32_t     cpu_prev_tick[CPU_STAT_MAX_TASKS];
static char         cpu_prev_name[CPU_STAT_MAX_TASKS][configMAX_TASK_NAME_LEN];
static UBaseType_t  cpu_prev_n     = 0;
static uint32_t     cpu_prev_total = 0;
static uint8_t      cpu_first      = 1;

static void cpu_stats_update(void)
{
    uint32_t    total_now = 0;
    UBaseType_t n, i, j;
    uint32_t    total_delta;
    float       sum;

    /* uxTaskGetSystemState 只有在 configUSE_TRACE_FACILITY==1 时才会被编译进
     * tasks.c; 而 TaskStatus_t.ulRunTimeCounter 还要求 configGENERATE_RUN_TIME_STATS==1。
     * 本工程的 Inc/FreeRTOSConfig.h 两个都没开, 直接调用会在链接期报
     *   L6218E: Undefined symbol uxTaskGetSystemState
     * 所以这里加编译期保护: 未开启时 n=0, 下面的 if(n==0) return 直接退出,
     * cpu_pct_* 保持 0, 完全不影响控制逻辑。 */
#if (configUSE_TRACE_FACILITY == 1) && (configGENERATE_RUN_TIME_STATS == 1)
    n = uxTaskGetSystemState(cpu_st, CPU_STAT_MAX_TASKS, &total_now);
#else
    n = 0;
#endif
    if (n == 0) return;

    /* 第一次只记基准, 不算百分比 */
    if (cpu_first)
    {
        for (i = 0; i < n; i++)
        {
            strncpy(cpu_prev_name[i], cpu_st[i].pcTaskName, configMAX_TASK_NAME_LEN - 1);
            cpu_prev_name[i][configMAX_TASK_NAME_LEN - 1] = '\0';
            cpu_prev_tick[i] = cpu_st[i].ulRunTimeCounter;
        }
        cpu_prev_n     = n;
        cpu_prev_total = total_now;
        cpu_first      = 0;
        return;
    }

    /* 无符号减法, 自动处理 32 位回绕 */
    total_delta    = total_now - cpu_prev_total;
    cpu_prev_total = total_now;

    cpu_pct_imu = cpu_pct_chassis = cpu_pct_rl = 0.0f;
    cpu_pct_dm8009 = cpu_pct_idle = 0.0f;

    for (i = 0; i < n; i++)
    {
        uint32_t delta = 0;
        float    pct;

        for (j = 0; j < cpu_prev_n; j++)
        {
            if (strncmp(cpu_prev_name[j], cpu_st[i].pcTaskName, configMAX_TASK_NAME_LEN) == 0)
            {
                delta = cpu_st[i].ulRunTimeCounter - cpu_prev_tick[j];
                cpu_prev_tick[j] = cpu_st[i].ulRunTimeCounter;
                break;
            }
        }

        pct = (total_delta > 0) ? (100.0f * (float)delta / (float)total_delta) : 0.0f;

        if      (strncmp(cpu_st[i].pcTaskName, "imuTask",     configMAX_TASK_NAME_LEN) == 0) cpu_pct_imu     += pct;
        else if (strncmp(cpu_st[i].pcTaskName, "ChassisTask", configMAX_TASK_NAME_LEN) == 0) cpu_pct_chassis += pct;
        else if (strncmp(cpu_st[i].pcTaskName, "RLTask",      configMAX_TASK_NAME_LEN) == 0) cpu_pct_rl      += pct;
        else if (strncmp(cpu_st[i].pcTaskName, "dm8009Task",  configMAX_TASK_NAME_LEN) == 0) cpu_pct_dm8009  += pct;
        else if (strncmp(cpu_st[i].pcTaskName, "IDLE",        configMAX_TASK_NAME_LEN) == 0) cpu_pct_idle    += pct;
    }

    /* 剩下的 = 其它任务 + 所有中断 */
    sum = cpu_pct_imu + cpu_pct_chassis + cpu_pct_rl + cpu_pct_dm8009 + cpu_pct_idle;
    cpu_pct_other = (sum < 100.0f) ? (100.0f - sum) : 0.0f;
}

static float safe_asin(float x)
{
    if (x >  1.0f) x =  1.0f;
    if (x < -1.0f) x = -1.0f;
    return asinf(x);
}

static float normalize_angle(float angle)
{
    while (angle >=  M_PI) angle -= 2.0f * M_PI;
    while (angle < -M_PI) angle += 2.0f * M_PI;
    return angle;
}

static void euler_to_gravity_body(float pitch, float roll,
                                   float *gx, float *gy, float *gz)
{
    float cp = cosf(pitch);
    *gx = sinf(pitch);
    *gy = -cp * sinf(roll);
    *gz = -cp * cosf(roll);
}

static void rl_build_observation(RLController *rl,
                                  const BalanceInfantry *bal)
{
    ObsVector *obs = &rl->obs;
    obs->gyro[0] = -bal->GyroData[1];
    obs->gyro[1] = bal->GyroData[0];
    obs->gyro[2] = bal->GyroData[2];

    float gx, gy, gz;
    {
        float pitch_raw = bal->Ins_Angle[2];
        float roll_raw  = bal->Ins_Angle[1];
        euler_to_gravity_body(pitch_raw, roll_raw, &gx, &gy, &gz);
    }

    float pitch = safe_asin(gx);
    float roll  = safe_asin(-gy);
    obs->euler[0] = -pitch;
    obs->euler[1] = roll;
\
    rl->pitch = -pitch;
    rl->roll  = roll;

    for (int i = 0; i < 4; i++) {
        obs->commands[i] = rl->commands[i];
    }
    for (int i = 0; i < 6; i++) {
        obs->last_actions[i] = rl->last_actions[i];
    }

    float dof_lf = normalize_angle(bal->phi_left[4]  - (float)PI_6);
    float dof_rf = normalize_angle(bal->phi_right[4] - (float)PI_6);
    float dof_lr = normalize_angle(bal->phi_left[1]  - (float)PI_5_6);
    float dof_rr = normalize_angle(bal->phi_right[1] - (float)PI_5_6);

    obs->dof_pos_err[0] = dof_lf;
    obs->dof_pos_err[1] = dof_rf;
    obs->dof_pos_err[2] = dof_lr;
    obs->dof_pos_err[3] = dof_rr;

    obs->dof_vel[0] = bal->DM8009_Motor[1].para.vel;
    obs->dof_vel[1] = -bal->DM8009_Motor[2].para.vel;
    obs->dof_vel[2] = bal->DM8009_Motor[0].para.vel;
    obs->dof_vel[3] = -bal->DM8009_Motor[3].para.vel;

    obs->wheel_vel[0] = bal->sensors_info.wheel_speed[0];
    obs->wheel_vel[1] = bal->sensors_info.wheel_speed[1];

    obs->theta[0] = normalize_angle(bal->theta_left);
    obs->theta[1] = normalize_angle(bal->theta_right);

    obs->leg_length[0] = bal->L_left[0];
    obs->leg_length[1] = bal->L_right[0];
}

static void rl_obs_normalize(ObsVector *obs,
                              const float mean[OBS_DIM],
                              const float std[OBS_DIM],
                              float eps)
{
    float *f = (float *)obs;
    for (int i = 0; i < OBS_DIM; i++) {
        f[i] = (f[i] - mean[i]) / (std[i] + eps);
    }
}

static void rl_push_history(float enc_input[ENC_INPUT_DIM],
                             float history_buf[ENC_INPUT_DIM],
                             int *history_pos,
                             const ObsVector *obs)
{
    int offset = (*history_pos % OBS_HISTORY_LENGTH) * OBS_DIM;
    memcpy(&history_buf[offset], obs, OBS_DIM * sizeof(float));

    int start = (*history_pos + 1) % OBS_HISTORY_LENGTH;
    for (int i = 0; i < OBS_HISTORY_LENGTH; i++) {
        int src = ((start + i) % OBS_HISTORY_LENGTH) * OBS_DIM;
        int dst = i * OBS_DIM;
        memcpy(&enc_input[dst], &history_buf[src], OBS_DIM * sizeof(float));
    }

    *history_pos += 1;
}

static void rl_action_clip(float action[6])
{
    const float leg_clip   = 100.0f;
    const float wheel_clip = 100.0f;
    const int   leg_idx[]  = {0, 1, 2, 3};
    const int   wheel_idx[] = {4, 5};

    for (int i = 0; i < 4; i++) {
        int j = leg_idx[i];
        if (action[j] >  leg_clip) action[j] =  leg_clip;
        if (action[j] < -leg_clip) action[j] = -leg_clip;
    }
    for (int i = 0; i < 2; i++) {
        int j = wheel_idx[i];
        if (action[j] >  wheel_clip) action[j] =  wheel_clip;
        if (action[j] < -wheel_clip) action[j] = -wheel_clip;
    }
}

/* 诊断: 电机 0 的 CAN 反馈帧率 (Hz)。由 RL_Step 每 10 ms 更新, 打到 VOFA 的 I60。 */
static float dbg_can_fb_hz = 0.0f;

static void rl_action_smooth(float filtered[6], const float raw[6], float alpha)
{
    for (int i = 0; i < 6; i++) {
        filtered[i] = (1.0f - alpha) * raw[i] + alpha * filtered[i];
    }
}

static void rl_compute_vmc_feedforward(float vmc[6],
                                        const float phi_l[5],
                                        const float phi_r[5],
                                        float pitch)
{
    const float L1 = VMC_L1, L2 = VMC_L2, L3 = VMC_L3, L4 = VMC_L4, L5 = VMC_L5;
    const float ff_force = VMC_FEEDFORWARD_FORCE;

    float phi1_l = phi_l[1];
    float phi4_l = phi_l[4];
    float phi1_r = phi_r[1];
    float phi4_r = phi_r[4];

    float phi0_l = 0, phi0_r = 0;

    {
        float xB = L1 * cosf(phi1_l), yB = L1 * sinf(phi1_l);
        float xD = L5 + L4 * cosf(phi4_l), yD = L4 * sinf(phi4_l);
        float dx = xD - xB, dy = yD - yB;
        float A0  = 2 * L2 * dx, b0 = 2 * L2 * dy;
        float C0  = L2*L2 + dx*dx + dy*dy - L3*L3;
        float disc = A0*A0 + b0*b0 - C0*C0;
        if (disc < 0) disc = 0;
        float phi2 = 2 * atan2f(b0 + sqrtf(disc), A0 + C0);
        float xC = xB + L2 * cosf(phi2), yC = yB + L2 * sinf(phi2);
        phi0_l = atan2f(yC, xC - L5/2);
        float phi3 = atan2f(yB - yD + L2*sinf(phi2), xB - xD + L2*cosf(phi2));
        float s1 = sinf(phi3 - phi2), s2 = sinf(phi3 - phi4_l), s3 = sinf(phi1_l - phi2);
        float theta = (float)(M_PI_2) - phi0_l - pitch;
        float ramp = (((float)M_PI/3) - fabsf(theta)) / ((float)M_PI/6);
        if (ramp < 0) ramp = 0; if (ramp > 1) ramp = 1;
        float ct = cosf(theta); if (ct < 0.0707f) ct = 0.0707f;
        float ff = ff_force / ct * ramp;
        float eps = 1e-8f;
        vmc[0] = L4 * sinf(phi0_l - phi2) * s2 / (s1 + eps) * ff;
        vmc[2] = L1 * sinf(phi0_l - phi3) * s3 / (s1 + eps) * ff;
    }
    {
        float xB = L1 * cosf(phi1_r), yB = L1 * sinf(phi1_r);
        float xD = L5 + L4 * cosf(phi4_r), yD = L4 * sinf(phi4_r);
        float dx = xD - xB, dy = yD - yB;
        float A0  = 2 * L2 * dx, b0 = 2 * L2 * dy;
        float C0  = L2*L2 + dx*dx + dy*dy - L3*L3;
        float disc = A0*A0 + b0*b0 - C0*C0;
        if (disc < 0) disc = 0;
        float phi2 = 2 * atan2f(b0 + sqrtf(disc), A0 + C0);
        float xC = xB + L2 * cosf(phi2), yC = yB + L2 * sinf(phi2);
        phi0_r = atan2f(yC, xC - L5/2);
        float phi3 = atan2f(yB - yD + L2*sinf(phi2), xB - xD + L2*cosf(phi2));
        float s1 = sinf(phi3 - phi2), s2 = sinf(phi3 - phi4_r), s3 = sinf(phi1_r - phi2);
        float theta = (float)(M_PI_2) - phi0_r - pitch;
        float ramp = (((float)M_PI/3) - fabsf(theta)) / ((float)M_PI/6);
        if (ramp < 0) ramp = 0; if (ramp > 1) ramp = 1;
        float ct = cosf(theta); if (ct < 0.0707f) ct = 0.0707f;
        float ff = ff_force / ct * ramp;
        float eps = 1e-8f;
        vmc[1] = L4 * sinf(phi0_r - phi2) * s2 / (s1 + eps) * ff;
        vmc[3] = L1 * sinf(phi0_r - phi3) * s3 / (s1 + eps) * ff;
    }
    vmc[4] = 0;
    vmc[5] = 0;

    for (int i = 0; i < 6; i++) {
        if (isnan(vmc[i])) vmc[i] = 0;
        if (vmc[i] >  40) vmc[i] =  40;
        if (vmc[i] < -40) vmc[i] = -40;
    }
}

static int _onnx_run(const float enc_input[ENC_INPUT_DIM],
                      const float obs_norm[OBS_DIM],
                      float action[6])
{
    static ai_handle net = AI_HANDLE_NULL;
    static ai_network_report report;
    static int init_done = 0;

    if (!init_done) {
        const char *name = ai_mnetwork_find(NULL, 0);
        if (!name) return -1;
        ai_error err = ai_mnetwork_create(name, &net, NULL);
        if (err.type) return -1;
        if (!ai_mnetwork_init(net)) return -1;
        if (!ai_mnetwork_get_report(net, &report)) return -1;
        init_done = 1;
    }

    float enc_norm[ENC_INPUT_DIM];
    for (int i = 0; i < ENC_INPUT_DIM; i++) {
        enc_norm[i] = (enc_input[i] - enc_mean[i]) / (enc_std[i] + NORM_EPS);
    }

    ai_buffer ai_input[2];
    ai_input[0] = report.inputs[0];
    ai_input[0].data = AI_HANDLE_PTR(enc_norm);
    ai_input[1] = report.inputs[1];
    ai_input[1].data = AI_HANDLE_PTR(obs_norm);

    ai_buffer ai_output;
    ai_output = report.outputs[0];
    ai_output.data = AI_HANDLE_PTR(action);

    if (ai_mnetwork_run(net, ai_input, &ai_output) != 1) {
        return -1;
    }
    return 0;
}

void RL_Init(RLController *rl)
{
    memset(rl, 0, sizeof(RLController));
    DWT_Init(168);
    rl->initialized = 1;
}

void RL_Reset(RLController *rl)
{
    RL_Init(rl);
}

void RL_Step(RLController *rl, BalanceInfantry *bal)
{
    if (!rl->initialized) {
        RL_Init(rl);
    }

    uint64_t t0 = DWT_GetTimeline_us();
    {
        static uint64_t last_step_us = 0;
        rl->step_dt_us = (last_step_us > 0) ? (float)(t0 - last_step_us) : 0;
        last_step_us = t0;
    }

    /* ── 诊断: 测电机 0 的 CAN 反馈帧率, 结果打到 VOFA 的 I51 (单位 Hz) ──
     * can_fb_cnt0 在 bsp_can.c 的 case 0x0 里 ++。
     *
     * 为什么不用 10 ms 窗口: 500 Hz 下 10 ms 只有 5 帧, 量化步长就是 100 Hz,
     *   读数会在 0/100/…/500 之间乱跳 —— 那是窗口太短, 不是 CAN 故障。
     * 这里累加 20 步(200 ms)再输出一次: 一个窗口约 100 帧, 量化误差降到 ±10 Hz。
     *
     * 正常值: 约 480~500 Hz (dm8009_task 每 2 ms 发一轮 MIT 指令, 电机每轮回一帧)。
     */
    {
        extern volatile uint32_t can_fb_cnt0;
        static uint32_t last_cnt = 0;
        static uint32_t acc_cnt  = 0;
        static uint64_t acc_t0   = 0;
        static uint8_t  decim    = 0;

        uint32_t now_cnt = can_fb_cnt0;
        if (acc_t0 == 0) {
            acc_t0   = t0;
            last_cnt = now_cnt;
        }
        acc_cnt += (now_cnt - last_cnt);
        last_cnt = now_cnt;

        if (++decim >= 20) {                       /* 20 × 10 ms = 200 ms */
            uint64_t dt = t0 - acc_t0;
            if (dt > 0) {
                dbg_can_fb_hz = (float)acc_cnt * 1000000.0f / (float)dt;
            }
            acc_cnt = 0;
            acc_t0  = t0;
            decim   = 0;
        }
    }

    rl->commands[0] = -bal->vx_set;

    /* ── 站立高度档位: 右手自定义按钮 / 键盘R 切换 (见文件顶部 HEIGHT_CMD_*) ──
     * leg_flag 由 ChassisControl.c 的 KEY_R 边沿检测翻转, 上电为 0。
     * 这里加一阶斜坡, 避免按下瞬间给策略一个 9 cm 的阶跃指令。 */
    {
        static float h_cmd = HEIGHT_CMD_DEFAULT;
        float h_tgt  = (bal->leg_flag) ? HEIGHT_CMD_SWITCH : HEIGHT_CMD_DEFAULT;
        float h_step = HEIGHT_CMD_SLEW * CTRL_DT;
        if (h_cmd < h_tgt) {
            h_cmd += h_step;
            if (h_cmd > h_tgt) h_cmd = h_tgt;
        } else if (h_cmd > h_tgt) {
            h_cmd -= h_step;
            if (h_cmd < h_tgt) h_cmd = h_tgt;
        }
        rl->commands[2] = h_cmd;
    }
    rl->commands[3] = 0.0f;

    if (bal->ChassisMode != CHASSIS_FOLLOW_GIMBAL_YAW) {
        rl->commands[1] = rc_ctrl.rc.rocker_l1 * -0.01515f * 0.8;
    } else {
        float yaw_error = bal->chassis_relative_angle;
        float yaw_error_dot = (yaw_error - rl->prev_yaw_error) / CTRL_DT;
        rl->prev_yaw_error = yaw_error;
        if (fabsf(yaw_error) < 0.07f) {
            rl->commands[1] = 0.0f;
        } else {
            rl->commands[1] = yaw_error * 3.0f - yaw_error_dot * 0.0f;
        }
    }

    rl_build_observation(rl, bal);

    float obs_norm[OBS_DIM];
    memcpy(obs_norm, &rl->obs, OBS_DIM * sizeof(float));
    rl_obs_normalize((ObsVector *)obs_norm, obs_mean, obs_std, NORM_EPS);

    rl_push_history(rl->enc_input, rl->history_buf, &rl->history_pos, &rl->obs);

    uint64_t t1 = DWT_GetTimeline_us();

    if (_onnx_run(rl->enc_input, obs_norm, rl->raw_action) != 0) {
        memset(rl->torques, 0, sizeof(rl->torques));
        return;
    }

    uint64_t t2 = DWT_GetTimeline_us();

    float clipped[6];
    memcpy(clipped, rl->raw_action, sizeof(clipped));
    rl_action_clip(clipped);

    /* 动作低通: filtered = (1-alpha)*raw + alpha*filtered
     * alpha=0 完全不平滑(训练时的设定); 这里是诊断性改动, 用来压 20~25 Hz 极限环。
     *
     * 一阶 IIR 的截止频率 fc = -ln(alpha) / (2*pi*CTRL_DT), RL 周期 10 ms:
     *   alpha=0.3 → fc = 19.2 Hz   (温和, 20Hz 处约 -3 dB)
     *   alpha=0.4 → fc = 14.6 Hz   (推荐起点, 20Hz 约 -6 dB)
     *   alpha=0.5 → fc = 11.0 Hz   (较强, 20Hz 约 -9 dB)
     * 平衡动态在 1~3 Hz, alpha=0.4 时那里只有约 -12 度相位滞后, 基本不影响站姿。
     * 如果有效但站姿变软, 往回调到 0.3; 如果无效, 加到 0.5 再试。
     */
    /* 实测结论 (2026-11): alpha=0 会"巨抖" => 这个滤波器不是多余的,
     * 它在压 14 Hz 处的回路增益。它有用, 原因正是 sim-to-real gap:
     *
     *   训练时 MuJoCo 里的 PD 用的是"零延迟的真值状态" —— action_delay_range
     *   只延迟 action, 不延迟 PD 的反馈。而硬件上 PD 用的 dof_pos/dof_vel
     *   是 2~3 ms 前的 CAN 反馈。
     *
     *   这个差别只在 kd 项上致命: 14 Hz 处 ω*kd/kp = 6.6, 阻尼项贡献了相位
     *   里的绝大部分, 而它正好是滞后的那一项 => 回路在 14 Hz 处的增益就贴在 1 附近。
     *   alpha=0.4 的 -2.8 dB 刚好把它压到 1 以下, 所以一去掉就失稳。
     *
     * 所以保留 alpha=0.4。配套改动: LEG_KD 已从 3.0 改回训练值 2.0 ——
     * 那一项本身就是相位最差的部分, 改回后 14 Hz 处增益再降 3.4 dB。
     * 若之后想继续减小 alpha, 考虑用陷波器代替一阶低通, 相位代价小得多。 */
    rl_action_smooth(rl->filtered_action, clipped, 0.4f);

    /* 轮子 [4][5] 不过滤, 直通原始值:
     * 轮子承担慢速平衡(俯仰/前后), 是相位裕度最紧的环, 再给它 ~12 度滞后
     * 就会出现低频摆动。20~25 Hz 极限环主要在关节通道。
     * (如果你测出来慢摆是"腿长上下起伏"而不是"机身前后晃", 那说明是关节环,
     *  这时应该把上面的 alpha 降到 0.3, 而不是放行轮子。) */
    rl->filtered_action[4] = clipped[4];
    rl->filtered_action[5] = clipped[5];

#if (RL_DEBUG_FREEZE_ACTION == 1)
    /* 诊断: 策略输出强制为 0 —— 环路打开, 只剩关节 PD 在工作 */
    memset(rl->filtered_action, 0, sizeof(rl->filtered_action));

#elif (RL_DEBUG_FREEZE_ACTION == 2)
    /* 诊断: 约 1 s 后抓一帧锁死 —— 保持当前姿态, 瞬态更小 */
    {
        static float    frozen[6];
        static uint8_t  got = 0;
        static uint32_t cnt = 0;
        if (!got) {
            if (++cnt >= 100) {
                memcpy(frozen, rl->filtered_action, sizeof(frozen));
                got = 1;
            }
        } else {
            memcpy(rl->filtered_action, frozen, sizeof(frozen));
        }
    }
#endif

    memcpy(rl->last_actions, rl->filtered_action, sizeof(rl->last_actions));

    /* 关节目标限幅: 直接取 ctrl_params.h 的训练限位 (与训练 action 解码一致)
     * 原硬编码 {-0.7,-0.7,-1.6,-1.6}/{0.7,0.7,1.6,1.6} 会把前关节 action 在 ±1.40 处削平,
     * 而训练允许到 ±4.54/±4.80, 削顶会激励 ~20Hz 极限环。 */
    const float lo[4] = {JOINT_LIMIT_LF_LO, JOINT_LIMIT_RF_LO,
                         JOINT_LIMIT_LR_LO, JOINT_LIMIT_RR_LO};
    const float hi[4] = {JOINT_LIMIT_LF_HI, JOINT_LIMIT_RF_HI,
                         JOINT_LIMIT_LR_HI, JOINT_LIMIT_RR_HI};
    for (int i = 0; i < 4; i++) {
        float t = rl->filtered_action[i] * ACTION_SCALE_JOINT;
        if (t > hi[i]) t = hi[i];
        if (t < lo[i]) t = lo[i];
        rl->target_pos[i] = t;
    }
    rl->target_pos[4] = rl->filtered_action[4] * ACTION_SCALE_WHEEL;
    rl->target_pos[5] = rl->filtered_action[5] * ACTION_SCALE_WHEEL;

    uint64_t t3 = DWT_GetTimeline_us();
    rl->timing_us[0] = (float)(t1 - t0);
    rl->timing_us[1] = (float)(t2 - t1);
    rl->timing_us[2] = (float)(t3 - t2);
    rl->timing_us[3] = (float)(t3 - t0);
    rl->step_cnt++;
}

/* 测试区计时起点 = 退出无力模式那一刻 (0 = 还没开始)。
 * 放在文件级是为了让末尾的 VOFA 时间戳也能用它 —— 辨识要的是
 * 「测试开始后的秒数」, 不是「开机后的秒数」, 后者毫无用处。 */
static uint64_t g_traj_t_run0 = 0;

#if TRAJ_TEST_ENABLE
/* ── 精确极坐标逆解: (L, φ0) -> (φ1, φ4) ──
 * 就是 chassis_algorithm.c:11-36 的 InvMotionSolve, 但把连杆数组当参数传入,
 * 这样左右腿共用一个函数 (右腿正解 MotionSolve 用的是同一套公式, 只是 L_right[])。
 *
 * 为什么不用 J_inv / J_l 做线性化: 那份雅可比和 MotionSolve 的正解对不上 ——
 * 在标称位形 (φ1=5π/6, φ4=π/6) 处数值差分得到 ∂L/∂φ1 ≈ -0.38、∂L/∂φ4 ≈ -0.10,
 * 而 J_l[0][1] / J_l[1][1] 都给出 ≈ -0.15。符号和倍数都不一致,
 * 拿它做逆解会把腿挪到错误位形(实测就是这样: 位置不对)。
 * 这里直接解代数闭式, 没有线性化误差, 也不依赖雅可比。
 *
 * 已手算验证: 代入 L=0.28035、φ0=1.5711 得到 φ1=2.61800 (=5π/6)、
 * φ4=0.52477 (≈π/6), 与正解一致。
 *
 * 返回 1 = 成功; 0 = 圆不相交(目标不可达), 调用方应保持原位不动。 */
static int traj_ik_exact(const float *Lk, float L, float phi0,
                         float *phi1_out, float *phi4_out)
{
    const float L1 = Lk[1], L2 = Lk[2], L3 = Lk[3], L4 = Lk[4], L5 = Lk[5];
    float xC, yC, A, B, C, D, E, F, disc, ef;

    if (L < 0.05f) return 0;
    xC = L * cosf(phi0) + L5 * 0.5f;
    yC = L * sinf(phi0);

    /* ── 关节 φ1 (原点-B 支链) ── */
    A = L1 + xC;
    B = L1 * L1 - xC * xC;
    C = L2 * L2 - yC * yC;
    D = L2 * L2 + yC * yC;
    disc = 2.0f * L1 * L1 * D + 2.0f * xC * xC * C - B * B - C * C;
    if (disc < 0.0f) {
        if (disc < -1.0e-4f) return 0;
        disc = 0.0f;                    /* 数值误差导致的小负值, 按相切处理 */
    }
    *phi1_out = 2.0f * atan2f(2.0f * L1 * yC + sqrtf(disc), A * A - C);

    /* ── 关节 φ4 (D 支链) ── */
    A = L3 + L4;
    B = L5 - xC;
    C = L3 - L4;
    D = xC + L4 - L5;
    E = A * A - B * B - yC * yC;
    F = B * B - C * C + yC * yC;
    ef = E * F;
    if (ef < 0.0f) {
        if (ef < -1.0e-4f) return 0;
        ef = 0.0f;
    }
    *phi4_out = 2.0f * atan2f(2.0f * L4 * yC - sqrtf(ef),
                              D * D + yC * yC - L3 * L3);
    return 1;
}
#endif

void RL_ComputeTorque(RLController *rl, BalanceInfantry *bal)
{
    rl->lqr_torques[0] = bal->output_torque_l[1];
    rl->lqr_torques[1] = bal->output_torque_r[1];
    rl->lqr_torques[2] = bal->output_torque_l[2];
    rl->lqr_torques[3] = bal->output_torque_r[2];
    rl->lqr_torques[4] = bal->output_torque_l[0];
    rl->lqr_torques[5] = bal->output_torque_r[0];
	
	    {
#if (VOFA_LOG_ENABLE) && (!TRAJ_TEST_ENABLE)
        /* 辨识模式(TRAJ_TEST_ENABLE=1)下不发这个 67 通道大帧,
         * 改发本文件末尾的精简辨识帧(帧长 272B -> 24B/72B, 不会再因 DMA
         * 忙而丢帧, 采样间隔完全均匀)。 */
        float vofa_torque[67];
        static uint32_t vofa_cnt2 = 0;
        static uint64_t last_vofa_us = 0;
        static float last_act4 = 0;
        static uint64_t last_act4_change_us = 0;
        static uint8_t first_act4 = 1;
        float act4_dt_us = 0;

        static float last_trq4 = 0;
        static uint64_t last_trq4_change_us = 0;
        static uint8_t first_trq4 = 1;
        float trq4_dt_us = 0;
        if (++vofa_cnt2 >= 1) {
            uint64_t now_us = DWT_GetTimeline_us();
            float vofa_dt_us = (last_vofa_us > 0) ? (float)(now_us - last_vofa_us) : 0;
            last_vofa_us = now_us;

            /* 每 500 ms 采一次各任务 CPU 占比 (用绝对时间, 不受 chassis 周期变化影响) */
            {
                static uint64_t cpu_stat_next_us = 0;
                if (now_us >= cpu_stat_next_us) {
                    cpu_stats_update();
                    cpu_stat_next_us = now_us + 500000ULL;
                }
            }

            if (first_act4) {
                first_act4 = 0;
                last_act4 = rl->filtered_action[4];
                last_act4_change_us = now_us;
            } else if (rl->filtered_action[4] != last_act4) {
                act4_dt_us = (float)(now_us - last_act4_change_us);
                last_act4 = rl->filtered_action[4];
                last_act4_change_us = now_us;
            }

            vofa_cnt2 = 0;
            vofa_torque[0] = rl->timing_us[0];
            vofa_torque[1] = rl->timing_us[1];
            vofa_torque[2] = vofa_dt_us;
            vofa_torque[3] = rl->timing_us[3];
            memcpy(&vofa_torque[4],  &rl->obs.gyro[0],      3 * sizeof(float));
            memcpy(&vofa_torque[7],  &rl->obs.euler[0],     2 * sizeof(float));
            memcpy(&vofa_torque[9],  &rl->obs.commands[0],  4 * sizeof(float));
            memcpy(&vofa_torque[13], &rl->obs.last_actions, 6 * sizeof(float));
            memcpy(&vofa_torque[19], &rl->obs.dof_pos_err[0],   4 * sizeof(float));
            memcpy(&vofa_torque[23], &rl->obs.dof_vel[0],       6 * sizeof(float));
            memcpy(&vofa_torque[29], &rl->obs.theta[0],         2 * sizeof(float));
            memcpy(&vofa_torque[31], &rl->obs.leg_length[0],    2 * sizeof(float));
            vofa_torque[33] = bal->output_torque_l[0];
            vofa_torque[34] = bal->output_torque_r[0];
            vofa_torque[35] = bal->output_torque_l[1];
            vofa_torque[36] = bal->output_torque_r[1];
            vofa_torque[37] = bal->output_torque_l[2];
            vofa_torque[38] = bal->output_torque_r[2];
            memcpy(&vofa_torque[39], rl->filtered_action,       6 * sizeof(float));
            memcpy(&vofa_torque[45], rl->target_pos,            6 * sizeof(float));
            /* I51~I56: 原来是 vmc_torques, 因 VMC_FEEDFORWARD_FORCE=0 恒为 0,
             * 改用 CPU 占比 (单位 %)。I51=imu I52=chassis I53=RL I54=dm8009 I55=IDLE I56=其它+中断 */
            /* I51 让位给 CAN 反馈帧率诊断 (CPU 统计已关闭, I51~I56 恒为 0, 无损失) */
            vofa_torque[51] = dbg_can_fb_hz;
            /* I52/I53 改为 leg_flag / stand_done_flag 诊断 (CPU 统计已关闭, 原恒为 0)
             *   I52 = leg_flag        按下 Custom_button_left 应 0 -> 255 翻转
             *   I53 = stand_done_flag 必须到 2, ChassisControl.c:802 那道门才打得开
             * 配合 I11 (= commands[2] 高度指令) 一起看:
             *   I52 翻转 + I11 爬到 0.37  = 整条链路通
             *   I52 不动                   = 卡在 stand_done_flag(看 I53)或按键边沿
             *   I52 翻转但 I11 又掉回 0.28 = 还有别的地方在清 leg_flag */
            vofa_torque[52] = (float)bal->leg_flag;
            vofa_torque[53] = (float)bal->stand_done_flag;
            vofa_torque[54] = cpu_pct_dm8009;
            vofa_torque[55] = cpu_pct_idle;
            vofa_torque[56] = cpu_pct_other;
            vofa_torque[57] = (float)rl->step_cnt;
            vofa_torque[58] = (float)bal->ChassisRC->rc.ch[0];
            vofa_torque[59] = (float)bal->ChassisRC->rc.ch[2];
            vofa_torque[60] = (float)bal->ChassisRC->rc.ch[1];
            memcpy(&vofa_torque[61], rl->torques, 6 * sizeof(float));

            if (first_trq4) {
                first_trq4 = 0;
                last_trq4 = vofa_torque[64];
                last_trq4_change_us = now_us;
            } else if (vofa_torque[64] != last_trq4) {
                trq4_dt_us = (float)(now_us - last_trq4_change_us);
                last_trq4 = vofa_torque[64];
                last_trq4_change_us = now_us;
            }

            vofa_torque[60] = trq4_dt_us;
#if VOFA_DMA_SEND
            vofa_justfloat_output(vofa_torque, 67, &huart1);
#else
            /* 只组帧不发送: 保留全部组帧 CPU 开销, 去掉 UART DMA 和 TxCplt 中断。
             * vofa_dma_busy 永远保持 0, vofa_justfloat_output 从不被调用。 */
            (void)vofa_torque;
#endif
        }
#endif
    }
	
    bal->rl_active = (bal->ChassisMode != CHASSIS_ZERO_FORCE) ? 1 : 0;

    const float kp[4] = {LEG_KP_LF, LEG_KP_RF, LEG_KP_LR, LEG_KP_RR};
    const float kd[4] = {LEG_KD_LF, LEG_KD_RF, LEG_KD_LR, LEG_KD_RR};
    const float wheel_kd[2] = {WHEEL_KD_LW, WHEEL_KD_RW};

    rl->dof_pos[0] = normalize_angle(bal->phi_left[4]  - (float)PI_6);
    rl->dof_pos[1] = normalize_angle(bal->phi_right[4] - (float)PI_6);
    rl->dof_pos[2] = normalize_angle(bal->phi_left[1]  - (float)PI_5_6);
    rl->dof_pos[3] = normalize_angle(bal->phi_right[1] - (float)PI_5_6);

    rl->dof_vel[0] = bal->DM8009_Motor[1].para.vel;
    rl->dof_vel[1] = -bal->DM8009_Motor[2].para.vel;
    rl->dof_vel[2] = bal->DM8009_Motor[0].para.vel;
    rl->dof_vel[3] = -bal->DM8009_Motor[3].para.vel;
    rl->dof_vel[4] = bal->sensors_info.wheel_speed[0];
    rl->dof_vel[5] = bal->sensors_info.wheel_speed[1];

    {
        float pitch = -bal->Ins_Angle[2];
        rl_compute_vmc_feedforward(rl->vmc_torques, bal->phi_left, bal->phi_right, pitch);
    }

#if TRAJ_TEST_ENABLE
    /* ── 测试区: 直接给 (腿长 L, 摆角 φ0) 目标, 关节角由精确逆解自动算 ──
     * 映射和列序说明见文件顶部 "测试区" 注释块。 */
    {
        static float    base_pos[4];

        uint64_t now = DWT_GetTimeline_us();

        /* ── 计时起点是「退出无力模式」那一刻, 不是上电那一刻 ──
         * CHASSIS_ZERO_FORCE 下本函数末尾会把 output_torque 整个清 0, 力矩根本
         * 发不出去。如果从上电就开始计时, 等你切出无力模式时斜坡可能已经跑完。
         *
         * g_traj_t_run0 是文件级变量, 末尾的 VOFA 时间戳也用它, 所以画出来的
         * 第 0 通道是「测试开始后的秒数」, 不是开机秒数。
         * 无力模式一直清 0, 所以重新进无力再退出会自动从头重来一遍。
         *
         * 用时间戳而不是"计数 × 固定周期": 后者假设每拍正好 2 ms, 一旦某拍
         * 被高优先级任务抢占, 计数就和真实时间脱节, 20 s 累积下来偏得出来。 */
        /* 退出无力后第一拍就直接按 (TRAJ_BASE_HEIGHT, TRAJ_BASE_PHI0) 算逆解:
         * 逆解是精确的, 不需要"先站到标称位形站稳再取基准"。
         * 之前那 1.5 s 的等待会让腿先从自由姿态爬 0.3~0.5 rad 回标称位形,
         * 那一段才是"退出无力蹬很久"的来源, 已经删掉。 */

        if (!bal->rl_active) {
            g_traj_t_run0 = 0;
        } else if (g_traj_t_run0 == 0) {
            g_traj_t_run0 = now;
            /* 基准位姿直接取策略 action=0 的站姿 (dof_pos = 0), 不做任何逆解。
             * 几何上这已经就是 (theta=0, L=0.2804):
             *   phi1 = 5π/6, phi4 = π/6  ->  x_C ≈ -0.000082, y_C = 0.280355
             *   L = 0.2804, phi0 = 1.5711, theta = π/2 - phi0 = -0.0003 ≈ 0
             * 所以不需要为了"θ=0 / L=0.28"再经一次逆解挪动它 ——
             * 之前那版逆解把腿挪到了错误位形, 就是因为极坐标雅可比不可信。 */
            base_pos[0] = DEFAULT_DOF_POS_LF;
            base_pos[1] = DEFAULT_DOF_POS_RF;
            base_pos[2] = DEFAULT_DOF_POS_LR;
            base_pos[3] = DEFAULT_DOF_POS_RR;
        }

        float t = (g_traj_t_run0 != 0) ? (float)(now - g_traj_t_run0) * 1.0e-6f : 0.0f;
        t -= TRAJ_DELAY_S;                       /* 退出无力后先空等, 留时间看清状态 */
        if (t < 0.0f) t = 0.0f;
#if !((TRAJ_MODE == 3) && (TRAJ_WAVE == 1))
        /* 正弦/扫频跑完后必须把指令撤回 0。
         * 斜坡模式靠把 t 钳在 TRAJ_DURATION_S 停在终点(这是有意的: 一直加力到
         * 30 rad/s 好观察脱离点)。但正弦模式一旦钳住 t, 相位就成了常数, s 会
         * 冻结在 sinf(终点相位) 这个任意值上 —— 轮子以那个速度无限转下去
         * (终点相位 ≈ 797 rad, sinf ≈ -0.8, A=30 时就是 -24 rad/s 反向飞转)。
         * 所以单记一个 traj_done, 只在正弦模式下把 s 清 0。
         * 只在需要 s 的模式下声明, 避免斜坡+轮速模式下报未使用告警。 */
        uint8_t traj_done = (t > TRAJ_DURATION_S) ? 1 : 0;
#endif
        if (t > TRAJ_DURATION_S) t = TRAJ_DURATION_S;

        /* 目标直接给绝对量: 腿长 = TRAJ_BASE_HEIGHT, 摆角 = PI/2 - pitch (即 theta=0)。
         * 关节角由下面的 traj_ik_exact 自动算, 这里不出现任何偏置。 */
        float Ltl    = TRAJ_BASE_HEIGHT;
        float Ltr    = TRAJ_BASE_HEIGHT;
        float phi0tl = TRAJ_BASE_PHI0 + TRAJ_PHI0_TILT;   /* 0.15 的倾角在这里生效 */
        float phi0tr = phi0tl;
#if TRAJ_PITCH_COMP
        /* 只有显式开启时才扣俯仰 —— 见 TRAJ_PITCH_COMP 注释里那条自激回路 */
        phi0tl -= bal->state_vector_l[4];
        phi0tr  = phi0tl;
#endif
        /* 正弦值 s 只在「腿模式」和「轮速正弦」下用到; 轮速斜坡模式既不需要
         * 正弦也不需要相位, 整段一起关掉, 否则会报 set-but-never-used 告警。 */
#if !((TRAJ_MODE == 3) && (TRAJ_WAVE == 1))
        float phase;
  #if TRAJ_SWEEP
        {
            /* 对数扫频的相位积分: φ(t) = 2π f0 T/ln(f1/f0) * ((f1/f0)^(t/T) - 1) */
            const float r  = TRAJ_F1_HZ / TRAJ_F0_HZ;
            const float w0 = 6.283185307f * TRAJ_F0_HZ * TRAJ_DURATION_S / logf(r);
            phase = w0 * (powf(r, t / TRAJ_DURATION_S) - 1.0f);
        }
  #else
        phase = 6.283185307f * TRAJ_FREQ_HZ * t;
  #endif
#if TRAJ_TRIANGLE
        /* (2/π)·asin(sin(x)): 同样的相位, 波形从正弦换成三角。
         * 幅值仍是 ±1, 周期不变, 上下行都是匀速 ——
         * 这样每一位形都有上行和下行两个点, 可以把摩擦迟滞分离出来。 */
        float s = 0.63661977f * asinf(sinf(phase));
#else
        float s = sinf(phase);
#endif
        if (traj_done) s = 0.0f;    /* 扫频结束: 撤回 0, 不要冻结在终点相位 */
#endif

#if (TRAJ_MODE == 0) || (TRAJ_MODE == 2)
        phi0tl += TRAJ_AMP_ANGLE * s;            /* 摆角: 只摆不伸缩 */
        phi0tr += TRAJ_AMP_ANGLE * s;
#endif
#if (TRAJ_MODE == 1) || (TRAJ_MODE == 2)
        Ltl += TRAJ_AMP_LEN * s;                 /* 腿长: 只伸缩不摆 */
        Ltr += TRAJ_AMP_LEN * s;
#endif

        /* ── 精确逆解: (φ0, L) -> (φ1, φ4), 直接得到 dof_pos 指令 ──
         * dof_pos 的定义见 AI_policy.c:937-940:
         *   LF = φ_left[4]  - PI_6      LR = φ_left[1]  - PI_5_6
         *   RF = φ_right[4] - PI_6      RR = φ_right[1] - PI_5_6
         * 所以目标位姿的 dof_pos 就是「逆解结果 - 偏移」, 直接当 target_pos 用。
         *
         * ！绝对不能减去「实测基准角」。实测基准角是塌陷后的位形(稳态误差约
         *   0.29 rad), 减它等于把这 0.29 rad 整个搬进指令: 指令就变成
         *   「塌陷位形 + 期望增量」, 不再是想要的 (L, φ0)。实测数据:
         *     命令 L = 0.25, 但 target_pos 对应的真实位姿是 L = 0.311。
         * base_pos = 0 对应的就是 dof_pos = 0, 即 φ4 = PI_6、φ1 = PI_5_6 ——
         * 这是常数, 不需要测。 */
        float dphiL_front = 0.0f, dphiL_rear = 0.0f, dphiR_front = 0.0f, dphiR_rear = 0.0f;

        /* 退出无力后第一拍就直接算, 不再等任何稳定时间。
         * 逆解是精确的(见 traj_ik_exact 注释), 不需要先站到标称位形再取基准;
         * 目标就是 (TRAJ_BASE_HEIGHT, TRAJ_BASE_PHI0) 对应的 dof_pos 绝对值。 */
        {
            float p1, p4;
            if (traj_ik_exact(bal->L_left, Ltl, phi0tl, &p1, &p4)) {
                dphiL_front = normalize_angle(p4 - (float)PI_6);     /* LF: φ_left[4]  */
                dphiL_rear  = normalize_angle(p1 - (float)PI_5_6);   /* LR: φ_left[1]  */
            }
            if (traj_ik_exact(bal->L_right, Ltr, phi0tr, &p1, &p4)) {
                dphiR_front = normalize_angle(p4 - (float)PI_6);     /* RF: φ_right[4] */
                dphiR_rear  = normalize_angle(p1 - (float)PI_5_6);   /* RR: φ_right[1] */
            }
        }

        if (TRAJ_MASK & 0x01) rl->target_pos[0] = base_pos[0] + dphiL_front;
        if (TRAJ_MASK & 0x02) rl->target_pos[1] = base_pos[1] + dphiR_front;
        if (TRAJ_MASK & 0x04) rl->target_pos[2] = base_pos[2] + dphiL_rear;
        if (TRAJ_MASK & 0x08) rl->target_pos[3] = base_pos[3] + dphiR_rear;

#if (TRAJ_MODE == 3)
        /* ── 轮速辨识: 四条腿锁在起始位姿, 只给两个轮子速度指令 ──
         * 轮子是速度环: torques[4] = wheel_kd*(target_pos[4] - dof_vel[4])
         * wheel_kd = 0.1, 所以 target_pos[4] 直接就是轮速指令 (rad/s)。
         *
         * TRAJ_WAVE = 0 正弦: 扫频率看幅值比 => 速度环带宽 / 粘滞阻尼
         * TRAJ_WAVE = 1 斜坡: 指令线性缓升, 轮子静止到某一刻突然脱离,
         *              脱离点的力矩 = 静摩擦; 之后 τ 对 ω 的斜率 = 粘滞阻尼
         *
         * 对应辨识精简帧 (8 通道):
         *   ch1 测试秒数  ch2 轮速指令  ch3/ch6 左右实际轮速
         *   ch4/ch7 左右 PD 力矩        ch5/ch8 左右下发力矩
         * ！必须把机器人吊起来, 轮子离地空转, 否则会跑掉而且腿控不住。 */
        {
            float wcmd;
#if TRAJ_WAVE
            wcmd = TRAJ_AMP_WHEEL * (t / TRAJ_DURATION_S);
#else
            wcmd = TRAJ_AMP_WHEEL * s;
#endif
            rl->target_pos[0] = base_pos[0];
            rl->target_pos[1] = base_pos[1];
            rl->target_pos[2] = base_pos[2];
            rl->target_pos[3] = base_pos[3];
            rl->target_pos[4] = wcmd;
            rl->target_pos[5] = wcmd;
        }
#else
        /* 腿测试: 轮子目标置 0 => 只剩阻尼, 不会把整车带走 */
        rl->target_pos[4] = 0.0f;
        rl->target_pos[5] = 0.0f;
#endif
    }
#endif

    for (int i = 0; i < 4; i++) {
        rl->torques[i] = kp[i] * (rl->target_pos[i] - rl->dof_pos[i]) - kd[i] * rl->dof_vel[i];
        rl->torques[i] += rl->vmc_torques[i];
    }

    rl->torques[4] = wheel_kd[0] * (rl->target_pos[4] - rl->dof_vel[4]);
    rl->torques[5] = wheel_kd[1] * (rl->target_pos[5] - rl->dof_vel[5]);

    for (int i = 0; i < 4; i++) {
        if (rl->torques[i] > 40.0f) rl->torques[i] = 40.0f;
        if (rl->torques[i] < -40.0f) rl->torques[i] = -40.0f;
    }
    for (int i = 4; i < 6; i++) {
        if (rl->torques[i] > 4.0f) rl->torques[i] = 4.0f;
        if (rl->torques[i] < -4.0f) rl->torques[i] = -4.0f;
    }

    {
        int unsafe = (fabsf(bal->Ins_Angle[2]) > 1.57079633f ||
                      fabsf(bal->Ins_Angle[1]) > 1.57079633f);
        for (int i = 0; i < 6 && !unsafe; i++) {
            if (rl->torques[i] != rl->torques[i]) unsafe = 1;
        }
        if (unsafe) {
            memset(rl->torques, 0, sizeof(rl->torques));
            memset(rl->vmc_torques, 0, sizeof(rl->vmc_torques));
            memset(bal->output_torque_l, 0, sizeof(bal->output_torque_l));
            memset(bal->output_torque_r, 0, sizeof(bal->output_torque_r));
            return;
        }
    }

    /* 气弹簧补偿: 用每拍实测 L 实时解算, 不做任何冻结。
     * 目的就是让真机等价于「仿真里没有气弹簧」的模型:
     *   物理上气弹簧产生 G, 电机补 -G, 关节上的净作用就等于只有 PD。
     * 所以 ch14~ch17 打的 rl->torques[] (纯 PD) 就是「没有气弹簧时该出多少力」,
     * 也正是和仿真对照该用的那个量。 */
    float gas_torque_l[2], gas_torque_r[2];
    gas_torque_l[0] = bal->J_l[0][1] * bal->F_Gas_left;
    gas_torque_l[1] = bal->J_l[1][1] * bal->F_Gas_left;
    gas_torque_r[0] = bal->J_r[0][1] * bal->F_Gas_right;
    gas_torque_r[1] = bal->J_r[1][1] * bal->F_Gas_right;

    if (gas_torque_l[0] != gas_torque_l[0] || gas_torque_l[1] != gas_torque_l[1] ||
        gas_torque_r[0] != gas_torque_r[0] || gas_torque_r[1] != gas_torque_r[1]) {
        memset(bal->output_torque_l, 0, sizeof(bal->output_torque_l));
        memset(bal->output_torque_r, 0, sizeof(bal->output_torque_r));
        return;
    }

    /* 力矩通道与电机对应关系 (dm8009_task.c:104-107):
     *   output_torque_l[1] -> Motor[1] (phi_left[4]  / φ4 / LF 前)
     *   output_torque_l[2] -> Motor[0] (phi_left[1]  / φ1 / LR 后)
     *   output_torque_r[1] -> Motor[2] (phi_right[4] / φ4 / RF 前)
     *   output_torque_r[2] -> Motor[3] (phi_right[1] / φ1 / RR 后)
     * rl->torques[] 的物理归属见 AI_policy.c:424-432 / 442-447.
     *
     * ── 气弹簧补偿的配对(已按符号逐层核算) ──
     * J_l[0][1] 用 L_left[1]/σ3 = ∂L/∂φ1 -> 后关节
     * J_l[1][1] 用 L_left[4]/σ2 = ∂L/∂φ4 -> 前关节
     * 电机方向: φ = +Motor.pos (左) / -Motor.pos (右), 且 mit_ctrl 传入 -output_torque_*,
     * 所以「φ 方向力矩 = -output_torque_*(1|2)」。
     * 要把气弹簧 G 抵消掉, 需要 φ 方向力矩 = -G + PD, 即
     *     output_torque = +G - PD
     * 实测验证: 对称位形下 G_front = -G_rear (我算的是 +0.15085 / -0.15086),
     * 所以原来的 -G_另一个关节 在标称站姿下恰好也等于 +G_本关节 —— 数值看不出差别。
     * 但 φ0 一偏离 PI/2 这个抵消就破了, 补偿会变成错的, 扫摆角时会咬人。
     * 这里改成「各关节用各自的」, 对称位形下行为与原代码完全一致。 */
    /* ── 退出无力软启动增益, 只乘 PD(反馈)部分 ──
     * 气弹簧补偿不能软化: 它抵消的是一个从一开始就存在的扰动力, 斜坡期间
     * 补偿不足就等于放任气弹簧把腿顶出去。所以 gas_torque 全额立即给,
     * 只有 PD 从 0 拉到 1。
     * 无力: 增益清 0(输出最后还是被下面 memset 清零)。退出后 SOFT_START_MS 毫秒 0 -> 1。 */
    float soft = 1.0f;
    {
        static float soft_s = 0.0f;
        if (!bal->rl_active) {
            soft_s = 0.0f;
        } else if (SOFT_START_MS > 0.0f) {
            soft_s += 0.002f * 1000.0f / SOFT_START_MS;   /* 本函数 500 Hz, 周期 2 ms */
            if (soft_s > 1.0f) soft_s = 1.0f;
        } else {
            soft_s = 1.0f;                                /* 0 = 关闭软启动 */
        }
        soft = soft_s;
    }

    bal->output_torque_l[1] =  gas_torque_l[1] - soft*rl->torques[0]*TORQUE_SCALE;   /* 前 */
    bal->output_torque_l[2] =  gas_torque_l[0] - soft*rl->torques[2]*TORQUE_SCALE;   /* 后 */
    bal->output_torque_r[1] =  gas_torque_r[1] - soft*rl->torques[1]*TORQUE_SCALE;   /* 前 */
    bal->output_torque_r[2] =  gas_torque_r[0] - soft*rl->torques[3]*TORQUE_SCALE;   /* 后 */
    bal->output_torque_l[0] = -soft*rl->torques[4] * TORQUE_SCALE;
    bal->output_torque_r[0] = -soft*rl->torques[5] * TORQUE_SCALE;

    {
        const float clip = 40.0f;
        if (bal->output_torque_l[0] >  clip) bal->output_torque_l[0] =  clip;
        if (bal->output_torque_l[0] < -clip) bal->output_torque_l[0] = -clip;
        if (bal->output_torque_l[1] >  clip) bal->output_torque_l[1] =  clip;
        if (bal->output_torque_l[1] < -clip) bal->output_torque_l[1] = -clip;
        if (bal->output_torque_l[2] >  clip) bal->output_torque_l[2] =  clip;
        if (bal->output_torque_l[2] < -clip) bal->output_torque_l[2] = -clip;
        if (bal->output_torque_r[0] >  clip) bal->output_torque_r[0] =  clip;
        if (bal->output_torque_r[0] < -clip) bal->output_torque_r[0] = -clip;
        if (bal->output_torque_r[1] >  clip) bal->output_torque_r[1] =  clip;
        if (bal->output_torque_r[1] < -clip) bal->output_torque_r[1] = -clip;
        if (bal->output_torque_r[2] >  clip) bal->output_torque_r[2] =  clip;
        if (bal->output_torque_r[2] < -clip) bal->output_torque_r[2] = -clip;
    }

    /* 无力模式: 输出恒 0 (dm8009_task.c:69-77 也会清零, 这里再兜一次) */
    if (!bal->rl_active) {
        memset(bal->output_torque_l, 0, sizeof(bal->output_torque_l));
        memset(bal->output_torque_r, 0, sizeof(bal->output_torque_r));
    }

#if (VOFA_LOG_ENABLE) && (TRAJ_TEST_ENABLE)
    /* ══════════════════════════════════════════════════════════════════
     *  辨识专用精简帧 (只在 TRAJ_TEST_ENABLE=1 时发)
     *
     *  为什么单独一帧:
     *    67 通道的帧 = 272 字节, 在 921600 波特下 2.95 ms/帧, 长于本任务
     *    的 2~3 ms 周期, DMA 忙时会丢帧, 采样变成不确定的。
     *    精简到 8/17 通道后只有 36/72 字节, 远短于周期, 一帧不丢,
     *    采样率等于任务周期且完全均匀 —— 这对后续做 FFT / 相位差是必需的。
     *
     *  VOFA+ 里要按下面设通道数 (JustFloat 帧本身不带通道号):
     *
     *    轮速模式 (TRAJ_MODE == 3)  共 8 通道
     *      ch1  [0] t_L       测试开始后秒数 (退出无力时归 0)
     *      ch2  [1] w_cmd     轮速指令 (rad/s, 左右相同)
     *      ch3  [2] w_act_L   左轮实际轮速 (rad/s)
     *      ch4  [3] tau_pd_L  左轮 PD 力矩 (N·m)
     *      ch5  [4] tau_out_L 左轮下发力矩 (N·m, = -tau_pd_L)
     *      ch6  [5] w_act_R   右轮实际轮速 (rad/s)
     *      ch7  [6] tau_pd_R  右轮 PD 力矩 (N·m)
     *      ch8  [7] tau_out_R 右轮下发力矩 (N·m, = -tau_pd_R)
     *
     *    腿模式 (TRAJ_MODE = 0/1/2) 共 25 通道
     *      ch1  [0]      t        测试开始后秒数
     *      [1..4]   指令 target_pos[0..3]      (LF RF LR RR)
     *      [5..8]   实际 dof_pos[0..3]
     *      [9..12]  速度 dof_vel[0..3]
     *      [13..16] PD 力矩 torques[0..3]
      *      ch18~ch21 [17..20] 实际腿长/摆角: L_left, L_right, phi_left[0], phi_right[0]
     *      ch22~ch25 [21..24] 气弹簧力矩: gas_l[0] 后, gas_l[1] 前, gas_r[0] 后, gas_r[1] 前
      *                           ← 直接看"腿到底变长还是变短", 不用靠肉眼
     *
     *  放在函数最后, 保证力矩是本次算出来的, 不是上一拍残留的。
     * ══════════════════════════════════════════════════════════════════ */
    {
        static float vofa_id[25];
        static uint32_t vofa_decim_cnt = 0;

        /* 降频: 每 TRAJ_VOFA_DECIM 拍才发一帧 (这块是函数最后一段, 可以直接 return)。
         * 时间戳取的是真实时间, 所以降频不影响时间轴的正确性, 只是点变稀。 */
        if (++vofa_decim_cnt < TRAJ_VOFA_DECIM) return;
        vofa_decim_cnt = 0;

        /* 第 0 通道 = 「测试开始后的秒数」, 不是开机秒数。
         * 退出无力那一刻归 0, 无力模式下恒为 0。开机秒数对辨识没有用处,
         * 而且会一直涨到几百秒, 在图上把有用的前 23 秒压缩得看不见。
         * 用 double 算再转 float: (float)us 在 >16.7e6 之后已经不精确。 */
        vofa_id[0] = (g_traj_t_run0 != 0)
                   ? (float)((double)(DWT_GetTimeline_us() - g_traj_t_run0) * 1.0e-6)
                   : 0.0f;

        /* 注意: 这里用 rl->dof_pos / rl->dof_vel / rl->dof_vel[4],
         * 不用 rl->obs.* —— obs 里的那几个量是 RL_Step(100 Hz) 刷新的,
         * 而本函数是 chassis task(~400 Hz) 在跑; 用 obs 的话「实际值」会
         * 比「指令」旧最多 10 ms, 画在同一张图上会凭空多出一个假滞后。
         * rl->dof_pos / dof_vel 就在本函数开头刷新, 与指令同一拍。 */
#if (TRAJ_MODE == 3)
        vofa_id[1] = rl->target_pos[4];        /* ch2 轮速指令 (左右相同) rad/s */
        vofa_id[2] = rl->dof_vel[4];           /* ch3 左轮实际轮速 rad/s */
        vofa_id[3] = rl->torques[4];           /* ch4 左轮 PD 力矩  N·m  */
        vofa_id[4] = bal->output_torque_l[0];  /* ch5 左轮下发力矩  N·m (= -ch4) */
        vofa_id[5] = rl->dof_vel[5];           /* ch6 右轮实际轮速 rad/s */
        vofa_id[6] = rl->torques[5];           /* ch7 右轮 PD 力矩  N·m  */
        vofa_id[7] = bal->output_torque_r[0];  /* ch8 右轮下发力矩  N·m (= -ch7) */
  #if VOFA_DMA_SEND
        vofa_justfloat_output(vofa_id, 8, &huart1);
  #endif
#else
        for (int i = 0; i < 4; i++) {
            vofa_id[1 + i]  = rl->target_pos[i];   /* 指令 */
            vofa_id[5 + i]  = rl->dof_pos[i];      /* 实际角 */
            vofa_id[9 + i]  = rl->dof_vel[i];      /* 角速度 */
            vofa_id[13 + i] = rl->torques[i];      /* PD 力矩 */
        }
        /* 直接读出腿长和摆角, 用来核对"命令 0.25 到底腿变长还是变短"。
         * 这两个量由 MotionSolve 每拍正解出来, 和 dof_pos 同一拍。 */
        vofa_id[17] = bal->L_left[0];          /* ch18 左腿实际腿长 (m) */
        vofa_id[18] = bal->L_right[0];         /* ch19 右腿实际腿长 (m) */
        vofa_id[19] = bal->phi_left[0];        /* ch20 左腿实际摆角 (rad) */
        vofa_id[20] = bal->phi_right[0];       /* ch21 右腿实际摆角 (rad) */

        /* 气弹簧力矩。气弹簧物理上一直存在, 辨识时真正作用在关节上的
         * 是「气弹簧 + PD」, 只取 torques 会漏掉这一项。 */
        vofa_id[21] = gas_torque_l[0];         /* ch22 左后 φ1 气弹簧力矩 (N·m) */
        vofa_id[22] = gas_torque_l[1];         /* ch23 左前 φ4 气弹簧力矩 (N·m) */
        vofa_id[23] = gas_torque_r[0];         /* ch24 右后 φ1 气弹簧力矩 (N·m) */
        vofa_id[24] = gas_torque_r[1];         /* ch25 右前 φ4 气弹簧力矩 (N·m) */
  #if VOFA_DMA_SEND
        vofa_justfloat_output(vofa_id, 25, &huart1);
  #endif
#endif
    }
#endif
}