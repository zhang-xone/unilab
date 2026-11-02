/**
  *********************************************************************
  * @file      ai_policy_task.c/h
  * @brief     AI policy task, 20ms?????推理, 输VMC馈力矩???
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */

#include "ai_policy_task.h"
#include "AI_policy.h"
#include "ChassisControl.h"
#include "cmsis_os.h"

extern BalanceInfantry balance_infantry;
RLController rl;

void RL_Task(void const *pvParameters)
{
    UBaseType_t uxHighWaterMark;
    uint32_t prev_wake_time;

    vTaskDelay(500);
    RL_Init(&rl);

    prev_wake_time = osKernelSysTick();
    while (1)
    {
        RL_Step(&rl, &balance_infantry);
        uxHighWaterMark = uxTaskGetStackHighWaterMark(NULL);
        osDelayUntil(&prev_wake_time, 10);
    }
}