/**
  *********************************************************************
  * @file      ai_policy_task.c/h
  * @brief     AI policy task, 20ms周期推理???
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */
#ifndef AI_POLICY_TASK_H
#define AI_POLICY_TASK_H

#include "main.h"
#include "FreeRTOS.h"
#include "task.h"
#include "AI_policy.h"

extern RLController rl;

void RL_Task(void const *pvParameters);

#endif