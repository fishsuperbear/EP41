// Generated automatically by python 3.8
#ifndef _DSV_PPS_UPDATE_MANAGER_H_
#define _DSV_PPS_UPDATE_MANAGER_H_

#include "desay/include/ppscontrol.h"
#include "desay/include/DsvPpsIfStruct.h"

void initPPS();

Dsvpps::ParameterString &getStartUpdateRequest2DSV();
void sendStartUpdateRequest2DSVData();

Dsvpps::PlaceHolder &getGetUpdateStatusRequest2DSV();
void sendGetUpdateStatusRequest2DSVData();

Dsvpps::PlaceHolder &getGetVersionRequest2DSV();
void sendGetVersionRequest2DSVData();

Dsvpps::PlaceHolder &getPartitionSwitchRequest2DSV();
void sendPartitionSwitchRequest2DSVData();

Dsvpps::PlaceHolder &getGetCurrentPartitionRequest2DSV();
void sendGetCurrentPartitionRequest2DSVData();

Dsvpps::PlaceHolder &getRebootSystemRequest2DSV();
void sendRebootSystemRequest2DSVData();

int32_t Get_Code_StartUpdate();
int32_t Get_Code_SwitchSlot();
std::string Get_Code_GetCurrentSlot();
std::string Get_Code_GetVersion();
int32_t Get_Code_Reboot();

int32_t Get_UpdateState();
int32_t Get_Update_Progress();

#endif // _DSV_PPS_UPDATE_MANAGER_H_
