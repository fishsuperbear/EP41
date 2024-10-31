/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: fault strategy notify to app
*/

#include "phm_server/include/fault_manager/strategy/phm_strategy_notify2app.h"
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/fault_manager/manager/phm_fault_dispatcher.h"

namespace hozon {
namespace netaos {
namespace phm_server {


PhmStrategyNotify2App::PhmStrategyNotify2App()
{
}

PhmStrategyNotify2App::~PhmStrategyNotify2App()
{
}

void
PhmStrategyNotify2App::Init()
{
    PHMS_INFO << "PhmStrategyNotify2App::Init";
}

void
PhmStrategyNotify2App::DeInit()
{
    PHMS_INFO << "PhmStrategyNotify2App::DeInit";
}

void
PhmStrategyNotify2App::Act(const FaultInfo& faultData)
{
    PHMS_INFO << "PhmStrategyNotify2App::Act";
    SendFaultPack fault;
    fault.faultId = faultData.faultId;
    fault.faultObj = faultData.faultObj;
    fault.faultStatus = faultData.faultStatus;
    fault.faultDomain = faultData.faultDomain;
    fault.faultOccurTime = faultData.faultOccurTime;
    FaultDispatcher::getInstance()->SendFault(fault);
}



}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
