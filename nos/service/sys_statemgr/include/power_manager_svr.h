/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: ssm
 * Created on: Oct 24, 2023
 * Author: aviroz
 */

#ifndef POWER_MANAGER_SERVICE_H
#define POWER_MANAGER_SERVICE_H

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <signal.h>
#include <future>
#include "hozon/netaos/v1/socpowerservice_skeleton.h"
#include <ara/core/initialization.h>
#include "ara/core/promise.h"

namespace hozon {
namespace netaos {
namespace ssm {

class StateManager;

class PowerManagerServiceSvr : public hozon::netaos::v1::skeleton::SocPowerServiceSkeleton {
public:
    using SocPowerServiceSkeleton = hozon::netaos::v1::skeleton::SocPowerServiceSkeleton;

    PowerManagerServiceSvr(std::string instance);
    virtual ~PowerManagerServiceSvr();

    void Init(std::shared_ptr<StateManager>);
    void DeInit();
    void Run();

private:
    void TriggerSysStateEvent();

private:
    std::shared_ptr<StateManager>  m_smgr;
    std::thread m_thr_pms;
    sig_atomic_t m_stopFlag = 0;
    uint8_t m_sys_state;
    uint8_t m_sm_state;

};


}}}
#endif
