/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: ssm
 * Created on: Oct 26, 2023
 * Author: aviroz
 */

#ifndef MCUSTATE_SERVICE_CLI_H
#define MCUSTATE_SERVICE_CLI_H

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <mutex>
#include <future>
#include "hozon/netaos/v1/mcustateservice_proxy.h"
#include <ara/core/initialization.h>
#include "ara/core/promise.h"

namespace hozon {
namespace netaos {
namespace ssm {

using namespace hozon::netaos::v1::proxy;
using McuStateServiceProxy = hozon::netaos::v1::proxy::McuStateServiceProxy;

class StateManager;

class McuStateServiceCli {
public:

    McuStateServiceCli(std::string);
    virtual ~McuStateServiceCli();

    void Init(std::shared_ptr<StateManager>);
    void DeInit();
    void Run();
    int32_t SocPowerModeRequest(hozon::netaos::PowerModeEnum& powermode);

private:
    void StartFindService();
    void StopFindService();

    void ServiceAvailabilityCallback(ara::com::ServiceHandleContainer<ara::com::HandleType> handles);
    void OnRecvSysStateCallback();

    void TriggerSysStateEvent();
private:
    std::shared_ptr<StateManager>  m_smgr;
    std::string m_instance_id;
    std::mutex m_proxy_mtx;
    ara::com::FindServiceHandle m_handle;
    std::shared_ptr<McuStateServiceProxy>  m_proxy;
    uint8_t m_l2_state;
    uint8_t m_l3_state;

};


}}}
#endif
