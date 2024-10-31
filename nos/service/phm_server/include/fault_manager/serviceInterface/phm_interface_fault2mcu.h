/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: mcu fault service
 */

#pragma once

#include "hozon/netaos/v1/mcufaultservice_proxy.h"
#include "phm_server/include/common/phm_server_def.h"
#include <string>

namespace hozon {
namespace netaos {
namespace phm_server {


class PhmInterfaceFault2mcu
{
public:
    PhmInterfaceFault2mcu();
    ~PhmInterfaceFault2mcu();

    void Init();
    void DeInit();

    void serviceAvailabilityCallback( ara::com::ServiceHandleContainer<ara::com::HandleType> handles);
    bool FaultReportToMCU(const hozon::netaos::phm_server::HzFaultEventToMCU& faultData);
    void FaultToHMIToMCU(const uint64_t faultData);

    void Send2McuCheck();
    void SendSig(const bool reportToMcuResult);
    void DoCmdGetPid(const std::string& cmd, std::string& outPid);

private:
    PhmInterfaceFault2mcu(const PhmInterfaceFault2mcu&);
    PhmInterfaceFault2mcu& operator=(const PhmInterfaceFault2mcu&);

    std::shared_ptr<hozon::netaos::v1::proxy::McuFaultServiceProxy> proxy_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
