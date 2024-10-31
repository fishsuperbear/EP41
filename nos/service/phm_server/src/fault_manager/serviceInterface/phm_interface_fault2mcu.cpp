/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: Send fault cluster level to mcu
*/

#include "phm_server/include/fault_manager/serviceInterface/phm_interface_fault2mcu.h"
#include "phm_server/include/common/phm_server_logger.h"
#include <iostream>
#include <future>
#include <functional>
#include <ara/core/initialization.h>
#include <signal.h>

namespace hozon {
namespace netaos {
namespace phm_server {


PhmInterfaceFault2mcu::PhmInterfaceFault2mcu()
{
    PHMS_INFO << "PhmInterfaceFault2mcu::PhmInterfaceFault2mcu";
}

PhmInterfaceFault2mcu::~PhmInterfaceFault2mcu()
{
    PHMS_INFO << "PhmInterfaceFault2mcu::~PhmInterfaceFault2mcu";
}

void
PhmInterfaceFault2mcu::Init()
{
    PHMS_INFO << "PhmInterfaceFault2mcu::Init";
    // ara::core::Initialize();

    hozon::netaos::v1::proxy::McuFaultServiceProxy::StartFindService(
        [this]( ara::com::ServiceHandleContainer<ara::com::HandleType> handles, ara::com::FindServiceHandle handler ) {
            (void) handler;
            PHMS_INFO << "PhmInterfaceFault2mcu::Init StartFindService size:" << handles.size();
            PhmInterfaceFault2mcu::serviceAvailabilityCallback( std::move( handles ) );
        },
        ara::com::InstanceIdentifier("1")
    );

    PHMS_INFO << "PhmInterfaceFault2mcu::Init end";
}

void
PhmInterfaceFault2mcu::DeInit()
{
    PHMS_INFO << "PhmInterfaceFault2mcu::DeInit";
}

void
PhmInterfaceFault2mcu::serviceAvailabilityCallback(ara::com::ServiceHandleContainer<ara::com::HandleType> handles)
{
    PHMS_INFO << "PhmInterfaceFault2mcu::serviceAvailabilityCallback";
    if (handles.size() > 0U) {
        if (proxy_ == nullptr) {
            PHMS_INFO << "PhmInterfaceFault2mcu::serviceAvailabilityCallback created proxy";
            proxy_ = std::make_shared<hozon::netaos::v1::proxy::McuFaultServiceProxy>( handles[ 0 ] );
            if ( proxy_ == nullptr ) {
                PHMS_INFO << "PhmInterfaceFault2mcu::serviceAvailabilityCallback create proxy failed";
            }
            Send2McuCheck();
        }
    }
    else {
        PHMS_INFO << "PhmInterfaceFault2mcu::serviceAvailabilityCallback service disconnected\n";
        proxy_ = nullptr;
    }
}

void
PhmInterfaceFault2mcu::Send2McuCheck()
{
    PHMS_INFO << "PhmInterfaceFault2mcu::Send2McuCheck start.";
    std::thread Send2McuTestThd([this] {
        HzFaultEventToMCU toMcuData;
        toMcuData.faultId = 4920;
        toMcuData.faultObj = 4;
        toMcuData.faultStatus = 0;
        memset(toMcuData.postProcessArray, 0, 60);

        bool result = FaultReportToMCU(toMcuData);
        if (!result) {
            PHMS_INFO << "PhmInterfaceFault2mcu::Send2McuCheck failed";
            return;
        }

        SendSig(result);
        return;
    });

    Send2McuTestThd.detach();
    return;
}

void
PhmInterfaceFault2mcu::DoCmdGetPid(const std::string& cmd, std::string& outPid)
{
    FILE* pFile = popen(cmd.c_str(), "r");
    char* buff = new char[256];
    memset(buff, 0, 256);
    fread(buff, 1, sizeof(buff), pFile);
    outPid = buff;
    pclose(pFile);
    delete[] buff;
    return;
}

void
PhmInterfaceFault2mcu::SendSig(const bool reportToMcuResult)
{
    const std::string& procName = "extwdg";
    std::string cmd = "ps -aux | grep " + procName + " | grep -v grep | awk '{print $2}'";
    std::string strPid;
    DoCmdGetPid(cmd, strPid);
    PHMS_INFO << "PhmInterfaceFault2mcu::SendSig cmd:" << cmd << ",pid:" << strPid;

    if (strPid.empty()) {
        PHMS_INFO << "PhmInterfaceFault2mcu::SendSig not exist process";
        return;
    }

    union sigval sv;
    sv.sival_int = (true == reportToMcuResult) ? 0 : 1;
    int result = sigqueue(std::stoi(strPid), 38, sv);
    if (0 != result) {
        PHMS_ERROR << "PhmInterfaceFault2mcu::SendSig failed result:" << result;
    }

    return;
}

bool
PhmInterfaceFault2mcu::FaultReportToMCU(const hozon::netaos::phm_server::HzFaultEventToMCU& faultData)
{
    PHMS_INFO << "PhmInterfaceFault2mcu::FaultReportToMCU start.";
    if (proxy_ == nullptr) {
        PHMS_INFO << "PhmInterfaceFault2mcu::FaultReportToMCU proxy_ is null";
        return false;
    }

    hozon::netaos::FaultDataStruct FaultData;
    FaultData.faultId = faultData.faultId;
    FaultData.faultObj = faultData.faultObj;
    FaultData.faultStatus = faultData.faultStatus;
    for (size_t i = 0; i < 60; ++i) {
        FaultData.postProcessArray[i] = faultData.postProcessArray[i];
    }

    auto FaultReportResult = proxy_->FaultReport(FaultData);
    if (ara::core::future_status::timeout == FaultReportResult.wait_for(std::chrono::milliseconds(50))) {
        PHMS_WARN << "PhmInterfaceFault2mcu::FaultReportToMCU report fault future timeout.";
        return false;
    }

    auto Future = FaultReportResult.GetResult();
    if (Future.HasValue()) {
        auto output = Future.Value();
        PHMS_INFO << "PhmInterfaceFault2mcu::FaultReportToMCU result:" << static_cast<int>(output.FaultReportResult);
        return true;
    }

    PHMS_WARN << "PhmInterfaceFault2mcu::FaultReportToMCU not has value";
    return false;
}

void
PhmInterfaceFault2mcu::FaultToHMIToMCU(const uint64_t faultData)
{
    if (proxy_ == nullptr) {
        PHMS_INFO << "PhmInterfaceFault2mcu::FaultToHMIToMCU proxy_ is null.";
        return;
    }

    proxy_->FaultToHMI(faultData);
    return;
}


}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
