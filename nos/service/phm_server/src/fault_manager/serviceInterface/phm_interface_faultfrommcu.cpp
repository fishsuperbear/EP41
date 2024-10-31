
#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <future>
#include <ara/core/initialization.h>
#include "ara/core/promise.h"
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/fault_manager/manager/phm_fault_dispatcher.h"
#include "phm_server/include/fault_manager/serviceInterface/phm_interface_faultfrommcu.h"

namespace hozon {
namespace netaos {
namespace phm_server {


PhmInterfaceFaultFromMcu* PhmInterfaceFaultFromMcu::instance_ = nullptr;
std::mutex PhmInterfaceFaultFromMcu::mtx_;
std::mutex recvFaultMtx_;

PhmInterfaceFaultFromMcu*
PhmInterfaceFaultFromMcu::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new PhmInterfaceFaultFromMcu();
        }
    }

    return instance_;
}


PhmInterfaceFaultFromMcu::PhmInterfaceFaultFromMcu()
: SocFaultServiceSkeleton::SocFaultServiceSkeleton(ara::com::InstanceIdentifier("1"))
{
    PHMS_INFO <<"PhmInterfaceFaultFromMcu::PhmInterfaceFaultFromMcu";
}

PhmInterfaceFaultFromMcu::~PhmInterfaceFaultFromMcu()
{
    PHMS_INFO <<"PhmInterfaceFaultFromMcu::~PhmInterfaceFaultFromMcu";
}

void
PhmInterfaceFaultFromMcu::Init()
{
    PHMS_INFO << "PhmInterfaceFaultFromMcu::Init";
}

void
PhmInterfaceFaultFromMcu::DeInit()
{
    PHMS_INFO << "PhmInterfaceFaultFromMcu::DeInit";
}

ara::core::Future<hozon::netaos::v1::skeleton::methods::SocFaultService::FaultEventReport::Output>
PhmInterfaceFaultFromMcu::FaultEventReport(const ::hozon::netaos::FaultEvent& FaultRecvData)
{
    std::lock_guard<std::mutex> lck(recvFaultMtx_);
    PHMS_INFO << "PhmInterfaceFaultFromMcu::FaultEventReport"
              << ",faultId: " << FaultRecvData.faultId
              << ",faultObj: " << FaultRecvData.faultObj
              << ",faultStatus: " << FaultRecvData.faultStatus
              << ",faultTime: " << FaultRecvData.faultTime
              << ",faultUtil: " << FaultRecvData.faultUtil;

    uint64_t local_time = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    Fault_t fault;
    fault.faultId = FaultRecvData.faultId;
    fault.faultObj = FaultRecvData.faultObj;
    fault.faultStatus = FaultRecvData.faultStatus;
    fault.faultOccurTime = local_time;
    fault.faultDomain = "MCU";
    FaultDispatcher::getInstance()->ReportFault(fault);

    std::unique_ptr<ara::core::Promise<hozon::netaos::v1::skeleton::methods::SocFaultService::FaultEventReport::Output>>
        prom(std::make_unique<ara::core::Promise<hozon::netaos::v1::skeleton::methods::SocFaultService::FaultEventReport::Output>>());
    hozon::netaos::v1::skeleton::methods::SocFaultService::FaultEventReport::Output res{hozon::netaos::ResultEnum::OK};
    prom->set_value(std::move(res));
    return prom->get_future();
}


}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
