/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: mcu fault service
 */

#pragma once

#include "hozon/netaos/v1/socfaultservice_skeleton.h"
#include <memory>


namespace hozon {
namespace netaos {
namespace phm_server {


class PhmInterfaceFaultFromMcu : public hozon::netaos::v1::skeleton::SocFaultServiceSkeleton
{
public:
    using SocFaultServiceSkeleton = hozon::netaos::v1::skeleton::SocFaultServiceSkeleton;

    static PhmInterfaceFaultFromMcu* getInstance();
    virtual ~PhmInterfaceFaultFromMcu();

    void Init();
    void DeInit();

    // method
    virtual ara::core::Future<hozon::netaos::v1::skeleton::methods::SocFaultService::FaultEventReport::Output>
        FaultEventReport(const ::hozon::netaos::FaultEvent& FaultRecvData);

private:
    PhmInterfaceFaultFromMcu();
    PhmInterfaceFaultFromMcu(const PhmInterfaceFaultFromMcu&);
    PhmInterfaceFaultFromMcu& operator=(const PhmInterfaceFaultFromMcu&);

    static std::mutex mtx_;
    static PhmInterfaceFaultFromMcu* instance_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
