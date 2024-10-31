/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_CHASSISREPORTSERVICEINTERFACE_COMMON_H
#define ARA_CHASSIS_CHASSISREPORTSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/chassis/impl_type_chassisreportmsg.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace chassis {

class ChassisReportServiceInterface {
public:
    constexpr ChassisReportServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/ChassisServiceInterface/ChassisReportServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace chassis
} // namespace ara

#endif // ARA_CHASSIS_CHASSISREPORTSERVICEINTERFACE_COMMON_H
