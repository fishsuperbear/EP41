/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSISCMD_CHASSISCOMMANDSERVICEINTERFACE_COMMON_H
#define ARA_CHASSISCMD_CHASSISCOMMANDSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/chassis/impl_type_chassiscommandmsg.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace chassiscmd {

class ChassisCommandServiceInterface {
public:
    constexpr ChassisCommandServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/ChassisServiceInterface/ChassisCommandServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace chassiscmd
} // namespace ara

#endif // ARA_CHASSISCMD_CHASSISCOMMANDSERVICEINTERFACE_COMMON_H
