/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_BODYCMD_BODYCOMMANDSERVICEINTERFACE_COMMON_H
#define ARA_BODYCMD_BODYCOMMANDSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/body/impl_type_bodycommandmsg.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace bodycmd {

class BodyCommandServiceInterface {
public:
    constexpr BodyCommandServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/BodyServiceInterface/BodyCommandServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace bodycmd
} // namespace ara

#endif // ARA_BODYCMD_BODYCOMMANDSERVICEINTERFACE_COMMON_H
