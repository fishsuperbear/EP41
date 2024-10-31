/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_BODY_BODYREPORTSERVICEINTERFACE_COMMON_H
#define ARA_BODY_BODYREPORTSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/body/impl_type_bodyreportmsg.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace body {

class BodyReportServiceInterface {
public:
    constexpr BodyReportServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/BodyServiceInterface/BodyReportServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace body
} // namespace ara

#endif // ARA_BODY_BODYREPORTSERVICEINTERFACE_COMMON_H
