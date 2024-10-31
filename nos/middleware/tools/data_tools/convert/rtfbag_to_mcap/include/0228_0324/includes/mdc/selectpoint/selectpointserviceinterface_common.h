/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SELECTPOINT_SELECTPOINTSERVICEINTERFACE_COMMON_H
#define MDC_SELECTPOINT_SELECTPOINTSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/selectpoint/impl_type_selectpoint.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace selectpoint {

class SelectPointServiceInterface {
public:
    constexpr SelectPointServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/FunctionalSoftware/SelectPointServiceInterface/SelectPointServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace selectpoint
} // namespace mdc

#endif // MDC_SELECTPOINT_SELECTPOINTSERVICEINTERFACE_COMMON_H
