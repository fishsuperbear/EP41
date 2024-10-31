/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_TRAFFICLIGHT_TRAFFICLIGHTINTERFACE_COMMON_H
#define MDC_TRAFFICLIGHT_TRAFFICLIGHTINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/trafficlight/impl_type_trafficlightarray.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace trafficlight {

class TrafficLightInterface {
public:
    constexpr TrafficLightInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/FunctionalSoftware/TrafficLightServiceInterface/TrafficLightInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace trafficlight
} // namespace mdc

#endif // MDC_TRAFFICLIGHT_TRAFFICLIGHTINTERFACE_COMMON_H
