/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_LOCATION_LOCATIONSERVICEINTERFACE_COMMON_H
#define MDC_LOCATION_LOCATIONSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/location/impl_type_location.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace location {

class LocationServiceInterface {
public:
    constexpr LocationServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/FunctionalSoftware/LocationServiceInterface/LocationServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace location
} // namespace mdc

#endif // MDC_LOCATION_LOCATIONSERVICEINTERFACE_COMMON_H
