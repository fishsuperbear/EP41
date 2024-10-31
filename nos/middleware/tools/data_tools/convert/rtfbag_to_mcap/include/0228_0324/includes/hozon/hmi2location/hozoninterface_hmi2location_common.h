/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI2LOCATION_HOZONINTERFACE_HMI2LOCATION_COMMON_H
#define HOZON_HMI2LOCATION_HOZONINTERFACE_HMI2LOCATION_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/hmi2location/impl_type_alghmiavplocframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmi2location {

class HozonInterface_Hmi2Location {
public:
    constexpr HozonInterface_Hmi2Location() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HozonInterface_Hmi2Location");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace hmi2location
} // namespace hozon

#endif // HOZON_HMI2LOCATION_HOZONINTERFACE_HMI2LOCATION_COMMON_H
