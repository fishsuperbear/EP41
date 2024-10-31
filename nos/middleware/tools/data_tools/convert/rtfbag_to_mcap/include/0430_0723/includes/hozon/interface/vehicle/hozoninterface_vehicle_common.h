/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_VEHICLE_HOZONINTERFACE_VEHICLE_COMMON_H
#define HOZON_INTERFACE_VEHICLE_HOZONINTERFACE_VEHICLE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/vehicle/impl_type_vehicleinfo.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace vehicle {

class HozonInterface_Vehicle {
public:
    constexpr HozonInterface_Vehicle() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Vehicle");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace vehicle
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_VEHICLE_HOZONINTERFACE_VEHICLE_COMMON_H
