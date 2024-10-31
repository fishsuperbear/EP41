/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_VEHICLESERVICEINTERFACE_COMMON_H
#define ARA_VEHICLE_VEHICLESERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/vehicle/impl_type_vehicleinfo.h"
#include "ara/vehicle/impl_type_flcinfo.h"
#include "ara/vehicle/impl_type_flrinfo.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace vehicle {

class VehicleServiceInterface {
public:
    constexpr VehicleServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/VehicleServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace vehicle
} // namespace ara

#endif // ARA_VEHICLE_VEHICLESERVICEINTERFACE_COMMON_H
