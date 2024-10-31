/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_PARKINGLOT_HOZONINTERFACE_PARKINGLOT_COMMON_H
#define HOZON_INTERFACE_PARKINGLOT_HOZONINTERFACE_PARKINGLOT_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/parkinglot/impl_type_parkinglotframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace parkinglot {

class HozonInterface_ParkingLot {
public:
    constexpr HozonInterface_ParkingLot() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_ParkingLot");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace parkinglot
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_PARKINGLOT_HOZONINTERFACE_PARKINGLOT_COMMON_H
