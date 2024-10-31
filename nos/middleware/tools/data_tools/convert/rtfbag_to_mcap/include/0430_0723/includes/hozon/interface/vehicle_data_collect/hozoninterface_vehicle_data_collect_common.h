/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_VEHICLE_DATA_COLLECT_HOZONINTERFACE_VEHICLE_DATA_COLLECT_COMMON_H
#define HOZON_INTERFACE_VEHICLE_DATA_COLLECT_HOZONINTERFACE_VEHICLE_DATA_COLLECT_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/vehicle_data_collect/impl_type_triggersignalframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace vehicle_data_collect {

class HozonInterface_vehicle_data_collect {
public:
    constexpr HozonInterface_vehicle_data_collect() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HozonInterface_vehicle_data_collect");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace vehicle_data_collect
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_VEHICLE_DATA_COLLECT_HOZONINTERFACE_VEHICLE_DATA_COLLECT_COMMON_H
