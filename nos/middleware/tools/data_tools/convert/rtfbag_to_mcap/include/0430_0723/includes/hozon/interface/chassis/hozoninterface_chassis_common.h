/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_CHASSIS_HOZONINTERFACE_CHASSIS_COMMON_H
#define HOZON_INTERFACE_CHASSIS_HOZONINTERFACE_CHASSIS_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/chassis/impl_type_chassisinfoframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace chassis {

class HozonInterface_Chassis {
public:
    constexpr HozonInterface_Chassis() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Chassis");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace chassis
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_CHASSIS_HOZONINTERFACE_CHASSIS_COMMON_H
