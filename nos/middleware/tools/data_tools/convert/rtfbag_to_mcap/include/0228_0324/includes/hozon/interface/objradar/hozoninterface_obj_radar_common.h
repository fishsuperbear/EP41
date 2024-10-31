/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_OBJRADAR_HOZONINTERFACE_OBJ_RADAR_COMMON_H
#define HOZON_INTERFACE_OBJRADAR_HOZONINTERFACE_OBJ_RADAR_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/sensors/impl_type_radartrackarrayframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace objradar {

class HozonInterface_Obj_Radar {
public:
    constexpr HozonInterface_Obj_Radar() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Obj_Radar");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace objradar
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_OBJRADAR_HOZONINTERFACE_OBJ_RADAR_COMMON_H
