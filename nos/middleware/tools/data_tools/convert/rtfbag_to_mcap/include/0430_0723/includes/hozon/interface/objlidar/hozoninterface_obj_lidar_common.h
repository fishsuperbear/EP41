/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_OBJLIDAR_HOZONINTERFACE_OBJ_LIDAR_COMMON_H
#define HOZON_INTERFACE_OBJLIDAR_HOZONINTERFACE_OBJ_LIDAR_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/object/impl_type_objectlidarframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace objlidar {

class HozonInterface_Obj_Lidar {
public:
    constexpr HozonInterface_Obj_Lidar() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Obj_Lidar");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace objlidar
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_OBJLIDAR_HOZONINTERFACE_OBJ_LIDAR_COMMON_H
