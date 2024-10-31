/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_LIDAR_LIDARSERVICEINTERFACE_COMMON_H
#define ARA_LIDAR_LIDARSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/lidar/impl_type_lidarpointcloud.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace lidar {

class LidarServiceInterface {
public:
    constexpr LidarServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/LidarServiceInterface/LidarServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace lidar
} // namespace ara

#endif // ARA_LIDAR_LIDARSERVICEINTERFACE_COMMON_H
