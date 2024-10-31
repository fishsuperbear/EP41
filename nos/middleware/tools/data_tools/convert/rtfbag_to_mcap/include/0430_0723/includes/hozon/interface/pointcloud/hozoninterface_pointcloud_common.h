/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_POINTCLOUD_HOZONINTERFACE_POINTCLOUD_COMMON_H
#define HOZON_INTERFACE_POINTCLOUD_HOZONINTERFACE_POINTCLOUD_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/sensors/impl_type_pointcloudframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace pointcloud {

class HozonInterface_PointCloud {
public:
    constexpr HozonInterface_PointCloud() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_PointCloud");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace pointcloud
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_POINTCLOUD_HOZONINTERFACE_POINTCLOUD_COMMON_H
