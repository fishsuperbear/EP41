/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_RAWPOINTCLOUD_HOZONINTERFACE_RAWPOINTCLOUD_COMMON_H
#define HOZON_INTERFACE_RAWPOINTCLOUD_HOZONINTERFACE_RAWPOINTCLOUD_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/sensors/impl_type_rawpointcloudframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace rawpointcloud {

class HozonInterface_RawPointCloud {
public:
    constexpr HozonInterface_RawPointCloud() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_RawPointCloud");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace rawpointcloud
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_RAWPOINTCLOUD_HOZONINTERFACE_RAWPOINTCLOUD_COMMON_H
