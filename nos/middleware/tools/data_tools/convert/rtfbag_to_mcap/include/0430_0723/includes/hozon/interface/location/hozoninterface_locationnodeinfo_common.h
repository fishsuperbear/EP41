/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_LOCATION_HOZONINTERFACE_LOCATIONNODEINFO_COMMON_H
#define HOZON_INTERFACE_LOCATION_HOZONINTERFACE_LOCATIONNODEINFO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/location/impl_type_locationnodeinfo.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace location {

class HozonInterface_LocationNodeInfo {
public:
    constexpr HozonInterface_LocationNodeInfo() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_LocationNodeInfo");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace location
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_LOCATION_HOZONINTERFACE_LOCATIONNODEINFO_COMMON_H
