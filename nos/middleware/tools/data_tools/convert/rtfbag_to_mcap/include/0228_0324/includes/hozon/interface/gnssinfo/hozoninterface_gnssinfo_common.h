/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_GNSSINFO_HOZONINTERFACE_GNSSINFO_COMMON_H
#define HOZON_INTERFACE_GNSSINFO_HOZONINTERFACE_GNSSINFO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/sensors/impl_type_gnssinfo.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace gnssinfo {

class HozonInterface_GnssInfo {
public:
    constexpr HozonInterface_GnssInfo() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_GnssInfo");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace gnssinfo
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_GNSSINFO_HOZONINTERFACE_GNSSINFO_COMMON_H
