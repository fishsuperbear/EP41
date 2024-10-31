/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_RADARF_RADARFRONTSERVICEINTERFACE_COMMON_H
#define HOZON_RADARF_RADARFRONTSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/sensors/impl_type_radartrackarrayframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace radarf {

class RadarFrontServiceInterface {
public:
    constexpr RadarFrontServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/RadarFrontServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace radarf
} // namespace hozon

#endif // HOZON_RADARF_RADARFRONTSERVICEINTERFACE_COMMON_H
