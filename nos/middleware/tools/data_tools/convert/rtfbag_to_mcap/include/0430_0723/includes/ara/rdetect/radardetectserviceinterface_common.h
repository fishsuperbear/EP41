/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_RDETECT_RADARDETECTSERVICEINTERFACE_COMMON_H
#define ARA_RDETECT_RADARDETECTSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/rdetect/impl_type_radardetectarray.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace rdetect {

class RadarDetectServiceInterface {
public:
    constexpr RadarDetectServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/RadarDetectServiceInterface/RadarDetectServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace rdetect
} // namespace ara

#endif // ARA_RDETECT_RADARDETECTSERVICEINTERFACE_COMMON_H
