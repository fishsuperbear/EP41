/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_RTRACK_RADARTRACKSERVICEINTERFACE_COMMON_H
#define ARA_RTRACK_RADARTRACKSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/rtrack/impl_type_radartrackarray.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace rtrack {

class RadarTrackServiceInterface {
public:
    constexpr RadarTrackServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/RadarTrackServiceInterface/RadarTrackServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace rtrack
} // namespace ara

#endif // ARA_RTRACK_RADARTRACKSERVICEINTERFACE_COMMON_H
