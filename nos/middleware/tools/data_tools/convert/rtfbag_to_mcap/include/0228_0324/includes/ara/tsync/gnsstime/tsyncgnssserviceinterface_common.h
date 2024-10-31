/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TSYNC_GNSSTIME_TSYNCGNSSSERVICEINTERFACE_COMMON_H
#define ARA_TSYNC_GNSSTIME_TSYNCGNSSSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/tsync/impl_type_gnssstruct.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace tsync {
namespace gnsstime {

class TsyncGnssServiceInterface {
public:
    constexpr TsyncGnssServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/TsyncGnssServiceInterface/TsyncGnssServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace gnsstime
} // namespace tsync
} // namespace ara

#endif // ARA_TSYNC_GNSSTIME_TSYNCGNSSSERVICEINTERFACE_COMMON_H
