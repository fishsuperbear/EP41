/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_GNSS_GNSSINFOSERVICEINTERFACE_COMMON_H
#define ARA_GNSS_GNSSINFOSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/gnss/impl_type_gnssinfo.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace gnss {

class GnssInfoServiceInterface {
public:
    constexpr GnssInfoServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/GnssInfoServiceInterface/GnssInfoServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace gnss
} // namespace ara

#endif // ARA_GNSS_GNSSINFOSERVICEINTERFACE_COMMON_H
