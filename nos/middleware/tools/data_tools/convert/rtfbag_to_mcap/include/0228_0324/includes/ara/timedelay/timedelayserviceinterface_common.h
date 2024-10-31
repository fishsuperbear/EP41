/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TIMEDELAY_TIMEDELAYSERVICEINTERFACE_COMMON_H
#define ARA_TIMEDELAY_TIMEDELAYSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/timedelay/impl_type_timedelaydatatype.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace timedelay {

class TimeDelayServiceInterface {
public:
    constexpr TimeDelayServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/TimeDelayServiceInterface/TimeDelayServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace timedelay
} // namespace ara

#endif // ARA_TIMEDELAY_TIMEDELAYSERVICEINTERFACE_COMMON_H
