/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_ADSFI_LANELINEARRAYINTERFACE_COMMON_H
#define MDC_ADSFI_LANELINEARRAYINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/adsfi/impl_type_lanelinearray.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace adsfi {

class LaneLineArrayInterface {
public:
    constexpr LaneLineArrayInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/FunctionalSoftware/LaneLineArrayServiceInterface/LaneLineArrayInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace adsfi
} // namespace mdc

#endif // MDC_ADSFI_LANELINEARRAYINTERFACE_COMMON_H
