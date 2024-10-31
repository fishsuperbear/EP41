/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_LANEARRAYINTF_COMMON_H
#define ADSFI_LANEARRAYINTF_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "adsfi/impl_type_lanearray.h"
#include <cfloat>
#include <cmath>

namespace adsfi {

class LaneArrayIntf {
public:
    constexpr LaneArrayIntf() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/FunctionalSoftware/LaneLineArrayServiceInterface/LaneArrayIntf");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace adsfi

#endif // ADSFI_LANEARRAYINTF_COMMON_H
