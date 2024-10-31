/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_OBJECT2DARRAYINTF_COMMON_H
#define ADSFI_OBJECT2DARRAYINTF_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "adsfi/impl_type_object2darray.h"
#include <cfloat>
#include <cmath>

namespace adsfi {

class Object2dArrayIntf {
public:
    constexpr Object2dArrayIntf() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/FunctionalSoftware/Object2dArrayServiceInterface/Object2dArrayIntf");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace adsfi

#endif // ADSFI_OBJECT2DARRAYINTF_COMMON_H
