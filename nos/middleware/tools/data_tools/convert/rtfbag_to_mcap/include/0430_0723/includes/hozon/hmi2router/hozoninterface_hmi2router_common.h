/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI2ROUTER_HOZONINTERFACE_HMI2ROUTER_COMMON_H
#define HOZON_HMI2ROUTER_HOZONINTERFACE_HMI2ROUTER_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/hmi2router/impl_type_algnnsrouteinfo.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmi2router {

class HozonInterface_Hmi2Router {
public:
    constexpr HozonInterface_Hmi2Router() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HozonInterface_Hmi2Router");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace hmi2router
} // namespace hozon

#endif // HOZON_HMI2ROUTER_HOZONINTERFACE_HMI2ROUTER_COMMON_H
