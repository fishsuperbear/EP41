/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef SOTIF_DECISIONSAFETYSERVICEINTERFACE_COMMON_H
#define SOTIF_DECISIONSAFETYSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "sotif/impl_type_decisionsafety.h"
#include <cfloat>
#include <cmath>

namespace sotif {

class DecisionSafetyServiceInterface {
public:
    constexpr DecisionSafetyServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/Sotif/DecistionSafetyServiceInterface/DecisionSafetyServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace sotif

#endif // SOTIF_DECISIONSAFETYSERVICEINTERFACE_COMMON_H
