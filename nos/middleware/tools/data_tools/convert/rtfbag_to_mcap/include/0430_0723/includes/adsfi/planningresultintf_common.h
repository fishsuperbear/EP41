/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_PLANNINGRESULTINTF_COMMON_H
#define ADSFI_PLANNINGRESULTINTF_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "adsfi/impl_type_planningresult.h"
#include <cfloat>
#include <cmath>

namespace adsfi {

class PlanningResultIntf {
public:
    constexpr PlanningResultIntf() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/FunctionalSoftware/PlanningResultServiceInterface/PlanningResultIntf");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace adsfi

#endif // ADSFI_PLANNINGRESULTINTF_COMMON_H
