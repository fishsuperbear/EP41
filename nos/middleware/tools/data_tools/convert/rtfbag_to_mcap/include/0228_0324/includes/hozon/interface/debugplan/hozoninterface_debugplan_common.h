/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_DEBUGPLAN_HOZONINTERFACE_DEBUGPLAN_COMMON_H
#define HOZON_INTERFACE_DEBUGPLAN_HOZONINTERFACE_DEBUGPLAN_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/debugplan/impl_type_debugplanframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace debugplan {

class HozonInterface_DebugPlan {
public:
    constexpr HozonInterface_DebugPlan() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_DebugPlan");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace debugplan
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_DEBUGPLAN_HOZONINTERFACE_DEBUGPLAN_COMMON_H
