/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_PLANNING_HOZONINTERFACE_PLANNING_COMMON_H
#define HOZON_INTERFACE_PLANNING_HOZONINTERFACE_PLANNING_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/planning/impl_type_egotrajectoryframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace planning {

class HozonInterface_Planning {
public:
    constexpr HozonInterface_Planning() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Planning");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace planning
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_PLANNING_HOZONINTERFACE_PLANNING_COMMON_H
