/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PLANNING_IMPL_TYPE_TRAJECTORYPOINTVECTOR_H
#define HOZON_PLANNING_IMPL_TYPE_TRAJECTORYPOINTVECTOR_H
#include "ara/core/vector.h"
#include "hozon/planning/impl_type_trajectorypoint.h"

namespace hozon {
namespace planning {
using TrajectoryPointVector = ara::core::Vector<hozon::planning::TrajectoryPoint>;
} // namespace planning
} // namespace hozon


#endif // HOZON_PLANNING_IMPL_TYPE_TRAJECTORYPOINTVECTOR_H
