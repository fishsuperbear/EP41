/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PLANNING_IMPL_TYPE_ADVISE_H
#define HOZON_PLANNING_IMPL_TYPE_ADVISE_H

#include "impl_type_uint8.h"
namespace hozon {
namespace planning {
enum class Advise : UInt8
{
    UNKNOWN = 0,
    DISALLOW_ENGAGE = 1,
    READY_TO_ENGAGE = 2,
    KEEP_ENGAGED = 3,
    PREPARE_DISENGAGE = 4
};
} // namespace planning
} // namespace hozon


#endif // HOZON_PLANNING_IMPL_TYPE_ADVISE_H
