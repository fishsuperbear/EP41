/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_SWCLUSTERSTATETYPE_H
#define MDC_SWM_IMPL_TYPE_SWCLUSTERSTATETYPE_H

#include "impl_type_uint8.h"
namespace mdc {
namespace swm {
enum class SwClusterStateType : UInt8
{
    kPresent = 0,
    kAdded = 1,
    kUpdated = 2,
    kRemoved = 3
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_SWCLUSTERSTATETYPE_H
