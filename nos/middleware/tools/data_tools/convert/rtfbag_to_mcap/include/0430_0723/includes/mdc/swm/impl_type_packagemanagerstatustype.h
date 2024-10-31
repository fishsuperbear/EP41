/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_PACKAGEMANAGERSTATUSTYPE_H
#define MDC_SWM_IMPL_TYPE_PACKAGEMANAGERSTATUSTYPE_H

#include "impl_type_uint8.h"
namespace mdc {
namespace swm {
enum class PackageManagerStatusType : UInt8
{
    kIdle = 0,
    kReady = 1,
    kBusy = 2,
    kActivating = 3,
    kActivated = 4,
    kRollingBack = 5,
    kRolledBack = 6,
    kCleaning_Up = 7,
    kVerifying = 8
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_PACKAGEMANAGERSTATUSTYPE_H
