/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_MONITORACTIONTYPE_H
#define IMPL_TYPE_MONITORACTIONTYPE_H

#include "impl_type_uint8.h"
enum class MonitorActionType : UInt8
{
    kPassed = 0,
    kFailed = 1,
    kPrepassed = 2,
    kPrefailed = 3,
    kFdcThresholdReached = 4,
    kResetTestFailed = 5,
    kFreezeDebouncing = 6,
    kResetDebouncing = 7,
    kPrestore = 8,
    kClearPrestore = 9
};


#endif // IMPL_TYPE_MONITORACTIONTYPE_H
