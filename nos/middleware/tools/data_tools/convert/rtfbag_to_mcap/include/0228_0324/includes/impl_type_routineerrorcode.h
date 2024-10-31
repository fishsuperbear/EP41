/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_ROUTINEERRORCODE_H
#define IMPL_TYPE_ROUTINEERRORCODE_H

#include "impl_type_uint8.h"
enum class RoutineErrorCode : UInt8
{
    kERoutineNoError = 0,
    kERoutineUndefined = 1,
    kERoutineNotFound = 2,
    kERoutineRequestInvalid = 3,
    kERoutineInternalError = 4,
    kERoutineHandlerNotAvailable = 5,
    kERoutineSequenceError = 6,
    kERoutineRequestOutofRange = 7
};


#endif // IMPL_TYPE_ROUTINEERRORCODE_H
