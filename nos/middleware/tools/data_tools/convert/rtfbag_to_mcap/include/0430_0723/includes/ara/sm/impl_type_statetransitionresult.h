/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SM_IMPL_TYPE_STATETRANSITIONRESULT_H
#define ARA_SM_IMPL_TYPE_STATETRANSITIONRESULT_H

#include "impl_type_uint8.h"
namespace ara {
namespace sm {
enum class StateTransitionResult : UInt8
{
    kSuccess = 0,
    kInvalid = 1,
    kFailed = 2,
    kTimeout = 3,
    kCommError = 4,
    kFileError = 5,
    kRejected = 6
};
} // namespace sm
} // namespace ara


#endif // ARA_SM_IMPL_TYPE_STATETRANSITIONRESULT_H
