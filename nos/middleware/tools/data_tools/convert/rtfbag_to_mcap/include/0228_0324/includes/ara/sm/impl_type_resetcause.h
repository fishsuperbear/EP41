/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SM_IMPL_TYPE_RESETCAUSE_H
#define ARA_SM_IMPL_TYPE_RESETCAUSE_H

#include "impl_type_uint8.h"
namespace ara {
namespace sm {
enum class ResetCause : UInt8
{
    kNormal = 0,
    kUpdate = 1
};
} // namespace sm
} // namespace ara


#endif // ARA_SM_IMPL_TYPE_RESETCAUSE_H
