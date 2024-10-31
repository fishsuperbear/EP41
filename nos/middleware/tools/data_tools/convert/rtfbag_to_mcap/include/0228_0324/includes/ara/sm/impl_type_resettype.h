/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SM_IMPL_TYPE_RESETTYPE_H
#define ARA_SM_IMPL_TYPE_RESETTYPE_H

#include "impl_type_uint8.h"
namespace ara {
namespace sm {
enum class ResetType : UInt8
{
    kSoft = 0,
    kHard = 1
};
} // namespace sm
} // namespace ara


#endif // ARA_SM_IMPL_TYPE_RESETTYPE_H
