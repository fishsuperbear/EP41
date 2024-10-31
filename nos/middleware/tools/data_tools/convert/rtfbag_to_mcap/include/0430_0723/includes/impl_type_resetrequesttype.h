/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_RESETREQUESTTYPE_H
#define IMPL_TYPE_RESETREQUESTTYPE_H

#include "impl_type_uint32_t.h"
enum class ResetRequestType : uint32_t
{
    kSoftReset = 0U,
    kHardReset = 1U,
    kKeyOffOnReset = 2U,
    kCustomReset = 3U
};


#endif // IMPL_TYPE_RESETREQUESTTYPE_H
