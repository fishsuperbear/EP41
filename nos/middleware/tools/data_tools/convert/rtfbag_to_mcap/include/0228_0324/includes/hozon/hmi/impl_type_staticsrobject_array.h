/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_STATICSROBJECT_ARRAY_H
#define HOZON_HMI_IMPL_TYPE_STATICSROBJECT_ARRAY_H
#include "ara/core/array.h"
#include "hozon/hmi/impl_type_staticsrobject_struct.h"

namespace hozon {
namespace hmi {
using StaticSRObject_Array = ara::core::Array<hozon::hmi::StaticSRObject_Struct, 16U>;
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_STATICSROBJECT_ARRAY_H
