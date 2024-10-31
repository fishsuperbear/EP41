/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_HPPOBJECT_ARRAY_H
#define HOZON_HMI_IMPL_TYPE_HPPOBJECT_ARRAY_H
#include "ara/core/array.h"
#include "hozon/hmi/impl_type_hppobject_struct.h"

namespace hozon {
namespace hmi {
using HPPObject_Array = ara::core::Array<hozon::hmi::HPPObject_Struct, 10U>;
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_HPPOBJECT_ARRAY_H
