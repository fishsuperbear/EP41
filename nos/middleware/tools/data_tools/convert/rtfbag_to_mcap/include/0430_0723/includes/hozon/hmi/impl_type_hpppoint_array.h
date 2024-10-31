/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_HPPPOINT_ARRAY_H
#define HOZON_HMI_IMPL_TYPE_HPPPOINT_ARRAY_H
#include "ara/core/array.h"
#include "hozon/hmi/impl_type_hppdim5f_struct.h"

namespace hozon {
namespace hmi {
using HPPPoint_Array = ara::core::Array<hozon::hmi::HPPDim5F_Struct, 5000U>;
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_HPPPOINT_ARRAY_H
