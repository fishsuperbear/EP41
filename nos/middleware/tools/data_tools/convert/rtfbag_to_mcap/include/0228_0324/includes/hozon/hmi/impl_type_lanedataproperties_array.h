/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_LANEDATAPROPERTIES_ARRAY_H
#define HOZON_HMI_IMPL_TYPE_LANEDATAPROPERTIES_ARRAY_H
#include "ara/core/array.h"
#include "hozon/hmi/impl_type_lanedata_struct.h"

namespace hozon {
namespace hmi {
using LaneDataProperties_Array = ara::core::Array<hozon::hmi::LaneData_Struct, 8U>;
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_LANEDATAPROPERTIES_ARRAY_H
