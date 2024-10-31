/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_PEDESTRIANDATAARRAY_H
#define HOZON_EQ3_IMPL_TYPE_PEDESTRIANDATAARRAY_H
#include "ara/core/array.h"
#include "hozon/eq3/impl_type_pedestrianinfo.h"

namespace hozon {
namespace eq3 {
using PedestrianDataArray = ara::core::Array<hozon::eq3::PedestrianInfo, 10U>;
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_PEDESTRIANDATAARRAY_H
