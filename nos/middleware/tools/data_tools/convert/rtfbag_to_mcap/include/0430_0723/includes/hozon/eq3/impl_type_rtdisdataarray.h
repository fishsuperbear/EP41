/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_RTDISDATAARRAY_H
#define HOZON_EQ3_IMPL_TYPE_RTDISDATAARRAY_H
#include "ara/core/array.h"
#include "hozon/eq3/impl_type_rtdisinfo.h"

namespace hozon {
namespace eq3 {
using RtDisDataArray = ara::core::Array<hozon::eq3::RtDisInfo, 6U>;
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_RTDISDATAARRAY_H
