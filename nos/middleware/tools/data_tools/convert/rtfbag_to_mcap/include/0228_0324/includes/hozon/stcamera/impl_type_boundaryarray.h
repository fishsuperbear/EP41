/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STCAMERA_IMPL_TYPE_BOUNDARYARRAY_H
#define HOZON_STCAMERA_IMPL_TYPE_BOUNDARYARRAY_H
#include "ara/core/array.h"
#include "hozon/stcamera/impl_type_boundary.h"

namespace hozon {
namespace stcamera {
using BoundaryArray = ara::core::Array<hozon::stcamera::Boundary, 10U>;
} // namespace stcamera
} // namespace hozon


#endif // HOZON_STCAMERA_IMPL_TYPE_BOUNDARYARRAY_H
