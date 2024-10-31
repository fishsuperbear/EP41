/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STCAMERA_IMPL_TYPE_CAMOBJECTARRAY_H
#define HOZON_STCAMERA_IMPL_TYPE_CAMOBJECTARRAY_H
#include "ara/core/array.h"
#include "hozon/stcamera/impl_type_camobject.h"

namespace hozon {
namespace stcamera {
using CamObjectArray = ara::core::Array<hozon::stcamera::CamObject, 20U>;
} // namespace stcamera
} // namespace hozon


#endif // HOZON_STCAMERA_IMPL_TYPE_CAMOBJECTARRAY_H
