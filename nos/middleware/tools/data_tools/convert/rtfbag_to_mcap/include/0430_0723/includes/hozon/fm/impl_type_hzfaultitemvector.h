/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_IMPL_TYPE_HZFAULTITEMVECTOR_H
#define HOZON_FM_IMPL_TYPE_HZFAULTITEMVECTOR_H
#include "ara/core/vector.h"
#include "hozon/fm/impl_type_hzfaultevent.h"

namespace hozon {
namespace fm {
using HzFaultItemVector = ara::core::Vector<hozon::fm::HzFaultEvent>;
} // namespace fm
} // namespace hozon


#endif // HOZON_FM_IMPL_TYPE_HZFAULTITEMVECTOR_H
