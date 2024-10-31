/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CANFDMSG_IMPL_TYPE_CANMSG_ARRAY_H
#define HOZON_CANFDMSG_IMPL_TYPE_CANMSG_ARRAY_H
#include "ara/core/array.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace canfdmsg {
using canmsg_array = ara::core::Array<uint8_t, 64U>;
} // namespace canfdmsg
} // namespace hozon


#endif // HOZON_CANFDMSG_IMPL_TYPE_CANMSG_ARRAY_H
