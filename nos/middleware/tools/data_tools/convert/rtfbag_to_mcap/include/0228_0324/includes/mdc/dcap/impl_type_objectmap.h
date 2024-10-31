/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DCAP_IMPL_TYPE_OBJECTMAP_H
#define MDC_DCAP_IMPL_TYPE_OBJECTMAP_H
#include "ara/core/map.h"
#include "impl_type_string.h"
#include "impl_type_stringvector.h"

namespace mdc {
namespace dcap {
using ObjectMap = ara::core::Map<String, StringVector>;
} // namespace dcap
} // namespace mdc


#endif // MDC_DCAP_IMPL_TYPE_OBJECTMAP_H
