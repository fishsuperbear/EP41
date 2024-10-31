/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_CANIDINFOLIST_H
#define MDC_DEVM_IMPL_TYPE_CANIDINFOLIST_H
#include "ara/core/array.h"
#include "mdc/devm/impl_type_canidinfo.h"

namespace mdc {
namespace devm {
using CanIdInfoList = ara::core::Array<mdc::devm::CanIdInfo, 60U>;
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_CANIDINFOLIST_H
