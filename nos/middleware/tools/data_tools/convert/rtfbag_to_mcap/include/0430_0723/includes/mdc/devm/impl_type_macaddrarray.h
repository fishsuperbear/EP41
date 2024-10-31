/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_MACADDRARRAY_H
#define MDC_DEVM_IMPL_TYPE_MACADDRARRAY_H
#include "ara/core/array.h"
#include "mdc/devm/impl_type_macaddresselem.h"

namespace mdc {
namespace devm {
using MacAddrArray = ara::core::Array<mdc::devm::MacAddressElem, 16U>;
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_MACADDRARRAY_H
