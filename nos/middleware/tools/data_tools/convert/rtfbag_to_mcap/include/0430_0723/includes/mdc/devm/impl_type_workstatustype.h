/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_WORKSTATUSTYPE_H
#define MDC_DEVM_IMPL_TYPE_WORKSTATUSTYPE_H

#include "impl_type_uint8.h"
namespace mdc {
namespace devm {
enum class WorkStatusType : UInt8
{
    kOk = 0,
    kError = 1,
    kUnkonw = 2
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_WORKSTATUSTYPE_H
