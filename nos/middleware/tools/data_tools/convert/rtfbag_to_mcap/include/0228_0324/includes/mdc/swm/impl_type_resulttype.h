/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_RESULTTYPE_H
#define MDC_SWM_IMPL_TYPE_RESULTTYPE_H

#include "impl_type_uint8.h"
namespace mdc {
namespace swm {
enum class ResultType : UInt8
{
    kSuccessfull = 0,
    kFailed = 1
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_RESULTTYPE_H
