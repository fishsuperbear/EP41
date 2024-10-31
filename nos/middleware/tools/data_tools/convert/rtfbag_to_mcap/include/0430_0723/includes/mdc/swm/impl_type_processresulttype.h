/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_PROCESSRESULTTYPE_H
#define MDC_SWM_IMPL_TYPE_PROCESSRESULTTYPE_H
#include <cfloat>
#include <cmath>
#include "mdc/swm/impl_type_processswpackagereturntype.h"
#include "mdc/swm/impl_type_transferidtype.h"

namespace mdc {
namespace swm {
struct ProcessResultType {
    ::mdc::swm::ProcessSwPackageReturnType result;
    ::mdc::swm::TransferIdType transferId;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
        fun(transferId);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
        fun(transferId);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("result", result);
        fun("transferId", transferId);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("result", result);
        fun("transferId", transferId);
    }

    bool operator==(const ::mdc::swm::ProcessResultType& t) const
    {
        return (result == t.result) && (transferId == t.transferId);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_PROCESSRESULTTYPE_H
