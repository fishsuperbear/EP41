/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_IMPL_TYPE_FMFAULTDETAIL_H
#define MDC_FM_IMPL_TYPE_FMFAULTDETAIL_H
#include <cfloat>
#include <cmath>
#include "mdc/fm/impl_type_fmfaultdata.h"
#include "impl_type_string.h"
#include "impl_type_uint64.h"

namespace mdc {
namespace fm {
struct FmFaultDetail {
    ::mdc::fm::FmFaultData faultData;
    ::String actionName;
    ::UInt64 timePointFirst;
    ::UInt64 timePointLast;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(faultData);
        fun(actionName);
        fun(timePointFirst);
        fun(timePointLast);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(faultData);
        fun(actionName);
        fun(timePointFirst);
        fun(timePointLast);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("faultData", faultData);
        fun("actionName", actionName);
        fun("timePointFirst", timePointFirst);
        fun("timePointLast", timePointLast);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("faultData", faultData);
        fun("actionName", actionName);
        fun("timePointFirst", timePointFirst);
        fun("timePointLast", timePointLast);
    }

    bool operator==(const ::mdc::fm::FmFaultDetail& t) const
    {
        return (faultData == t.faultData) && (actionName == t.actionName) && (timePointFirst == t.timePointFirst) && (timePointLast == t.timePointLast);
    }
};
} // namespace fm
} // namespace mdc


#endif // MDC_FM_IMPL_TYPE_FMFAULTDETAIL_H
