/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_PRECHECKRESULT_H
#define MDC_SWM_IMPL_TYPE_PRECHECKRESULT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "mdc/swm/impl_type_checkitemresultvector.h"

namespace mdc {
namespace swm {
struct PreCheckResult {
    ::UInt8 result;
    ::mdc::swm::CheckItemResultVector results;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
        fun(results);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
        fun(results);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("result", result);
        fun("results", results);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("result", result);
        fun("results", results);
    }

    bool operator==(const ::mdc::swm::PreCheckResult& t) const
    {
        return (result == t.result) && (results == t.results);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_PRECHECKRESULT_H
