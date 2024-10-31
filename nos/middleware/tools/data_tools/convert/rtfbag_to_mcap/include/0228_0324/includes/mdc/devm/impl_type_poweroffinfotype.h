/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_POWEROFFINFOTYPE_H
#define MDC_DEVM_IMPL_TYPE_POWEROFFINFOTYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint64.h"
#include "impl_type_uint8.h"

namespace mdc {
namespace devm {
struct PowerOffInfoType {
    ::UInt64 utcTime;
    ::UInt8 reason;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(utcTime);
        fun(reason);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(utcTime);
        fun(reason);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("utcTime", utcTime);
        fun("reason", reason);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("utcTime", utcTime);
        fun("reason", reason);
    }

    bool operator==(const ::mdc::devm::PowerOffInfoType& t) const
    {
        return (utcTime == t.utcTime) && (reason == t.reason);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_POWEROFFINFOTYPE_H
