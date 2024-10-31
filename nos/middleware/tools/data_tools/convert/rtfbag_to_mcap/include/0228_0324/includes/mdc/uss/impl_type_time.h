/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_USS_IMPL_TYPE_TIME_H
#define MDC_USS_IMPL_TYPE_TIME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"

namespace mdc {
namespace uss {
struct Time {
    ::UInt32 sec;
    ::UInt32 nsec;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(sec);
        fun(nsec);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(sec);
        fun(nsec);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("sec", sec);
        fun("nsec", nsec);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("sec", sec);
        fun("nsec", nsec);
    }

    bool operator==(const ::mdc::uss::Time& t) const
    {
        return (sec == t.sec) && (nsec == t.nsec);
    }
};
} // namespace uss
} // namespace mdc


#endif // MDC_USS_IMPL_TYPE_TIME_H
