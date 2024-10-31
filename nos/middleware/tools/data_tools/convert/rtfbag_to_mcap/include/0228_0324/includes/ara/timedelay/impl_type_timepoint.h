/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TIMEDELAY_IMPL_TYPE_TIMEPOINT_H
#define ARA_TIMEDELAY_IMPL_TYPE_TIMEPOINT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"

namespace ara {
namespace timedelay {
struct TimePoint {
    ::UInt32 sec;
    ::UInt32 usec;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(sec);
        fun(usec);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(sec);
        fun(usec);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("sec", sec);
        fun("usec", usec);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("sec", sec);
        fun("usec", usec);
    }

    bool operator==(const ::ara::timedelay::TimePoint& t) const
    {
        return (sec == t.sec) && (usec == t.usec);
    }
};
} // namespace timedelay
} // namespace ara


#endif // ARA_TIMEDELAY_IMPL_TYPE_TIMEPOINT_H
