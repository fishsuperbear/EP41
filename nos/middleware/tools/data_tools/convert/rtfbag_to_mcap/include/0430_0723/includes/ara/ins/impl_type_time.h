/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_INS_IMPL_TYPE_TIME_H
#define ARA_INS_IMPL_TYPE_TIME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"

namespace ara {
namespace ins {
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

    bool operator==(const ::ara::ins::Time& t) const
    {
        return (sec == t.sec) && (nsec == t.nsec);
    }
};
} // namespace ins
} // namespace ara


#endif // ARA_INS_IMPL_TYPE_TIME_H
