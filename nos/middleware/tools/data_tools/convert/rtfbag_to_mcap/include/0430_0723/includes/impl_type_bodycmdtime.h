/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_BODYCMDTIME_H
#define IMPL_TYPE_BODYCMDTIME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"

struct BodyCmdTime {
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

    bool operator==(const ::BodyCmdTime& t) const
    {
        return (sec == t.sec) && (nsec == t.nsec);
    }
};


#endif // IMPL_TYPE_BODYCMDTIME_H
