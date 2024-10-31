/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_ALGTSRMSG_H
#define IMPL_TYPE_ALGTSRMSG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct AlgTsrMsg {
    ::UInt8 spd_limit;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(spd_limit);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(spd_limit);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("spd_limit", spd_limit);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("spd_limit", spd_limit);
    }

    bool operator==(const ::AlgTsrMsg& t) const
    {
        return (spd_limit == t.spd_limit);
    }
};


#endif // IMPL_TYPE_ALGTSRMSG_H
