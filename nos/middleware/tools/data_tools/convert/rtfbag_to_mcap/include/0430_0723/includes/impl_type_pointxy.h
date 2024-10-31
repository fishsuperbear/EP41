/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_POINTXY_H
#define IMPL_TYPE_POINTXY_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"

struct Pointxy {
    ::Int32 x;
    ::Int32 y;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(x);
        fun(y);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
    }

    bool operator==(const ::Pointxy& t) const
    {
        return (x == t.x) && (y == t.y);
    }
};


#endif // IMPL_TYPE_POINTXY_H
