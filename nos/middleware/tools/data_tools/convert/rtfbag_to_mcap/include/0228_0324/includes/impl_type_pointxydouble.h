/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_POINTXYDOUBLE_H
#define IMPL_TYPE_POINTXYDOUBLE_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"

struct PointxyDouble {
    ::Double x;
    ::Double y;

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

    bool operator==(const ::PointxyDouble& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON);
    }
};


#endif // IMPL_TYPE_POINTXYDOUBLE_H
