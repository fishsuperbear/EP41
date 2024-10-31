/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_QUATERNION_H
#define IMPL_TYPE_QUATERNION_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"

struct Quaternion {
    ::Float x;
    ::Float y;
    ::Float z;
    ::Float w;

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
        fun(z);
        fun(w);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(z);
        fun(w);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("w", w);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("w", w);
    }

    bool operator==(const ::Quaternion& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(w - t.w)) < DBL_EPSILON);
    }
};


#endif // IMPL_TYPE_QUATERNION_H
