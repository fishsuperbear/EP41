/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_RECT_H
#define IMPL_TYPE_RECT_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"

struct Rect {
    ::Float x;
    ::Float y;
    ::Float width;
    ::Float height;

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
        fun(width);
        fun(height);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(width);
        fun(height);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("width", width);
        fun("height", height);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("width", width);
        fun("height", height);
    }

    bool operator==(const ::Rect& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(width - t.width)) < DBL_EPSILON) && (fabs(static_cast<double>(height - t.height)) < DBL_EPSILON);
    }
};


#endif // IMPL_TYPE_RECT_H
