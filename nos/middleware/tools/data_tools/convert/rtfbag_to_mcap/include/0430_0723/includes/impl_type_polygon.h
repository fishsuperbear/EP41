/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_POLYGON_H
#define IMPL_TYPE_POLYGON_H
#include <cfloat>
#include <cmath>
#include "impl_type_pointarray.h"

struct Polygon {
    ::PointArray points;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(points);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(points);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("points", points);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("points", points);
    }

    bool operator==(const ::Polygon& t) const
    {
        return (points == t.points);
    }
};


#endif // IMPL_TYPE_POLYGON_H
