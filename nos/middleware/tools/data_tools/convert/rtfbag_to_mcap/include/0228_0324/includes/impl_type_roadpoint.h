/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_ROADPOINT_H
#define IMPL_TYPE_ROADPOINT_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "impl_type_float.h"

struct RoadPoint {
    ::Double x;
    ::Double y;
    ::Double z;
    ::Double theta;
    ::Float curvature;

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
        fun(theta);
        fun(curvature);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(z);
        fun(theta);
        fun(curvature);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("theta", theta);
        fun("curvature", curvature);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("theta", theta);
        fun("curvature", curvature);
    }

    bool operator==(const ::RoadPoint& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(theta - t.theta)) < DBL_EPSILON) && (fabs(static_cast<double>(curvature - t.curvature)) < DBL_EPSILON);
    }
};


#endif // IMPL_TYPE_ROADPOINT_H
