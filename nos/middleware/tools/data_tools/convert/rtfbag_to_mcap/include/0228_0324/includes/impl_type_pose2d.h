/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_POSE2D_H
#define IMPL_TYPE_POSE2D_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"

struct Pose2D {
    ::Double x;
    ::Double y;
    ::Double theta;

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
        fun(theta);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(theta);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("theta", theta);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("theta", theta);
    }

    bool operator==(const ::Pose2D& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(theta - t.theta)) < DBL_EPSILON);
    }
};


#endif // IMPL_TYPE_POSE2D_H
