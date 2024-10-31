/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PARKINGLOT_IMPL_TYPE_PATHPOINT_H
#define HOZON_PARKINGLOT_IMPL_TYPE_PATHPOINT_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace parkinglot {
struct PathPoint {
    ::Float x;
    ::Float y;
    ::Float z;
    ::Float yaw;
    ::Float accumulate_s;
    ::UInt8 gear;

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
        fun(yaw);
        fun(accumulate_s);
        fun(gear);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(z);
        fun(yaw);
        fun(accumulate_s);
        fun(gear);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("yaw", yaw);
        fun("accumulate_s", accumulate_s);
        fun("gear", gear);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("yaw", yaw);
        fun("accumulate_s", accumulate_s);
        fun("gear", gear);
    }

    bool operator==(const ::hozon::parkinglot::PathPoint& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(yaw - t.yaw)) < DBL_EPSILON) && (fabs(static_cast<double>(accumulate_s - t.accumulate_s)) < DBL_EPSILON) && (gear == t.gear);
    }
};
} // namespace parkinglot
} // namespace hozon


#endif // HOZON_PARKINGLOT_IMPL_TYPE_PATHPOINT_H
