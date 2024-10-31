/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI2ROUTER_IMPL_TYPE_ALGPOSELOCAL_H
#define HOZON_HMI2ROUTER_IMPL_TYPE_ALGPOSELOCAL_H
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmi2router {
struct AlgPoseLocal {
    float x;
    float y;
    float z;
    float heading;
    float s;

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
        fun(heading);
        fun(s);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(z);
        fun(heading);
        fun(s);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("heading", heading);
        fun("s", s);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("heading", heading);
        fun("s", s);
    }

    bool operator==(const ::hozon::hmi2router::AlgPoseLocal& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(heading - t.heading)) < DBL_EPSILON) && (fabs(static_cast<double>(s - t.s)) < DBL_EPSILON);
    }
};
} // namespace hmi2router
} // namespace hozon


#endif // HOZON_HMI2ROUTER_IMPL_TYPE_ALGPOSELOCAL_H
