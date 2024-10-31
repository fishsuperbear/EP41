/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_TRAJPOINT_H
#define ADSFI_IMPL_TYPE_TRAJPOINT_H
#include <cfloat>
#include <cmath>
#include "adsfi/impl_type_pathpoint.h"
#include "impl_type_double.h"

namespace adsfi {
struct TrajPoint {
    ::adsfi::PathPoint pathPoint;
    ::Double speed;
    ::Double velocity;
    ::Double acceleration;
    ::Double relativeTime;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(pathPoint);
        fun(speed);
        fun(velocity);
        fun(acceleration);
        fun(relativeTime);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(pathPoint);
        fun(speed);
        fun(velocity);
        fun(acceleration);
        fun(relativeTime);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("pathPoint", pathPoint);
        fun("speed", speed);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
        fun("relativeTime", relativeTime);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("pathPoint", pathPoint);
        fun("speed", speed);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
        fun("relativeTime", relativeTime);
    }

    bool operator==(const ::adsfi::TrajPoint& t) const
    {
        return (pathPoint == t.pathPoint) && (fabs(static_cast<double>(speed - t.speed)) < DBL_EPSILON) && (fabs(static_cast<double>(velocity - t.velocity)) < DBL_EPSILON) && (fabs(static_cast<double>(acceleration - t.acceleration)) < DBL_EPSILON) && (fabs(static_cast<double>(relativeTime - t.relativeTime)) < DBL_EPSILON);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_TRAJPOINT_H
