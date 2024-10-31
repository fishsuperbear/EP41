/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_TRAJECTORYPOINT_H
#define ARA_ADSFI_IMPL_TYPE_TRAJECTORYPOINT_H
#include <cfloat>
#include <cmath>
#include "ara/adsfi/impl_type_pathpoint.h"
#include "impl_type_double.h"

namespace ara {
namespace adsfi {
struct TrajectoryPoint {
    ::ara::adsfi::PathPoint pathPoint;
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

    bool operator==(const ::ara::adsfi::TrajectoryPoint& t) const
    {
        return (pathPoint == t.pathPoint) && (fabs(static_cast<double>(speed - t.speed)) < DBL_EPSILON) && (fabs(static_cast<double>(velocity - t.velocity)) < DBL_EPSILON) && (fabs(static_cast<double>(acceleration - t.acceleration)) < DBL_EPSILON) && (fabs(static_cast<double>(relativeTime - t.relativeTime)) < DBL_EPSILON);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_TRAJECTORYPOINT_H
