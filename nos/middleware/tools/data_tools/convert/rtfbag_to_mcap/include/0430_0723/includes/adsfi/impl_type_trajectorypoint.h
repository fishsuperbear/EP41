/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_TRAJECTORYPOINT_H
#define ADSFI_IMPL_TYPE_TRAJECTORYPOINT_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "adsfi/impl_type_waypoint.h"
#include "ara/common/impl_type_commonheader.h"

namespace adsfi {
struct TrajectoryPoint {
    ::Double speed;
    ::Double acc;
    ::Double timeRelative;
    ::Double deltaAcc;
    ::Double steerAngle;
    ::adsfi::WayPoint wayPoint;
    ::ara::common::CommonHeader header;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(speed);
        fun(acc);
        fun(timeRelative);
        fun(deltaAcc);
        fun(steerAngle);
        fun(wayPoint);
        fun(header);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(speed);
        fun(acc);
        fun(timeRelative);
        fun(deltaAcc);
        fun(steerAngle);
        fun(wayPoint);
        fun(header);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("speed", speed);
        fun("acc", acc);
        fun("timeRelative", timeRelative);
        fun("deltaAcc", deltaAcc);
        fun("steerAngle", steerAngle);
        fun("wayPoint", wayPoint);
        fun("header", header);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("speed", speed);
        fun("acc", acc);
        fun("timeRelative", timeRelative);
        fun("deltaAcc", deltaAcc);
        fun("steerAngle", steerAngle);
        fun("wayPoint", wayPoint);
        fun("header", header);
    }

    bool operator==(const ::adsfi::TrajectoryPoint& t) const
    {
        return (fabs(static_cast<double>(speed - t.speed)) < DBL_EPSILON) && (fabs(static_cast<double>(acc - t.acc)) < DBL_EPSILON) && (fabs(static_cast<double>(timeRelative - t.timeRelative)) < DBL_EPSILON) && (fabs(static_cast<double>(deltaAcc - t.deltaAcc)) < DBL_EPSILON) && (fabs(static_cast<double>(steerAngle - t.steerAngle)) < DBL_EPSILON) && (wayPoint == t.wayPoint) && (header == t.header);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_TRAJECTORYPOINT_H
