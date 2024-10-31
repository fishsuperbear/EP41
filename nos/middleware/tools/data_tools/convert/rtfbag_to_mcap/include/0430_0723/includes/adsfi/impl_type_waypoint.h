/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_WAYPOINT_H
#define ADSFI_IMPL_TYPE_WAYPOINT_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_string.h"

namespace adsfi {
struct WayPoint {
    ::Double x;
    ::Double y;
    ::Double z;
    ::Double theta;
    ::Double curvature;
    ::Double s;
    ::Double deltaCurvature;
    ::Double deltaX;
    ::Double deltaY;
    ::ara::common::CommonHeader header;
    ::String laneId;

    static bool IsPlane()
    {
        return false;
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
        fun(s);
        fun(deltaCurvature);
        fun(deltaX);
        fun(deltaY);
        fun(header);
        fun(laneId);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(z);
        fun(theta);
        fun(curvature);
        fun(s);
        fun(deltaCurvature);
        fun(deltaX);
        fun(deltaY);
        fun(header);
        fun(laneId);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("theta", theta);
        fun("curvature", curvature);
        fun("s", s);
        fun("deltaCurvature", deltaCurvature);
        fun("deltaX", deltaX);
        fun("deltaY", deltaY);
        fun("header", header);
        fun("laneId", laneId);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("theta", theta);
        fun("curvature", curvature);
        fun("s", s);
        fun("deltaCurvature", deltaCurvature);
        fun("deltaX", deltaX);
        fun("deltaY", deltaY);
        fun("header", header);
        fun("laneId", laneId);
    }

    bool operator==(const ::adsfi::WayPoint& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(theta - t.theta)) < DBL_EPSILON) && (fabs(static_cast<double>(curvature - t.curvature)) < DBL_EPSILON) && (fabs(static_cast<double>(s - t.s)) < DBL_EPSILON) && (fabs(static_cast<double>(deltaCurvature - t.deltaCurvature)) < DBL_EPSILON) && (fabs(static_cast<double>(deltaX - t.deltaX)) < DBL_EPSILON) && (fabs(static_cast<double>(deltaY - t.deltaY)) < DBL_EPSILON) && (header == t.header) && (laneId == t.laneId);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_WAYPOINT_H
