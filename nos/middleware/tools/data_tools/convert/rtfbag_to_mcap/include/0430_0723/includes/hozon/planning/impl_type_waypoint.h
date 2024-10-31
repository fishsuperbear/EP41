/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PLANNING_IMPL_TYPE_WAYPOINT_H
#define HOZON_PLANNING_IMPL_TYPE_WAYPOINT_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_double.h"
#include "impl_type_string.h"

namespace hozon {
namespace planning {
struct WayPoint {
    ::hozon::common::CommonHeader header;
    ::Double x;
    ::Double y;
    ::Double z;
    ::Double theta;
    ::Double curvature;
    ::Double s;
    ::Double deltaCurvature;
    ::String laneId;
    ::Double deltaX;
    ::Double deltaY;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(x);
        fun(y);
        fun(z);
        fun(theta);
        fun(curvature);
        fun(s);
        fun(deltaCurvature);
        fun(laneId);
        fun(deltaX);
        fun(deltaY);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(x);
        fun(y);
        fun(z);
        fun(theta);
        fun(curvature);
        fun(s);
        fun(deltaCurvature);
        fun(laneId);
        fun(deltaX);
        fun(deltaY);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("theta", theta);
        fun("curvature", curvature);
        fun("s", s);
        fun("deltaCurvature", deltaCurvature);
        fun("laneId", laneId);
        fun("deltaX", deltaX);
        fun("deltaY", deltaY);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("theta", theta);
        fun("curvature", curvature);
        fun("s", s);
        fun("deltaCurvature", deltaCurvature);
        fun("laneId", laneId);
        fun("deltaX", deltaX);
        fun("deltaY", deltaY);
    }

    bool operator==(const ::hozon::planning::WayPoint& t) const
    {
        return (header == t.header) && (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(theta - t.theta)) < DBL_EPSILON) && (fabs(static_cast<double>(curvature - t.curvature)) < DBL_EPSILON) && (fabs(static_cast<double>(s - t.s)) < DBL_EPSILON) && (fabs(static_cast<double>(deltaCurvature - t.deltaCurvature)) < DBL_EPSILON) && (laneId == t.laneId) && (fabs(static_cast<double>(deltaX - t.deltaX)) < DBL_EPSILON) && (fabs(static_cast<double>(deltaY - t.deltaY)) < DBL_EPSILON);
    }
};
} // namespace planning
} // namespace hozon


#endif // HOZON_PLANNING_IMPL_TYPE_WAYPOINT_H
