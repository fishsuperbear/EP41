/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_LANEPOINT_H
#define HOZON_LOCATION_IMPL_TYPE_LANEPOINT_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "hozon/composite/impl_type_point3d_double.h"

namespace hozon {
namespace location {
struct LanePoint {
    ::Float curvature;
    ::Float slope;
    ::Float banking;
    ::Float headingAngle;
    ::hozon::composite::Point3D_double lanePoint;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(curvature);
        fun(slope);
        fun(banking);
        fun(headingAngle);
        fun(lanePoint);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(curvature);
        fun(slope);
        fun(banking);
        fun(headingAngle);
        fun(lanePoint);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("curvature", curvature);
        fun("slope", slope);
        fun("banking", banking);
        fun("headingAngle", headingAngle);
        fun("lanePoint", lanePoint);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("curvature", curvature);
        fun("slope", slope);
        fun("banking", banking);
        fun("headingAngle", headingAngle);
        fun("lanePoint", lanePoint);
    }

    bool operator==(const ::hozon::location::LanePoint& t) const
    {
        return (fabs(static_cast<double>(curvature - t.curvature)) < DBL_EPSILON) && (fabs(static_cast<double>(slope - t.slope)) < DBL_EPSILON) && (fabs(static_cast<double>(banking - t.banking)) < DBL_EPSILON) && (fabs(static_cast<double>(headingAngle - t.headingAngle)) < DBL_EPSILON) && (lanePoint == t.lanePoint);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_LANEPOINT_H
