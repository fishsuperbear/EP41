/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_PATHPOINT_H
#define ARA_ADSFI_IMPL_TYPE_PATHPOINT_H
#include <cfloat>
#include <cmath>
#include "impl_type_point.h"
#include "impl_type_double.h"

namespace ara {
namespace adsfi {
struct PathPoint {
    ::Point point;
    ::Double s;
    ::Double theta;
    ::Double kappa;
    ::Double dkappa;
    ::Double heading;
    ::Double ddkappa;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(point);
        fun(s);
        fun(theta);
        fun(kappa);
        fun(dkappa);
        fun(heading);
        fun(ddkappa);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(point);
        fun(s);
        fun(theta);
        fun(kappa);
        fun(dkappa);
        fun(heading);
        fun(ddkappa);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("point", point);
        fun("s", s);
        fun("theta", theta);
        fun("kappa", kappa);
        fun("dkappa", dkappa);
        fun("heading", heading);
        fun("ddkappa", ddkappa);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("point", point);
        fun("s", s);
        fun("theta", theta);
        fun("kappa", kappa);
        fun("dkappa", dkappa);
        fun("heading", heading);
        fun("ddkappa", ddkappa);
    }

    bool operator==(const ::ara::adsfi::PathPoint& t) const
    {
        return (point == t.point) && (fabs(static_cast<double>(s - t.s)) < DBL_EPSILON) && (fabs(static_cast<double>(theta - t.theta)) < DBL_EPSILON) && (fabs(static_cast<double>(kappa - t.kappa)) < DBL_EPSILON) && (fabs(static_cast<double>(dkappa - t.dkappa)) < DBL_EPSILON) && (fabs(static_cast<double>(heading - t.heading)) < DBL_EPSILON) && (fabs(static_cast<double>(ddkappa - t.ddkappa)) < DBL_EPSILON);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_PATHPOINT_H
