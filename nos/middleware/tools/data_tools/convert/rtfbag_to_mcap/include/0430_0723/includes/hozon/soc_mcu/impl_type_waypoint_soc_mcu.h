/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_WAYPOINT_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_WAYPOINT_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"

namespace hozon {
namespace soc_mcu {
struct WayPoint_soc_mcu {
    ::Double x;
    ::Double y;
    ::Double z;
    ::Double theta;
    float curvature;
    float s;
    float deltaCurvature;
    ::Double deltaX;
    ::Double deltaY;

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
        fun(theta);
        fun(curvature);
        fun(s);
        fun(deltaCurvature);
        fun(deltaX);
        fun(deltaY);
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
    }

    bool operator==(const ::hozon::soc_mcu::WayPoint_soc_mcu& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(theta - t.theta)) < DBL_EPSILON) && (fabs(static_cast<double>(curvature - t.curvature)) < DBL_EPSILON) && (fabs(static_cast<double>(s - t.s)) < DBL_EPSILON) && (fabs(static_cast<double>(deltaCurvature - t.deltaCurvature)) < DBL_EPSILON) && (fabs(static_cast<double>(deltaX - t.deltaX)) < DBL_EPSILON) && (fabs(static_cast<double>(deltaY - t.deltaY)) < DBL_EPSILON);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_WAYPOINT_SOC_MCU_H
