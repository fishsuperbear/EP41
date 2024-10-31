/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_TRAJECTORYPOINT_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_TRAJECTORYPOINT_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"

namespace hozon {
namespace soc_mcu {
struct TrajectoryPoint_soc_mcu {
    ::Float timeRelative;
    ::Float x;
    ::Float y;
    ::Float z;
    ::Float theta;
    ::Float curvature;
    ::Float s;
    ::Float speed;
    ::Float acc;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(timeRelative);
        fun(x);
        fun(y);
        fun(z);
        fun(theta);
        fun(curvature);
        fun(s);
        fun(speed);
        fun(acc);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(timeRelative);
        fun(x);
        fun(y);
        fun(z);
        fun(theta);
        fun(curvature);
        fun(s);
        fun(speed);
        fun(acc);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("timeRelative", timeRelative);
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("theta", theta);
        fun("curvature", curvature);
        fun("s", s);
        fun("speed", speed);
        fun("acc", acc);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("timeRelative", timeRelative);
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("theta", theta);
        fun("curvature", curvature);
        fun("s", s);
        fun("speed", speed);
        fun("acc", acc);
    }

    bool operator==(const ::hozon::soc_mcu::TrajectoryPoint_soc_mcu& t) const
    {
        return (fabs(static_cast<double>(timeRelative - t.timeRelative)) < DBL_EPSILON) && (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(theta - t.theta)) < DBL_EPSILON) && (fabs(static_cast<double>(curvature - t.curvature)) < DBL_EPSILON) && (fabs(static_cast<double>(s - t.s)) < DBL_EPSILON) && (fabs(static_cast<double>(speed - t.speed)) < DBL_EPSILON) && (fabs(static_cast<double>(acc - t.acc)) < DBL_EPSILON);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_TRAJECTORYPOINT_SOC_MCU_H
