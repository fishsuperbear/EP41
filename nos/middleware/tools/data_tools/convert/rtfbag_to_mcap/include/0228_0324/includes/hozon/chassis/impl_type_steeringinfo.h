/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_STEERINGINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_STEERINGINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace chassis {
struct SteeringInfo {
    ::Double SteeringAngle;
    ::Boolean SteeringAngleValid;
    ::Double SteeringAngleSpeed;
    ::Boolean SteeringAngleSpeedValid;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SteeringAngle);
        fun(SteeringAngleValid);
        fun(SteeringAngleSpeed);
        fun(SteeringAngleSpeedValid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SteeringAngle);
        fun(SteeringAngleValid);
        fun(SteeringAngleSpeed);
        fun(SteeringAngleSpeedValid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("SteeringAngle", SteeringAngle);
        fun("SteeringAngleValid", SteeringAngleValid);
        fun("SteeringAngleSpeed", SteeringAngleSpeed);
        fun("SteeringAngleSpeedValid", SteeringAngleSpeedValid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("SteeringAngle", SteeringAngle);
        fun("SteeringAngleValid", SteeringAngleValid);
        fun("SteeringAngleSpeed", SteeringAngleSpeed);
        fun("SteeringAngleSpeedValid", SteeringAngleSpeedValid);
    }

    bool operator==(const ::hozon::chassis::SteeringInfo& t) const
    {
        return (fabs(static_cast<double>(SteeringAngle - t.SteeringAngle)) < DBL_EPSILON) && (SteeringAngleValid == t.SteeringAngleValid) && (fabs(static_cast<double>(SteeringAngleSpeed - t.SteeringAngleSpeed)) < DBL_EPSILON) && (SteeringAngleSpeedValid == t.SteeringAngleSpeedValid);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_STEERINGINFO_H
