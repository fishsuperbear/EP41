/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_WHEELSPEED_H
#define ARA_CHASSIS_IMPL_TYPE_WHEELSPEED_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_int32.h"
#include "impl_type_boolean.h"
#include "impl_type_float.h"

namespace ara {
namespace chassis {
struct WheelSpeed {
    ::UInt8 wheelDirection;
    ::Int32 wheelCount;
    ::Boolean wheelCountValid;
    ::Float wheelSpeedMps;
    ::Boolean wheelSpeedMpsValid;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(wheelDirection);
        fun(wheelCount);
        fun(wheelCountValid);
        fun(wheelSpeedMps);
        fun(wheelSpeedMpsValid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(wheelDirection);
        fun(wheelCount);
        fun(wheelCountValid);
        fun(wheelSpeedMps);
        fun(wheelSpeedMpsValid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("wheelDirection", wheelDirection);
        fun("wheelCount", wheelCount);
        fun("wheelCountValid", wheelCountValid);
        fun("wheelSpeedMps", wheelSpeedMps);
        fun("wheelSpeedMpsValid", wheelSpeedMpsValid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("wheelDirection", wheelDirection);
        fun("wheelCount", wheelCount);
        fun("wheelCountValid", wheelCountValid);
        fun("wheelSpeedMps", wheelSpeedMps);
        fun("wheelSpeedMpsValid", wheelSpeedMpsValid);
    }

    bool operator==(const ::ara::chassis::WheelSpeed& t) const
    {
        return (wheelDirection == t.wheelDirection) && (wheelCount == t.wheelCount) && (wheelCountValid == t.wheelCountValid) && (fabs(static_cast<double>(wheelSpeedMps - t.wheelSpeedMps)) < DBL_EPSILON) && (wheelSpeedMpsValid == t.wheelSpeedMpsValid);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_WHEELSPEED_H
