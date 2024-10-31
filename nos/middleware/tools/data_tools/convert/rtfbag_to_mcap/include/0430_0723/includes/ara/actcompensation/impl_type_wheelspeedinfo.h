/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ACTCOMPENSATION_IMPL_TYPE_WHEELSPEEDINFO_H
#define ARA_ACTCOMPENSATION_IMPL_TYPE_WHEELSPEEDINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"

namespace ara {
namespace actcompensation {
struct WheelSpeedInfo {
    ::Float wheelSpeedMps;
    ::UInt32 wheelCount;
    ::UInt8 wheelDirection;
    ::UInt8 wheelSpeedMpsValid;
    ::UInt8 wheelCountValid;
    ::UInt8 wheelDirectionValid;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(wheelSpeedMps);
        fun(wheelCount);
        fun(wheelDirection);
        fun(wheelSpeedMpsValid);
        fun(wheelCountValid);
        fun(wheelDirectionValid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(wheelSpeedMps);
        fun(wheelCount);
        fun(wheelDirection);
        fun(wheelSpeedMpsValid);
        fun(wheelCountValid);
        fun(wheelDirectionValid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("wheelSpeedMps", wheelSpeedMps);
        fun("wheelCount", wheelCount);
        fun("wheelDirection", wheelDirection);
        fun("wheelSpeedMpsValid", wheelSpeedMpsValid);
        fun("wheelCountValid", wheelCountValid);
        fun("wheelDirectionValid", wheelDirectionValid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("wheelSpeedMps", wheelSpeedMps);
        fun("wheelCount", wheelCount);
        fun("wheelDirection", wheelDirection);
        fun("wheelSpeedMpsValid", wheelSpeedMpsValid);
        fun("wheelCountValid", wheelCountValid);
        fun("wheelDirectionValid", wheelDirectionValid);
    }

    bool operator==(const ::ara::actcompensation::WheelSpeedInfo& t) const
    {
        return (fabs(static_cast<double>(wheelSpeedMps - t.wheelSpeedMps)) < DBL_EPSILON) && (wheelCount == t.wheelCount) && (wheelDirection == t.wheelDirection) && (wheelSpeedMpsValid == t.wheelSpeedMpsValid) && (wheelCountValid == t.wheelCountValid) && (wheelDirectionValid == t.wheelDirectionValid);
    }
};
} // namespace actcompensation
} // namespace ara


#endif // ARA_ACTCOMPENSATION_IMPL_TYPE_WHEELSPEEDINFO_H
