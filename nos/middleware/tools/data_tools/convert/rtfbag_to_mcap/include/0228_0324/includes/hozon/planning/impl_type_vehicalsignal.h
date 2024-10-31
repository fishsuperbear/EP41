/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PLANNING_IMPL_TYPE_VEHICALSIGNAL_H
#define HOZON_PLANNING_IMPL_TYPE_VEHICALSIGNAL_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace planning {
struct VehicalSignal {
    ::UInt8 turnSignal;
    ::Boolean highBeam;
    ::Boolean lowBeam;
    ::Boolean horn;
    ::Boolean emergencyLight;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(turnSignal);
        fun(highBeam);
        fun(lowBeam);
        fun(horn);
        fun(emergencyLight);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(turnSignal);
        fun(highBeam);
        fun(lowBeam);
        fun(horn);
        fun(emergencyLight);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("turnSignal", turnSignal);
        fun("highBeam", highBeam);
        fun("lowBeam", lowBeam);
        fun("horn", horn);
        fun("emergencyLight", emergencyLight);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("turnSignal", turnSignal);
        fun("highBeam", highBeam);
        fun("lowBeam", lowBeam);
        fun("horn", horn);
        fun("emergencyLight", emergencyLight);
    }

    bool operator==(const ::hozon::planning::VehicalSignal& t) const
    {
        return (turnSignal == t.turnSignal) && (highBeam == t.highBeam) && (lowBeam == t.lowBeam) && (horn == t.horn) && (emergencyLight == t.emergencyLight);
    }
};
} // namespace planning
} // namespace hozon


#endif // HOZON_PLANNING_IMPL_TYPE_VEHICALSIGNAL_H
