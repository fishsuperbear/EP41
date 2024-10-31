/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CONTROL_IMPL_TYPE_VEHICLESIGNAL_H
#define HOZON_CONTROL_IMPL_TYPE_VEHICLESIGNAL_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace control {
struct VehicleSignal {
    ::UInt8 turnSignal;
    ::Boolean lowBeam;
    ::Boolean horn;
    ::Boolean emergencyLight;
    ::Boolean highBeam;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(turnSignal);
        fun(lowBeam);
        fun(horn);
        fun(emergencyLight);
        fun(highBeam);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(turnSignal);
        fun(lowBeam);
        fun(horn);
        fun(emergencyLight);
        fun(highBeam);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("turnSignal", turnSignal);
        fun("lowBeam", lowBeam);
        fun("horn", horn);
        fun("emergencyLight", emergencyLight);
        fun("highBeam", highBeam);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("turnSignal", turnSignal);
        fun("lowBeam", lowBeam);
        fun("horn", horn);
        fun("emergencyLight", emergencyLight);
        fun("highBeam", highBeam);
    }

    bool operator==(const ::hozon::control::VehicleSignal& t) const
    {
        return (turnSignal == t.turnSignal) && (lowBeam == t.lowBeam) && (horn == t.horn) && (emergencyLight == t.emergencyLight) && (highBeam == t.highBeam);
    }
};
} // namespace control
} // namespace hozon


#endif // HOZON_CONTROL_IMPL_TYPE_VEHICLESIGNAL_H
