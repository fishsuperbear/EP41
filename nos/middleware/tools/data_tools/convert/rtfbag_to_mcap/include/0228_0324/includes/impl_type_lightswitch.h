/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_LIGHTSWITCH_H
#define IMPL_TYPE_LIGHTSWITCH_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct LightSwitch {
    ::UInt8 turnLightSwitch;
    ::UInt8 hazardLightSwicth;
    ::UInt8 positionLightSwitch;
    ::UInt8 daytimeRunningLightSwitch;
    ::UInt8 highLowBeamSwitch;
    ::UInt8 fogFrontSwitch;
    ::UInt8 fogRearSwitch;
    ::UInt8 hornSwitch;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(turnLightSwitch);
        fun(hazardLightSwicth);
        fun(positionLightSwitch);
        fun(daytimeRunningLightSwitch);
        fun(highLowBeamSwitch);
        fun(fogFrontSwitch);
        fun(fogRearSwitch);
        fun(hornSwitch);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(turnLightSwitch);
        fun(hazardLightSwicth);
        fun(positionLightSwitch);
        fun(daytimeRunningLightSwitch);
        fun(highLowBeamSwitch);
        fun(fogFrontSwitch);
        fun(fogRearSwitch);
        fun(hornSwitch);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("turnLightSwitch", turnLightSwitch);
        fun("hazardLightSwicth", hazardLightSwicth);
        fun("positionLightSwitch", positionLightSwitch);
        fun("daytimeRunningLightSwitch", daytimeRunningLightSwitch);
        fun("highLowBeamSwitch", highLowBeamSwitch);
        fun("fogFrontSwitch", fogFrontSwitch);
        fun("fogRearSwitch", fogRearSwitch);
        fun("hornSwitch", hornSwitch);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("turnLightSwitch", turnLightSwitch);
        fun("hazardLightSwicth", hazardLightSwicth);
        fun("positionLightSwitch", positionLightSwitch);
        fun("daytimeRunningLightSwitch", daytimeRunningLightSwitch);
        fun("highLowBeamSwitch", highLowBeamSwitch);
        fun("fogFrontSwitch", fogFrontSwitch);
        fun("fogRearSwitch", fogRearSwitch);
        fun("hornSwitch", hornSwitch);
    }

    bool operator==(const ::LightSwitch& t) const
    {
        return (turnLightSwitch == t.turnLightSwitch) && (hazardLightSwicth == t.hazardLightSwicth) && (positionLightSwitch == t.positionLightSwitch) && (daytimeRunningLightSwitch == t.daytimeRunningLightSwitch) && (highLowBeamSwitch == t.highLowBeamSwitch) && (fogFrontSwitch == t.fogFrontSwitch) && (fogRearSwitch == t.fogRearSwitch) && (hornSwitch == t.hornSwitch);
    }
};


#endif // IMPL_TYPE_LIGHTSWITCH_H
