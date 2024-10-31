/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_LIGHTSTATUS_H
#define IMPL_TYPE_LIGHTSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct LightStatus {
    ::UInt8 turnLightStatus;
    ::UInt8 hazardLightStatus;
    ::UInt8 brakeLightStatus;
    ::UInt8 positionLightStatus;
    ::UInt8 reverseLightStatus;
    ::UInt8 daytimeRunningLightStatus;
    ::UInt8 ambientLightStatus;
    ::UInt8 highLowBeamStatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(turnLightStatus);
        fun(hazardLightStatus);
        fun(brakeLightStatus);
        fun(positionLightStatus);
        fun(reverseLightStatus);
        fun(daytimeRunningLightStatus);
        fun(ambientLightStatus);
        fun(highLowBeamStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(turnLightStatus);
        fun(hazardLightStatus);
        fun(brakeLightStatus);
        fun(positionLightStatus);
        fun(reverseLightStatus);
        fun(daytimeRunningLightStatus);
        fun(ambientLightStatus);
        fun(highLowBeamStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("turnLightStatus", turnLightStatus);
        fun("hazardLightStatus", hazardLightStatus);
        fun("brakeLightStatus", brakeLightStatus);
        fun("positionLightStatus", positionLightStatus);
        fun("reverseLightStatus", reverseLightStatus);
        fun("daytimeRunningLightStatus", daytimeRunningLightStatus);
        fun("ambientLightStatus", ambientLightStatus);
        fun("highLowBeamStatus", highLowBeamStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("turnLightStatus", turnLightStatus);
        fun("hazardLightStatus", hazardLightStatus);
        fun("brakeLightStatus", brakeLightStatus);
        fun("positionLightStatus", positionLightStatus);
        fun("reverseLightStatus", reverseLightStatus);
        fun("daytimeRunningLightStatus", daytimeRunningLightStatus);
        fun("ambientLightStatus", ambientLightStatus);
        fun("highLowBeamStatus", highLowBeamStatus);
    }

    bool operator==(const ::LightStatus& t) const
    {
        return (turnLightStatus == t.turnLightStatus) && (hazardLightStatus == t.hazardLightStatus) && (brakeLightStatus == t.brakeLightStatus) && (positionLightStatus == t.positionLightStatus) && (reverseLightStatus == t.reverseLightStatus) && (daytimeRunningLightStatus == t.daytimeRunningLightStatus) && (ambientLightStatus == t.ambientLightStatus) && (highLowBeamStatus == t.highLowBeamStatus);
    }
};


#endif // IMPL_TYPE_LIGHTSTATUS_H
