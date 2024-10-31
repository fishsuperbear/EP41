/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ACTCOMPENSATION_IMPL_TYPE_VEHICLEMOTION_H
#define ARA_ACTCOMPENSATION_IMPL_TYPE_VEHICLEMOTION_H
#include <cfloat>
#include <cmath>
#include "ara/actcompensation/impl_type_float32withvalid.h"
#include "ara/actcompensation/impl_type_uint32withvalid.h"

namespace ara {
namespace actcompensation {
struct VehicleMotion {
    ::ara::actcompensation::Float32WithValid vx;
    ::ara::actcompensation::Float32WithValid vy;
    ::ara::actcompensation::Float32WithValid vz;
    ::ara::actcompensation::Float32WithValid ax;
    ::ara::actcompensation::Float32WithValid ay;
    ::ara::actcompensation::Float32WithValid az;
    ::ara::actcompensation::Float32WithValid yaw;
    ::ara::actcompensation::Float32WithValid pitch;
    ::ara::actcompensation::Float32WithValid roll;
    ::ara::actcompensation::Float32WithValid yawRate;
    ::ara::actcompensation::Float32WithValid pitchRate;
    ::ara::actcompensation::Float32WithValid rollRate;
    ::ara::actcompensation::Float32WithValid yawAcceleration;
    ::ara::actcompensation::Float32WithValid pitchAcceleration;
    ::ara::actcompensation::Float32WithValid rollAcceleration;
    ::ara::actcompensation::Uint32WithValid odometer;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vx);
        fun(vy);
        fun(vz);
        fun(ax);
        fun(ay);
        fun(az);
        fun(yaw);
        fun(pitch);
        fun(roll);
        fun(yawRate);
        fun(pitchRate);
        fun(rollRate);
        fun(yawAcceleration);
        fun(pitchAcceleration);
        fun(rollAcceleration);
        fun(odometer);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vx);
        fun(vy);
        fun(vz);
        fun(ax);
        fun(ay);
        fun(az);
        fun(yaw);
        fun(pitch);
        fun(roll);
        fun(yawRate);
        fun(pitchRate);
        fun(rollRate);
        fun(yawAcceleration);
        fun(pitchAcceleration);
        fun(rollAcceleration);
        fun(odometer);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vx", vx);
        fun("vy", vy);
        fun("vz", vz);
        fun("ax", ax);
        fun("ay", ay);
        fun("az", az);
        fun("yaw", yaw);
        fun("pitch", pitch);
        fun("roll", roll);
        fun("yawRate", yawRate);
        fun("pitchRate", pitchRate);
        fun("rollRate", rollRate);
        fun("yawAcceleration", yawAcceleration);
        fun("pitchAcceleration", pitchAcceleration);
        fun("rollAcceleration", rollAcceleration);
        fun("odometer", odometer);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vx", vx);
        fun("vy", vy);
        fun("vz", vz);
        fun("ax", ax);
        fun("ay", ay);
        fun("az", az);
        fun("yaw", yaw);
        fun("pitch", pitch);
        fun("roll", roll);
        fun("yawRate", yawRate);
        fun("pitchRate", pitchRate);
        fun("rollRate", rollRate);
        fun("yawAcceleration", yawAcceleration);
        fun("pitchAcceleration", pitchAcceleration);
        fun("rollAcceleration", rollAcceleration);
        fun("odometer", odometer);
    }

    bool operator==(const ::ara::actcompensation::VehicleMotion& t) const
    {
        return (vx == t.vx) && (vy == t.vy) && (vz == t.vz) && (ax == t.ax) && (ay == t.ay) && (az == t.az) && (yaw == t.yaw) && (pitch == t.pitch) && (roll == t.roll) && (yawRate == t.yawRate) && (pitchRate == t.pitchRate) && (rollRate == t.rollRate) && (yawAcceleration == t.yawAcceleration) && (pitchAcceleration == t.pitchAcceleration) && (rollAcceleration == t.rollAcceleration) && (odometer == t.odometer);
    }
};
} // namespace actcompensation
} // namespace ara


#endif // ARA_ACTCOMPENSATION_IMPL_TYPE_VEHICLEMOTION_H
