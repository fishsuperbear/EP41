/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_VEHICLEMOTION_H
#define ARA_CHASSIS_IMPL_TYPE_VEHICLEMOTION_H
#include <cfloat>
#include <cmath>
#include "ara/chassis/impl_type_float32withvalid.h"

namespace ara {
namespace chassis {
struct VehicleMotion {
    ::ara::chassis::Float32WithValid vx;
    ::ara::chassis::Float32WithValid vy;
    ::ara::chassis::Float32WithValid vz;
    ::ara::chassis::Float32WithValid ax;
    ::ara::chassis::Float32WithValid ay;
    ::ara::chassis::Float32WithValid az;
    ::ara::chassis::Float32WithValid yaw;
    ::ara::chassis::Float32WithValid pitch;
    ::ara::chassis::Float32WithValid roll;
    ::ara::chassis::Float32WithValid yawRate;
    ::ara::chassis::Float32WithValid pitchRate;
    ::ara::chassis::Float32WithValid rollRate;
    ::ara::chassis::Float32WithValid yawAcceleration;
    ::ara::chassis::Float32WithValid pitchAcceleration;
    ::ara::chassis::Float32WithValid rollAcceleration;

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
    }

    bool operator==(const ::ara::chassis::VehicleMotion& t) const
    {
        return (vx == t.vx) && (vy == t.vy) && (vz == t.vz) && (ax == t.ax) && (ay == t.ay) && (az == t.az) && (yaw == t.yaw) && (pitch == t.pitch) && (roll == t.roll) && (yawRate == t.yawRate) && (pitchRate == t.pitchRate) && (rollRate == t.rollRate) && (yawAcceleration == t.yawAcceleration) && (pitchAcceleration == t.pitchAcceleration) && (rollAcceleration == t.rollAcceleration);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_VEHICLEMOTION_H
