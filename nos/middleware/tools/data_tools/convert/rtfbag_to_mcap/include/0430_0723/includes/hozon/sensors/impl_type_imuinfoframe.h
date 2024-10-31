/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_IMUINFOFRAME_H
#define HOZON_SENSORS_IMPL_TYPE_IMUINFOFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/sensors/impl_type_geometrypoit.h"
#include "impl_type_uint16.h"
#include "impl_type_float.h"
#include "hozon/sensors/impl_type_imupose.h"

namespace hozon {
namespace sensors {
struct ImuInfoFrame {
    ::hozon::sensors::GeometryPoit angularVelocity;
    ::hozon::sensors::GeometryPoit acceleration;
    ::hozon::sensors::GeometryPoit imuVBAngularVelocity;
    ::hozon::sensors::GeometryPoit imuVBLinearAcceleration;
    ::UInt16 imuStatus;
    ::Float temperature;
    ::hozon::sensors::GeometryPoit gyroOffset;
    ::hozon::sensors::GeometryPoit accelOffset;
    ::hozon::sensors::GeometryPoit ins2antoffset;
    ::hozon::sensors::ImuPose imu2bodyosffet;
    ::Float imuyaw;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(angularVelocity);
        fun(acceleration);
        fun(imuVBAngularVelocity);
        fun(imuVBLinearAcceleration);
        fun(imuStatus);
        fun(temperature);
        fun(gyroOffset);
        fun(accelOffset);
        fun(ins2antoffset);
        fun(imu2bodyosffet);
        fun(imuyaw);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(angularVelocity);
        fun(acceleration);
        fun(imuVBAngularVelocity);
        fun(imuVBLinearAcceleration);
        fun(imuStatus);
        fun(temperature);
        fun(gyroOffset);
        fun(accelOffset);
        fun(ins2antoffset);
        fun(imu2bodyosffet);
        fun(imuyaw);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("angularVelocity", angularVelocity);
        fun("acceleration", acceleration);
        fun("imuVBAngularVelocity", imuVBAngularVelocity);
        fun("imuVBLinearAcceleration", imuVBLinearAcceleration);
        fun("imuStatus", imuStatus);
        fun("temperature", temperature);
        fun("gyroOffset", gyroOffset);
        fun("accelOffset", accelOffset);
        fun("ins2antoffset", ins2antoffset);
        fun("imu2bodyosffet", imu2bodyosffet);
        fun("imuyaw", imuyaw);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("angularVelocity", angularVelocity);
        fun("acceleration", acceleration);
        fun("imuVBAngularVelocity", imuVBAngularVelocity);
        fun("imuVBLinearAcceleration", imuVBLinearAcceleration);
        fun("imuStatus", imuStatus);
        fun("temperature", temperature);
        fun("gyroOffset", gyroOffset);
        fun("accelOffset", accelOffset);
        fun("ins2antoffset", ins2antoffset);
        fun("imu2bodyosffet", imu2bodyosffet);
        fun("imuyaw", imuyaw);
    }

    bool operator==(const ::hozon::sensors::ImuInfoFrame& t) const
    {
        return (angularVelocity == t.angularVelocity) && (acceleration == t.acceleration) && (imuVBAngularVelocity == t.imuVBAngularVelocity) && (imuVBLinearAcceleration == t.imuVBLinearAcceleration) && (imuStatus == t.imuStatus) && (fabs(static_cast<double>(temperature - t.temperature)) < DBL_EPSILON) && (gyroOffset == t.gyroOffset) && (accelOffset == t.accelOffset) && (ins2antoffset == t.ins2antoffset) && (imu2bodyosffet == t.imu2bodyosffet) && (fabs(static_cast<double>(imuyaw - t.imuyaw)) < DBL_EPSILON);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_IMUINFOFRAME_H
