/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_INSINFOFRAME_H
#define HOZON_SENSORS_IMPL_TYPE_INSINFOFRAME_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "hozon/sensors/impl_type_geometrypoit.h"
#include "impl_type_float.h"
#include "impl_type_uint16.h"
#include "impl_type_uint16_t.h"

namespace hozon {
namespace sensors {
struct InsInfoFrame {
    ::Double latitude;
    ::Double longitude;
    ::Double altitude;
    ::hozon::sensors::GeometryPoit attitude;
    ::hozon::sensors::GeometryPoit linearVelocity;
    ::hozon::sensors::GeometryPoit augularVelocity;
    ::hozon::sensors::GeometryPoit linearAcceleration;
    ::Float heading;
    ::hozon::sensors::GeometryPoit mountingError;
    ::hozon::sensors::GeometryPoit sdPosition;
    ::hozon::sensors::GeometryPoit sdAttitude;
    ::hozon::sensors::GeometryPoit sdVelocity;
    ::UInt16 sysStatus;
    ::UInt16 gpsStatus;
    ::uint16_t sensorUsed;
    float wheelVelocity;
    float odoSF;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(latitude);
        fun(longitude);
        fun(altitude);
        fun(attitude);
        fun(linearVelocity);
        fun(augularVelocity);
        fun(linearAcceleration);
        fun(heading);
        fun(mountingError);
        fun(sdPosition);
        fun(sdAttitude);
        fun(sdVelocity);
        fun(sysStatus);
        fun(gpsStatus);
        fun(sensorUsed);
        fun(wheelVelocity);
        fun(odoSF);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(latitude);
        fun(longitude);
        fun(altitude);
        fun(attitude);
        fun(linearVelocity);
        fun(augularVelocity);
        fun(linearAcceleration);
        fun(heading);
        fun(mountingError);
        fun(sdPosition);
        fun(sdAttitude);
        fun(sdVelocity);
        fun(sysStatus);
        fun(gpsStatus);
        fun(sensorUsed);
        fun(wheelVelocity);
        fun(odoSF);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("latitude", latitude);
        fun("longitude", longitude);
        fun("altitude", altitude);
        fun("attitude", attitude);
        fun("linearVelocity", linearVelocity);
        fun("augularVelocity", augularVelocity);
        fun("linearAcceleration", linearAcceleration);
        fun("heading", heading);
        fun("mountingError", mountingError);
        fun("sdPosition", sdPosition);
        fun("sdAttitude", sdAttitude);
        fun("sdVelocity", sdVelocity);
        fun("sysStatus", sysStatus);
        fun("gpsStatus", gpsStatus);
        fun("sensorUsed", sensorUsed);
        fun("wheelVelocity", wheelVelocity);
        fun("odoSF", odoSF);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("latitude", latitude);
        fun("longitude", longitude);
        fun("altitude", altitude);
        fun("attitude", attitude);
        fun("linearVelocity", linearVelocity);
        fun("augularVelocity", augularVelocity);
        fun("linearAcceleration", linearAcceleration);
        fun("heading", heading);
        fun("mountingError", mountingError);
        fun("sdPosition", sdPosition);
        fun("sdAttitude", sdAttitude);
        fun("sdVelocity", sdVelocity);
        fun("sysStatus", sysStatus);
        fun("gpsStatus", gpsStatus);
        fun("sensorUsed", sensorUsed);
        fun("wheelVelocity", wheelVelocity);
        fun("odoSF", odoSF);
    }

    bool operator==(const ::hozon::sensors::InsInfoFrame& t) const
    {
        return (fabs(static_cast<double>(latitude - t.latitude)) < DBL_EPSILON) && (fabs(static_cast<double>(longitude - t.longitude)) < DBL_EPSILON) && (fabs(static_cast<double>(altitude - t.altitude)) < DBL_EPSILON) && (attitude == t.attitude) && (linearVelocity == t.linearVelocity) && (augularVelocity == t.augularVelocity) && (linearAcceleration == t.linearAcceleration) && (fabs(static_cast<double>(heading - t.heading)) < DBL_EPSILON) && (mountingError == t.mountingError) && (sdPosition == t.sdPosition) && (sdAttitude == t.sdAttitude) && (sdVelocity == t.sdVelocity) && (sysStatus == t.sysStatus) && (gpsStatus == t.gpsStatus) && (sensorUsed == t.sensorUsed) && (fabs(static_cast<double>(wheelVelocity - t.wheelVelocity)) < DBL_EPSILON) && (fabs(static_cast<double>(odoSF - t.odoSF)) < DBL_EPSILON);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_INSINFOFRAME_H
