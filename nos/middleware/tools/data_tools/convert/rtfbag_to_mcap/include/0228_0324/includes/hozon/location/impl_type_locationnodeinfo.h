/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_LOCATIONNODEINFO_H
#define HOZON_LOCATION_IMPL_TYPE_LOCATIONNODEINFO_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32_t.h"
#include "impl_type_double.h"
#include "hozon/composite/impl_type_point3d_double.h"
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_quaternion.h"
#include "hozon/object/impl_type_point3f.h"
#include "hozon/composite/impl_type_floatarray36.h"
#include "impl_type_float.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace location {
struct LocationNodeInfo {
    ::hozon::common::CommonHeader header;
    ::uint32_t gpsWeek;
    ::Double gpsSec;
    ::hozon::composite::Point3D_double posSmooth;
    ::UInt8 type;
    ::hozon::composite::Point3D_double attitude;
    ::hozon::composite::Quaternion quaternion;
    ::hozon::composite::Point3D_double linearVelocity;
    ::hozon::object::Point3f gyroBias;
    ::hozon::object::Point3f accelBias;
    ::hozon::composite::Point3D_double sdPosition;
    ::hozon::composite::Point3D_double sdAttitude;
    ::hozon::composite::Point3D_double sdVelocity;
    ::hozon::composite::FloatArray36 covariance;
    ::uint32_t sysStatus;
    ::uint32_t gpsStatus;
    ::Float heading;
    ::uint32_t warn_info;
    ::uint32_t errorCode;
    ::uint32_t innerCode;
    ::hozon::composite::Point3D_double posGCJ02;
    ::hozon::composite::Point3D_double angularVelocity;
    ::hozon::composite::Point3D_double linearAcceleration;
    ::hozon::composite::Point3D_double mountingError;
    ::uint32_t sensorUsed;
    ::Float wheelVelocity;
    ::Float odoSF;
    ::Boolean validEstimate;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(gpsWeek);
        fun(gpsSec);
        fun(posSmooth);
        fun(type);
        fun(attitude);
        fun(quaternion);
        fun(linearVelocity);
        fun(gyroBias);
        fun(accelBias);
        fun(sdPosition);
        fun(sdAttitude);
        fun(sdVelocity);
        fun(covariance);
        fun(sysStatus);
        fun(gpsStatus);
        fun(heading);
        fun(warn_info);
        fun(errorCode);
        fun(innerCode);
        fun(posGCJ02);
        fun(angularVelocity);
        fun(linearAcceleration);
        fun(mountingError);
        fun(sensorUsed);
        fun(wheelVelocity);
        fun(odoSF);
        fun(validEstimate);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(gpsWeek);
        fun(gpsSec);
        fun(posSmooth);
        fun(type);
        fun(attitude);
        fun(quaternion);
        fun(linearVelocity);
        fun(gyroBias);
        fun(accelBias);
        fun(sdPosition);
        fun(sdAttitude);
        fun(sdVelocity);
        fun(covariance);
        fun(sysStatus);
        fun(gpsStatus);
        fun(heading);
        fun(warn_info);
        fun(errorCode);
        fun(innerCode);
        fun(posGCJ02);
        fun(angularVelocity);
        fun(linearAcceleration);
        fun(mountingError);
        fun(sensorUsed);
        fun(wheelVelocity);
        fun(odoSF);
        fun(validEstimate);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("posSmooth", posSmooth);
        fun("type", type);
        fun("attitude", attitude);
        fun("quaternion", quaternion);
        fun("linearVelocity", linearVelocity);
        fun("gyroBias", gyroBias);
        fun("accelBias", accelBias);
        fun("sdPosition", sdPosition);
        fun("sdAttitude", sdAttitude);
        fun("sdVelocity", sdVelocity);
        fun("covariance", covariance);
        fun("sysStatus", sysStatus);
        fun("gpsStatus", gpsStatus);
        fun("heading", heading);
        fun("warn_info", warn_info);
        fun("errorCode", errorCode);
        fun("innerCode", innerCode);
        fun("posGCJ02", posGCJ02);
        fun("angularVelocity", angularVelocity);
        fun("linearAcceleration", linearAcceleration);
        fun("mountingError", mountingError);
        fun("sensorUsed", sensorUsed);
        fun("wheelVelocity", wheelVelocity);
        fun("odoSF", odoSF);
        fun("validEstimate", validEstimate);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("posSmooth", posSmooth);
        fun("type", type);
        fun("attitude", attitude);
        fun("quaternion", quaternion);
        fun("linearVelocity", linearVelocity);
        fun("gyroBias", gyroBias);
        fun("accelBias", accelBias);
        fun("sdPosition", sdPosition);
        fun("sdAttitude", sdAttitude);
        fun("sdVelocity", sdVelocity);
        fun("covariance", covariance);
        fun("sysStatus", sysStatus);
        fun("gpsStatus", gpsStatus);
        fun("heading", heading);
        fun("warn_info", warn_info);
        fun("errorCode", errorCode);
        fun("innerCode", innerCode);
        fun("posGCJ02", posGCJ02);
        fun("angularVelocity", angularVelocity);
        fun("linearAcceleration", linearAcceleration);
        fun("mountingError", mountingError);
        fun("sensorUsed", sensorUsed);
        fun("wheelVelocity", wheelVelocity);
        fun("odoSF", odoSF);
        fun("validEstimate", validEstimate);
    }

    bool operator==(const ::hozon::location::LocationNodeInfo& t) const
    {
        return (header == t.header) && (gpsWeek == t.gpsWeek) && (fabs(static_cast<double>(gpsSec - t.gpsSec)) < DBL_EPSILON) && (posSmooth == t.posSmooth) && (type == t.type) && (attitude == t.attitude) && (quaternion == t.quaternion) && (linearVelocity == t.linearVelocity) && (gyroBias == t.gyroBias) && (accelBias == t.accelBias) && (sdPosition == t.sdPosition) && (sdAttitude == t.sdAttitude) && (sdVelocity == t.sdVelocity) && (covariance == t.covariance) && (sysStatus == t.sysStatus) && (gpsStatus == t.gpsStatus) && (fabs(static_cast<double>(heading - t.heading)) < DBL_EPSILON) && (warn_info == t.warn_info) && (errorCode == t.errorCode) && (innerCode == t.innerCode) && (posGCJ02 == t.posGCJ02) && (angularVelocity == t.angularVelocity) && (linearAcceleration == t.linearAcceleration) && (mountingError == t.mountingError) && (sensorUsed == t.sensorUsed) && (fabs(static_cast<double>(wheelVelocity - t.wheelVelocity)) < DBL_EPSILON) && (fabs(static_cast<double>(odoSF - t.odoSF)) < DBL_EPSILON) && (validEstimate == t.validEstimate);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_LOCATIONNODEINFO_H
