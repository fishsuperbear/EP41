/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_PREDICTOBJECT_H
#define ADSFI_IMPL_TYPE_PREDICTOBJECT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_point.h"
#include "impl_type_double.h"
#include "impl_type_boolean.h"
#include "impl_type_pointarray.h"
#include "impl_type_uint8.h"
#include "ara/common/impl_type_commontime.h"
#include "impl_type_matrix3d.h"
#include "adsfi/impl_type_trajinpredictionvector.h"

namespace adsfi {
struct PredictObject {
    ::UInt32 id;
    ::Point position;
    ::Double theta;
    ::Point velocity;
    ::Boolean hasVelocity;
    ::Point size;
    ::PointArray polygonPoint;
    ::Double trackingTime;
    ::UInt8 cls;
    ::Double confidence;
    ::ara::common::CommonTime gpsTime;
    ::Point acceleration;
    ::Point anchorPoint;
    ::PointArray boundingBox;
    ::Double heightAboveGround;
    ::Matrix3d positionCovariance;
    ::Matrix3d velocityCovariance;
    ::Matrix3d accelerationCovariance;
    ::UInt8 lightStatus;
    ::Double predictedPeriod;
    ::adsfi::TrajInPredictionVector trajectory;
    ::UInt8 intent;
    ::UInt8 priority;
    ::Boolean isStatic;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(position);
        fun(theta);
        fun(velocity);
        fun(hasVelocity);
        fun(size);
        fun(polygonPoint);
        fun(trackingTime);
        fun(cls);
        fun(confidence);
        fun(gpsTime);
        fun(acceleration);
        fun(anchorPoint);
        fun(boundingBox);
        fun(heightAboveGround);
        fun(positionCovariance);
        fun(velocityCovariance);
        fun(accelerationCovariance);
        fun(lightStatus);
        fun(predictedPeriod);
        fun(trajectory);
        fun(intent);
        fun(priority);
        fun(isStatic);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(position);
        fun(theta);
        fun(velocity);
        fun(hasVelocity);
        fun(size);
        fun(polygonPoint);
        fun(trackingTime);
        fun(cls);
        fun(confidence);
        fun(gpsTime);
        fun(acceleration);
        fun(anchorPoint);
        fun(boundingBox);
        fun(heightAboveGround);
        fun(positionCovariance);
        fun(velocityCovariance);
        fun(accelerationCovariance);
        fun(lightStatus);
        fun(predictedPeriod);
        fun(trajectory);
        fun(intent);
        fun(priority);
        fun(isStatic);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("position", position);
        fun("theta", theta);
        fun("velocity", velocity);
        fun("hasVelocity", hasVelocity);
        fun("size", size);
        fun("polygonPoint", polygonPoint);
        fun("trackingTime", trackingTime);
        fun("cls", cls);
        fun("confidence", confidence);
        fun("gpsTime", gpsTime);
        fun("acceleration", acceleration);
        fun("anchorPoint", anchorPoint);
        fun("boundingBox", boundingBox);
        fun("heightAboveGround", heightAboveGround);
        fun("positionCovariance", positionCovariance);
        fun("velocityCovariance", velocityCovariance);
        fun("accelerationCovariance", accelerationCovariance);
        fun("lightStatus", lightStatus);
        fun("predictedPeriod", predictedPeriod);
        fun("trajectory", trajectory);
        fun("intent", intent);
        fun("priority", priority);
        fun("isStatic", isStatic);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("position", position);
        fun("theta", theta);
        fun("velocity", velocity);
        fun("hasVelocity", hasVelocity);
        fun("size", size);
        fun("polygonPoint", polygonPoint);
        fun("trackingTime", trackingTime);
        fun("cls", cls);
        fun("confidence", confidence);
        fun("gpsTime", gpsTime);
        fun("acceleration", acceleration);
        fun("anchorPoint", anchorPoint);
        fun("boundingBox", boundingBox);
        fun("heightAboveGround", heightAboveGround);
        fun("positionCovariance", positionCovariance);
        fun("velocityCovariance", velocityCovariance);
        fun("accelerationCovariance", accelerationCovariance);
        fun("lightStatus", lightStatus);
        fun("predictedPeriod", predictedPeriod);
        fun("trajectory", trajectory);
        fun("intent", intent);
        fun("priority", priority);
        fun("isStatic", isStatic);
    }

    bool operator==(const ::adsfi::PredictObject& t) const
    {
        return (id == t.id) && (position == t.position) && (fabs(static_cast<double>(theta - t.theta)) < DBL_EPSILON) && (velocity == t.velocity) && (hasVelocity == t.hasVelocity) && (size == t.size) && (polygonPoint == t.polygonPoint) && (fabs(static_cast<double>(trackingTime - t.trackingTime)) < DBL_EPSILON) && (cls == t.cls) && (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON) && (gpsTime == t.gpsTime) && (acceleration == t.acceleration) && (anchorPoint == t.anchorPoint) && (boundingBox == t.boundingBox) && (fabs(static_cast<double>(heightAboveGround - t.heightAboveGround)) < DBL_EPSILON) && (positionCovariance == t.positionCovariance) && (velocityCovariance == t.velocityCovariance) && (accelerationCovariance == t.accelerationCovariance) && (lightStatus == t.lightStatus) && (fabs(static_cast<double>(predictedPeriod - t.predictedPeriod)) < DBL_EPSILON) && (trajectory == t.trajectory) && (intent == t.intent) && (priority == t.priority) && (isStatic == t.isStatic);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_PREDICTOBJECT_H
