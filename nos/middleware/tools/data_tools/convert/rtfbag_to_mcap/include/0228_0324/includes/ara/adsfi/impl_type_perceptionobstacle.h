/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_PERCEPTIONOBSTACLE_H
#define ARA_ADSFI_IMPL_TYPE_PERCEPTIONOBSTACLE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"
#include "impl_type_point.h"
#include "impl_type_double.h"
#include "impl_type_pointarray.h"
#include "impl_type_uint8_t.h"
#include "ara/adsfi/impl_type_time.h"
#include "impl_type_doublearray9.h"

namespace ara {
namespace adsfi {
struct PerceptionObstacle {
    ::uint32_t id;
    ::Point position;
    ::Double theta;
    ::Point velocity;
    bool hasVelocity;
    ::Point size;
    ::PointArray polygonPoint;
    ::Double trackingTime;
    ::uint8_t type;
    ::Double confidence;
    ::ara::adsfi::Time gpsTime;
    ::uint8_t confidenceType;
    ::PointArray drops;
    ::Point acceleration;
    ::Point anchorPoint;
    ::PointArray boundingBox;
    ::uint8_t subType;
    ::Double heightAboveGround;
    ::DoubleArray9 positionCovariance;
    ::DoubleArray9 velocityCovariance;
    ::DoubleArray9 accelerationCovariance;
    ::uint8_t lightStatus;

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
        fun(type);
        fun(confidence);
        fun(gpsTime);
        fun(confidenceType);
        fun(drops);
        fun(acceleration);
        fun(anchorPoint);
        fun(boundingBox);
        fun(subType);
        fun(heightAboveGround);
        fun(positionCovariance);
        fun(velocityCovariance);
        fun(accelerationCovariance);
        fun(lightStatus);
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
        fun(type);
        fun(confidence);
        fun(gpsTime);
        fun(confidenceType);
        fun(drops);
        fun(acceleration);
        fun(anchorPoint);
        fun(boundingBox);
        fun(subType);
        fun(heightAboveGround);
        fun(positionCovariance);
        fun(velocityCovariance);
        fun(accelerationCovariance);
        fun(lightStatus);
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
        fun("type", type);
        fun("confidence", confidence);
        fun("gpsTime", gpsTime);
        fun("confidenceType", confidenceType);
        fun("drops", drops);
        fun("acceleration", acceleration);
        fun("anchorPoint", anchorPoint);
        fun("boundingBox", boundingBox);
        fun("subType", subType);
        fun("heightAboveGround", heightAboveGround);
        fun("positionCovariance", positionCovariance);
        fun("velocityCovariance", velocityCovariance);
        fun("accelerationCovariance", accelerationCovariance);
        fun("lightStatus", lightStatus);
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
        fun("type", type);
        fun("confidence", confidence);
        fun("gpsTime", gpsTime);
        fun("confidenceType", confidenceType);
        fun("drops", drops);
        fun("acceleration", acceleration);
        fun("anchorPoint", anchorPoint);
        fun("boundingBox", boundingBox);
        fun("subType", subType);
        fun("heightAboveGround", heightAboveGround);
        fun("positionCovariance", positionCovariance);
        fun("velocityCovariance", velocityCovariance);
        fun("accelerationCovariance", accelerationCovariance);
        fun("lightStatus", lightStatus);
    }

    bool operator==(const ::ara::adsfi::PerceptionObstacle& t) const
    {
        return (id == t.id) && (position == t.position) && (fabs(static_cast<double>(theta - t.theta)) < DBL_EPSILON) && (velocity == t.velocity) && (hasVelocity == t.hasVelocity) && (size == t.size) && (polygonPoint == t.polygonPoint) && (fabs(static_cast<double>(trackingTime - t.trackingTime)) < DBL_EPSILON) && (type == t.type) && (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON) && (gpsTime == t.gpsTime) && (confidenceType == t.confidenceType) && (drops == t.drops) && (acceleration == t.acceleration) && (anchorPoint == t.anchorPoint) && (boundingBox == t.boundingBox) && (subType == t.subType) && (fabs(static_cast<double>(heightAboveGround - t.heightAboveGround)) < DBL_EPSILON) && (positionCovariance == t.positionCovariance) && (velocityCovariance == t.velocityCovariance) && (accelerationCovariance == t.accelerationCovariance) && (lightStatus == t.lightStatus);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_PERCEPTIONOBSTACLE_H
