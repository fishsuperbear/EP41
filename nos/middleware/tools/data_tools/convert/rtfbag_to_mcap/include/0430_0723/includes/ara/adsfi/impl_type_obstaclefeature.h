/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_OBSTACLEFEATURE_H
#define ARA_ADSFI_IMPL_TYPE_OBSTACLEFEATURE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"
#include "impl_type_pointarray.h"
#include "impl_type_point.h"
#include "impl_type_double.h"
#include "ara/adsfi/impl_type_time.h"
#include "impl_type_uint8_t.h"
#include "ara/adsfi/impl_type_predictiontrajectorypointvector.h"
#include "ara/adsfi/impl_type_trajectorypointvector.h"
#include "ara/adsfi/impl_type_trajectoryinpredictionvector.h"

namespace ara {
namespace adsfi {
struct ObstacleFeature {
    ::uint32_t id;
    ::PointArray polygonPoint;
    ::Point position;
    ::Point frontPosition;
    ::Point velocity;
    ::Point rawVelocity;
    ::Point acceleration;
    ::Double velocityHeading;
    ::Double speed;
    ::Double acc;
    ::Double theta;
    ::Double length;
    ::Double width;
    ::Double height;
    ::Double trackingTime;
    ::ara::adsfi::Time timestamp;
    ::Point tPosition;
    ::Point tVelocity;
    ::Double tVelocityHeading;
    ::Double tSpeed;
    ::Point tAcceleration;
    ::Double tAcc;
    bool isStill;
    ::uint8_t type;
    ::Double labelUpdateTimeDelta;
    ::uint8_t priority;
    bool isNearJunction;
    ::ara::adsfi::PredictionTrajectoryPointVector futureTrajectoryPoints;
    ::ara::adsfi::TrajectoryPointVector shortTermPredictedTrajectoryPoints;
    ::ara::adsfi::TrajectoryInPredictionVector predictedTrajectory;
    ::ara::adsfi::TrajectoryPointVector adcTrajectoryPoint;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(polygonPoint);
        fun(position);
        fun(frontPosition);
        fun(velocity);
        fun(rawVelocity);
        fun(acceleration);
        fun(velocityHeading);
        fun(speed);
        fun(acc);
        fun(theta);
        fun(length);
        fun(width);
        fun(height);
        fun(trackingTime);
        fun(timestamp);
        fun(tPosition);
        fun(tVelocity);
        fun(tVelocityHeading);
        fun(tSpeed);
        fun(tAcceleration);
        fun(tAcc);
        fun(isStill);
        fun(type);
        fun(labelUpdateTimeDelta);
        fun(priority);
        fun(isNearJunction);
        fun(futureTrajectoryPoints);
        fun(shortTermPredictedTrajectoryPoints);
        fun(predictedTrajectory);
        fun(adcTrajectoryPoint);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(polygonPoint);
        fun(position);
        fun(frontPosition);
        fun(velocity);
        fun(rawVelocity);
        fun(acceleration);
        fun(velocityHeading);
        fun(speed);
        fun(acc);
        fun(theta);
        fun(length);
        fun(width);
        fun(height);
        fun(trackingTime);
        fun(timestamp);
        fun(tPosition);
        fun(tVelocity);
        fun(tVelocityHeading);
        fun(tSpeed);
        fun(tAcceleration);
        fun(tAcc);
        fun(isStill);
        fun(type);
        fun(labelUpdateTimeDelta);
        fun(priority);
        fun(isNearJunction);
        fun(futureTrajectoryPoints);
        fun(shortTermPredictedTrajectoryPoints);
        fun(predictedTrajectory);
        fun(adcTrajectoryPoint);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("polygonPoint", polygonPoint);
        fun("position", position);
        fun("frontPosition", frontPosition);
        fun("velocity", velocity);
        fun("rawVelocity", rawVelocity);
        fun("acceleration", acceleration);
        fun("velocityHeading", velocityHeading);
        fun("speed", speed);
        fun("acc", acc);
        fun("theta", theta);
        fun("length", length);
        fun("width", width);
        fun("height", height);
        fun("trackingTime", trackingTime);
        fun("timestamp", timestamp);
        fun("tPosition", tPosition);
        fun("tVelocity", tVelocity);
        fun("tVelocityHeading", tVelocityHeading);
        fun("tSpeed", tSpeed);
        fun("tAcceleration", tAcceleration);
        fun("tAcc", tAcc);
        fun("isStill", isStill);
        fun("type", type);
        fun("labelUpdateTimeDelta", labelUpdateTimeDelta);
        fun("priority", priority);
        fun("isNearJunction", isNearJunction);
        fun("futureTrajectoryPoints", futureTrajectoryPoints);
        fun("shortTermPredictedTrajectoryPoints", shortTermPredictedTrajectoryPoints);
        fun("predictedTrajectory", predictedTrajectory);
        fun("adcTrajectoryPoint", adcTrajectoryPoint);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("polygonPoint", polygonPoint);
        fun("position", position);
        fun("frontPosition", frontPosition);
        fun("velocity", velocity);
        fun("rawVelocity", rawVelocity);
        fun("acceleration", acceleration);
        fun("velocityHeading", velocityHeading);
        fun("speed", speed);
        fun("acc", acc);
        fun("theta", theta);
        fun("length", length);
        fun("width", width);
        fun("height", height);
        fun("trackingTime", trackingTime);
        fun("timestamp", timestamp);
        fun("tPosition", tPosition);
        fun("tVelocity", tVelocity);
        fun("tVelocityHeading", tVelocityHeading);
        fun("tSpeed", tSpeed);
        fun("tAcceleration", tAcceleration);
        fun("tAcc", tAcc);
        fun("isStill", isStill);
        fun("type", type);
        fun("labelUpdateTimeDelta", labelUpdateTimeDelta);
        fun("priority", priority);
        fun("isNearJunction", isNearJunction);
        fun("futureTrajectoryPoints", futureTrajectoryPoints);
        fun("shortTermPredictedTrajectoryPoints", shortTermPredictedTrajectoryPoints);
        fun("predictedTrajectory", predictedTrajectory);
        fun("adcTrajectoryPoint", adcTrajectoryPoint);
    }

    bool operator==(const ::ara::adsfi::ObstacleFeature& t) const
    {
        return (id == t.id) && (polygonPoint == t.polygonPoint) && (position == t.position) && (frontPosition == t.frontPosition) && (velocity == t.velocity) && (rawVelocity == t.rawVelocity) && (acceleration == t.acceleration) && (fabs(static_cast<double>(velocityHeading - t.velocityHeading)) < DBL_EPSILON) && (fabs(static_cast<double>(speed - t.speed)) < DBL_EPSILON) && (fabs(static_cast<double>(acc - t.acc)) < DBL_EPSILON) && (fabs(static_cast<double>(theta - t.theta)) < DBL_EPSILON) && (fabs(static_cast<double>(length - t.length)) < DBL_EPSILON) && (fabs(static_cast<double>(width - t.width)) < DBL_EPSILON) && (fabs(static_cast<double>(height - t.height)) < DBL_EPSILON) && (fabs(static_cast<double>(trackingTime - t.trackingTime)) < DBL_EPSILON) && (timestamp == t.timestamp) && (tPosition == t.tPosition) && (tVelocity == t.tVelocity) && (fabs(static_cast<double>(tVelocityHeading - t.tVelocityHeading)) < DBL_EPSILON) && (fabs(static_cast<double>(tSpeed - t.tSpeed)) < DBL_EPSILON) && (tAcceleration == t.tAcceleration) && (fabs(static_cast<double>(tAcc - t.tAcc)) < DBL_EPSILON) && (isStill == t.isStill) && (type == t.type) && (fabs(static_cast<double>(labelUpdateTimeDelta - t.labelUpdateTimeDelta)) < DBL_EPSILON) && (priority == t.priority) && (isNearJunction == t.isNearJunction) && (futureTrajectoryPoints == t.futureTrajectoryPoints) && (shortTermPredictedTrajectoryPoints == t.shortTermPredictedTrajectoryPoints) && (predictedTrajectory == t.predictedTrajectory) && (adcTrajectoryPoint == t.adcTrajectoryPoint);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_OBSTACLEFEATURE_H
