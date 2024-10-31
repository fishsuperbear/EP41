/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_PREDICTIONOBSTACLE_H
#define ARA_ADSFI_IMPL_TYPE_PREDICTIONOBSTACLE_H
#include <cfloat>
#include <cmath>
#include "ara/adsfi/impl_type_perceptionobstacle.h"
#include "ara/adsfi/impl_type_time.h"
#include "impl_type_double.h"
#include "ara/adsfi/impl_type_trajectoryinpredictionvector.h"
#include "impl_type_uint8_t.h"
#include "ara/adsfi/impl_type_obstaclefeature.h"

namespace ara {
namespace adsfi {
struct PredictionObstacle {
    ::ara::adsfi::PerceptionObstacle object;
    ::ara::adsfi::Time gpsTime;
    ::Double predictedPeriod;
    ::ara::adsfi::TrajectoryInPredictionVector trajectory;
    ::uint8_t intent;
    ::uint8_t priority;
    bool isStatic;
    ::ara::adsfi::ObstacleFeature feature;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(object);
        fun(gpsTime);
        fun(predictedPeriod);
        fun(trajectory);
        fun(intent);
        fun(priority);
        fun(isStatic);
        fun(feature);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(object);
        fun(gpsTime);
        fun(predictedPeriod);
        fun(trajectory);
        fun(intent);
        fun(priority);
        fun(isStatic);
        fun(feature);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("object", object);
        fun("gpsTime", gpsTime);
        fun("predictedPeriod", predictedPeriod);
        fun("trajectory", trajectory);
        fun("intent", intent);
        fun("priority", priority);
        fun("isStatic", isStatic);
        fun("feature", feature);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("object", object);
        fun("gpsTime", gpsTime);
        fun("predictedPeriod", predictedPeriod);
        fun("trajectory", trajectory);
        fun("intent", intent);
        fun("priority", priority);
        fun("isStatic", isStatic);
        fun("feature", feature);
    }

    bool operator==(const ::ara::adsfi::PredictionObstacle& t) const
    {
        return (object == t.object) && (gpsTime == t.gpsTime) && (fabs(static_cast<double>(predictedPeriod - t.predictedPeriod)) < DBL_EPSILON) && (trajectory == t.trajectory) && (intent == t.intent) && (priority == t.priority) && (isStatic == t.isStatic) && (feature == t.feature);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_PREDICTIONOBSTACLE_H
