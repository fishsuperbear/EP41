/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_OBJPREDICTIONOUTARRAY_H
#define ARA_ADSFI_IMPL_TYPE_OBJPREDICTIONOUTARRAY_H
#include <cfloat>
#include <cmath>
#include "ara/egotrajectory/impl_type_header.h"
#include "ara/adsfi/impl_type_predictionobstaclevector.h"
#include "ara/adsfi/impl_type_time.h"
#include "impl_type_uint8_t.h"
#include "impl_type_uint16_t.h"

namespace ara {
namespace adsfi {
struct ObjPredictionOutArray {
    ::ara::egotrajectory::Header header;
    bool isValid;
    ::ara::adsfi::PredictionObstacleVector predictionObstacle;
    ::ara::adsfi::Time startTime;
    ::ara::adsfi::Time endTime;
    ::uint8_t changeOriginFlag;
    ::uint8_t selfIntent;
    ::uint16_t scenario;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(isValid);
        fun(predictionObstacle);
        fun(startTime);
        fun(endTime);
        fun(changeOriginFlag);
        fun(selfIntent);
        fun(scenario);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(isValid);
        fun(predictionObstacle);
        fun(startTime);
        fun(endTime);
        fun(changeOriginFlag);
        fun(selfIntent);
        fun(scenario);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("isValid", isValid);
        fun("predictionObstacle", predictionObstacle);
        fun("startTime", startTime);
        fun("endTime", endTime);
        fun("changeOriginFlag", changeOriginFlag);
        fun("selfIntent", selfIntent);
        fun("scenario", scenario);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("isValid", isValid);
        fun("predictionObstacle", predictionObstacle);
        fun("startTime", startTime);
        fun("endTime", endTime);
        fun("changeOriginFlag", changeOriginFlag);
        fun("selfIntent", selfIntent);
        fun("scenario", scenario);
    }

    bool operator==(const ::ara::adsfi::ObjPredictionOutArray& t) const
    {
        return (header == t.header) && (isValid == t.isValid) && (predictionObstacle == t.predictionObstacle) && (startTime == t.startTime) && (endTime == t.endTime) && (changeOriginFlag == t.changeOriginFlag) && (selfIntent == t.selfIntent) && (scenario == t.scenario);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_OBJPREDICTIONOUTARRAY_H
