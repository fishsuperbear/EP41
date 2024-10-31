/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PREDICTION_IMPL_TYPE_PREDICTIONTRAJ_H
#define HOZON_PREDICTION_IMPL_TYPE_PREDICTIONTRAJ_H
#include <cfloat>
#include <cmath>
#include "hozon/prediction/impl_type_predictpointvector.h"
#include "impl_type_float.h"

namespace hozon {
namespace prediction {
struct PredictionTraj {
    ::hozon::prediction::PredictPointVector pathPoints;
    ::Float probability;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(pathPoints);
        fun(probability);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(pathPoints);
        fun(probability);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("pathPoints", pathPoints);
        fun("probability", probability);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("pathPoints", pathPoints);
        fun("probability", probability);
    }

    bool operator==(const ::hozon::prediction::PredictionTraj& t) const
    {
        return (pathPoints == t.pathPoints) && (fabs(static_cast<double>(probability - t.probability)) < DBL_EPSILON);
    }
};
} // namespace prediction
} // namespace hozon


#endif // HOZON_PREDICTION_IMPL_TYPE_PREDICTIONTRAJ_H
