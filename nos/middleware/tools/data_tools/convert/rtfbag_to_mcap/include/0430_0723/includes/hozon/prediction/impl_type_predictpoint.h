/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PREDICTION_IMPL_TYPE_PREDICTPOINT_H
#define HOZON_PREDICTION_IMPL_TYPE_PREDICTPOINT_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"

namespace hozon {
namespace prediction {
struct PredictPoint {
    ::Float x;
    ::Float y;
    ::Float z;
    ::Float theta;
    ::Float velocity;
    ::Float acc;
    ::Float confidence;
    ::Float timeRelative;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(x);
        fun(y);
        fun(z);
        fun(theta);
        fun(velocity);
        fun(acc);
        fun(confidence);
        fun(timeRelative);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(z);
        fun(theta);
        fun(velocity);
        fun(acc);
        fun(confidence);
        fun(timeRelative);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("theta", theta);
        fun("velocity", velocity);
        fun("acc", acc);
        fun("confidence", confidence);
        fun("timeRelative", timeRelative);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("theta", theta);
        fun("velocity", velocity);
        fun("acc", acc);
        fun("confidence", confidence);
        fun("timeRelative", timeRelative);
    }

    bool operator==(const ::hozon::prediction::PredictPoint& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(theta - t.theta)) < DBL_EPSILON) && (fabs(static_cast<double>(velocity - t.velocity)) < DBL_EPSILON) && (fabs(static_cast<double>(acc - t.acc)) < DBL_EPSILON) && (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON) && (fabs(static_cast<double>(timeRelative - t.timeRelative)) < DBL_EPSILON);
    }
};
} // namespace prediction
} // namespace hozon


#endif // HOZON_PREDICTION_IMPL_TYPE_PREDICTPOINT_H
