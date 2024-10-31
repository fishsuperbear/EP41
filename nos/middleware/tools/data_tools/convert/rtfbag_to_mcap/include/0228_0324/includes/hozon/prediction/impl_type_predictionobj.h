/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PREDICTION_IMPL_TYPE_PREDICTIONOBJ_H
#define HOZON_PREDICTION_IMPL_TYPE_PREDICTIONOBJ_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/prediction/impl_type_predictiontrajvector.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace prediction {
struct PredictionObj {
    ::UInt32 id;
    ::hozon::prediction::PredictionTrajVector trajectory;
    ::UInt8 intent;
    ::Float intentProbability;
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
        fun(trajectory);
        fun(intent);
        fun(intentProbability);
        fun(priority);
        fun(isStatic);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(trajectory);
        fun(intent);
        fun(intentProbability);
        fun(priority);
        fun(isStatic);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("trajectory", trajectory);
        fun("intent", intent);
        fun("intentProbability", intentProbability);
        fun("priority", priority);
        fun("isStatic", isStatic);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("trajectory", trajectory);
        fun("intent", intent);
        fun("intentProbability", intentProbability);
        fun("priority", priority);
        fun("isStatic", isStatic);
    }

    bool operator==(const ::hozon::prediction::PredictionObj& t) const
    {
        return (id == t.id) && (trajectory == t.trajectory) && (intent == t.intent) && (fabs(static_cast<double>(intentProbability - t.intentProbability)) < DBL_EPSILON) && (priority == t.priority) && (isStatic == t.isStatic);
    }
};
} // namespace prediction
} // namespace hozon


#endif // HOZON_PREDICTION_IMPL_TYPE_PREDICTIONOBJ_H
