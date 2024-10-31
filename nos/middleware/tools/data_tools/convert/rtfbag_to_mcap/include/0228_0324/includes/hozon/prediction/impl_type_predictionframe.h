/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PREDICTION_IMPL_TYPE_PREDICTIONFRAME_H
#define HOZON_PREDICTION_IMPL_TYPE_PREDICTIONFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "impl_type_double.h"
#include "hozon/prediction/impl_type_predictionobjvector.h"

namespace hozon {
namespace prediction {
struct PredictionFrame {
    ::hozon::common::CommonHeader header;
    ::UInt32 locSeq;
    ::Double startTime;
    ::Double endTime;
    ::hozon::prediction::PredictionObjVector predictionObjs;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(locSeq);
        fun(startTime);
        fun(endTime);
        fun(predictionObjs);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(locSeq);
        fun(startTime);
        fun(endTime);
        fun(predictionObjs);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("startTime", startTime);
        fun("endTime", endTime);
        fun("predictionObjs", predictionObjs);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("startTime", startTime);
        fun("endTime", endTime);
        fun("predictionObjs", predictionObjs);
    }

    bool operator==(const ::hozon::prediction::PredictionFrame& t) const
    {
        return (header == t.header) && (locSeq == t.locSeq) && (fabs(static_cast<double>(startTime - t.startTime)) < DBL_EPSILON) && (fabs(static_cast<double>(endTime - t.endTime)) < DBL_EPSILON) && (predictionObjs == t.predictionObjs);
    }
};
} // namespace prediction
} // namespace hozon


#endif // HOZON_PREDICTION_IMPL_TYPE_PREDICTIONFRAME_H
