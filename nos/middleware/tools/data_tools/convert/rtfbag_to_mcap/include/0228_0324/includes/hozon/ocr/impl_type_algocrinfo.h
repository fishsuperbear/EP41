/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OCR_IMPL_TYPE_ALGOCRINFO_H
#define HOZON_OCR_IMPL_TYPE_ALGOCRINFO_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_point2d.h"
#include "impl_type_string.h"

namespace hozon {
namespace ocr {
struct AlgOcrinfo {
    ::hozon::composite::Point2D center;
    float ratio;
    ::String result;
    float confidence;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(center);
        fun(ratio);
        fun(result);
        fun(confidence);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(center);
        fun(ratio);
        fun(result);
        fun(confidence);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("center", center);
        fun("ratio", ratio);
        fun("result", result);
        fun("confidence", confidence);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("center", center);
        fun("ratio", ratio);
        fun("result", result);
        fun("confidence", confidence);
    }

    bool operator==(const ::hozon::ocr::AlgOcrinfo& t) const
    {
        return (center == t.center) && (fabs(static_cast<double>(ratio - t.ratio)) < DBL_EPSILON) && (result == t.result) && (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON);
    }
};
} // namespace ocr
} // namespace hozon


#endif // HOZON_OCR_IMPL_TYPE_ALGOCRINFO_H
