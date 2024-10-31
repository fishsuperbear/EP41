/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_BODYREPORTFLOAT32WITHVALID_H
#define IMPL_TYPE_BODYREPORTFLOAT32WITHVALID_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_uint8.h"

struct BodyReportFloat32WithValid {
    ::Float value;
    ::UInt8 confidence;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(value);
        fun(confidence);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(value);
        fun(confidence);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("value", value);
        fun("confidence", confidence);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("value", value);
        fun("confidence", confidence);
    }

    bool operator==(const ::BodyReportFloat32WithValid& t) const
    {
        return (fabs(static_cast<double>(value - t.value)) < DBL_EPSILON) && (confidence == t.confidence);
    }
};


#endif // IMPL_TYPE_BODYREPORTFLOAT32WITHVALID_H
