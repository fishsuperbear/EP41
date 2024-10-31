/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_TYRESTATUS_H
#define IMPL_TYPE_TYRESTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_bodyreportuint8withvalid.h"

struct TyreStatus {
    ::Float tyrePressureValue;
    ::BodyReportUint8WithValid tyrePressureLeakageStatus;
    ::BodyReportUint8WithValid tyrePressureTirePStatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(tyrePressureValue);
        fun(tyrePressureLeakageStatus);
        fun(tyrePressureTirePStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(tyrePressureValue);
        fun(tyrePressureLeakageStatus);
        fun(tyrePressureTirePStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("tyrePressureValue", tyrePressureValue);
        fun("tyrePressureLeakageStatus", tyrePressureLeakageStatus);
        fun("tyrePressureTirePStatus", tyrePressureTirePStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("tyrePressureValue", tyrePressureValue);
        fun("tyrePressureLeakageStatus", tyrePressureLeakageStatus);
        fun("tyrePressureTirePStatus", tyrePressureTirePStatus);
    }

    bool operator==(const ::TyreStatus& t) const
    {
        return (fabs(static_cast<double>(tyrePressureValue - t.tyrePressureValue)) < DBL_EPSILON) && (tyrePressureLeakageStatus == t.tyrePressureLeakageStatus) && (tyrePressureTirePStatus == t.tyrePressureTirePStatus);
    }
};


#endif // IMPL_TYPE_TYRESTATUS_H
