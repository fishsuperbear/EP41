/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_ACCELWITHCOVARIANCE_H
#define IMPL_TYPE_ACCELWITHCOVARIANCE_H
#include <cfloat>
#include <cmath>
#include "impl_type_accel.h"
#include "impl_type_doublearray36.h"

struct AccelWithCovariance {
    ::Accel accel;
    ::DoubleArray36 covariance;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(accel);
        fun(covariance);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(accel);
        fun(covariance);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("accel", accel);
        fun("covariance", covariance);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("accel", accel);
        fun("covariance", covariance);
    }

    bool operator==(const ::AccelWithCovariance& t) const
    {
        return (accel == t.accel) && (covariance == t.covariance);
    }
};


#endif // IMPL_TYPE_ACCELWITHCOVARIANCE_H
