/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_TWISTWITHCOVARIANCE_H
#define IMPL_TYPE_TWISTWITHCOVARIANCE_H
#include <cfloat>
#include <cmath>
#include "impl_type_twist.h"
#include "impl_type_doublearray36.h"

struct TwistWithCovariance {
    ::Twist twist;
    ::DoubleArray36 covariance;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(twist);
        fun(covariance);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(twist);
        fun(covariance);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("twist", twist);
        fun("covariance", covariance);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("twist", twist);
        fun("covariance", covariance);
    }

    bool operator==(const ::TwistWithCovariance& t) const
    {
        return (twist == t.twist) && (covariance == t.covariance);
    }
};


#endif // IMPL_TYPE_TWISTWITHCOVARIANCE_H
