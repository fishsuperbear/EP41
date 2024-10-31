/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_POSEWITHCOVARIANCE_H
#define IMPL_TYPE_POSEWITHCOVARIANCE_H
#include <cfloat>
#include <cmath>
#include "impl_type_pose.h"
#include "impl_type_doublearray36.h"

struct PoseWithCovariance {
    ::Pose pose;
    ::DoubleArray36 covariance;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(pose);
        fun(covariance);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(pose);
        fun(covariance);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("pose", pose);
        fun("covariance", covariance);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("pose", pose);
        fun("covariance", covariance);
    }

    bool operator==(const ::PoseWithCovariance& t) const
    {
        return (pose == t.pose) && (covariance == t.covariance);
    }
};


#endif // IMPL_TYPE_POSEWITHCOVARIANCE_H
