/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_PATHWITHCOVARIANCE_H
#define IMPL_TYPE_PATHWITHCOVARIANCE_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_posewithcovariancestamped.h"
#include "impl_type_timewindow.h"

struct PathWithCovariance {
    ::ara::common::CommonHeader header;
    ::PoseWithCovarianceStamped poses;
    ::TimeWindow time_window;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(poses);
        fun(time_window);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(poses);
        fun(time_window);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("poses", poses);
        fun("time_window", time_window);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("poses", poses);
        fun("time_window", time_window);
    }

    bool operator==(const ::PathWithCovariance& t) const
    {
        return (header == t.header) && (poses == t.poses) && (time_window == t.time_window);
    }
};


#endif // IMPL_TYPE_PATHWITHCOVARIANCE_H
