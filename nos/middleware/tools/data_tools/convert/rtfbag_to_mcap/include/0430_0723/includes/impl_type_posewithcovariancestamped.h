/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_POSEWITHCOVARIANCESTAMPED_H
#define IMPL_TYPE_POSEWITHCOVARIANCESTAMPED_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_posewithcovariance.h"

struct PoseWithCovarianceStamped {
    ::ara::common::CommonHeader header;
    ::PoseWithCovariance pose;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(pose);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(pose);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("pose", pose);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("pose", pose);
    }

    bool operator==(const ::PoseWithCovarianceStamped& t) const
    {
        return (header == t.header) && (pose == t.pose);
    }
};


#endif // IMPL_TYPE_POSEWITHCOVARIANCESTAMPED_H
