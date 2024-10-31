/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_POSEWITHCOVARIANCESTAMPED_H
#define HOZON_COMPOSITE_IMPL_TYPE_POSEWITHCOVARIANCESTAMPED_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/composite/impl_type_posewithcovariance.h"

namespace hozon {
namespace composite {
struct PoseWithCovarianceStamped {
    ::hozon::common::CommonHeader header;
    ::hozon::composite::PoseWithCovariance pose;

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

    bool operator==(const ::hozon::composite::PoseWithCovarianceStamped& t) const
    {
        return (header == t.header) && (pose == t.pose);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_POSEWITHCOVARIANCESTAMPED_H
