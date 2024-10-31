/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_TWISTWITHCOVARIANCE_H
#define HOZON_COMPOSITE_IMPL_TYPE_TWISTWITHCOVARIANCE_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_twist.h"
#include "hozon/composite/impl_type_floatarray36.h"

namespace hozon {
namespace composite {
struct TwistWithCovariance {
    ::hozon::composite::Twist twist;
    ::hozon::composite::FloatArray36 covariance;

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

    bool operator==(const ::hozon::composite::TwistWithCovariance& t) const
    {
        return (twist == t.twist) && (covariance == t.covariance);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_TWISTWITHCOVARIANCE_H
