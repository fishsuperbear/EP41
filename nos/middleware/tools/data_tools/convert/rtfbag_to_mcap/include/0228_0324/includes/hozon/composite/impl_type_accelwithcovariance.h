/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_ACCELWITHCOVARIANCE_H
#define HOZON_COMPOSITE_IMPL_TYPE_ACCELWITHCOVARIANCE_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_accel.h"
#include "hozon/composite/impl_type_floatarray36.h"

namespace hozon {
namespace composite {
struct AccelWithCovariance {
    ::hozon::composite::Accel accel;
    ::hozon::composite::FloatArray36 covariance;

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

    bool operator==(const ::hozon::composite::AccelWithCovariance& t) const
    {
        return (accel == t.accel) && (covariance == t.covariance);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_ACCELWITHCOVARIANCE_H
