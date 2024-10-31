/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_ACCEL_H
#define HOZON_COMPOSITE_IMPL_TYPE_ACCEL_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_vector3.h"

namespace hozon {
namespace composite {
struct Accel {
    ::hozon::composite::Vector3 linearRaw;
    ::hozon::composite::Vector3 linear;
    ::hozon::composite::Vector3 angular;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(linearRaw);
        fun(linear);
        fun(angular);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(linearRaw);
        fun(linear);
        fun(angular);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("linearRaw", linearRaw);
        fun("linear", linear);
        fun("angular", angular);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("linearRaw", linearRaw);
        fun("linear", linear);
        fun("angular", angular);
    }

    bool operator==(const ::hozon::composite::Accel& t) const
    {
        return (linearRaw == t.linearRaw) && (linear == t.linear) && (angular == t.angular);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_ACCEL_H
