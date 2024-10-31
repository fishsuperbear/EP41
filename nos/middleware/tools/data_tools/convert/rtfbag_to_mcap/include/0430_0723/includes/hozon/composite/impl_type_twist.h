/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_TWIST_H
#define HOZON_COMPOSITE_IMPL_TYPE_TWIST_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_vector3.h"

namespace hozon {
namespace composite {
struct Twist {
    ::hozon::composite::Vector3 linear;
    ::hozon::composite::Vector3 angularRaw;
    ::hozon::composite::Vector3 angular;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(linear);
        fun(angularRaw);
        fun(angular);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(linear);
        fun(angularRaw);
        fun(angular);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("linear", linear);
        fun("angularRaw", angularRaw);
        fun("angular", angular);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("linear", linear);
        fun("angularRaw", angularRaw);
        fun("angular", angular);
    }

    bool operator==(const ::hozon::composite::Twist& t) const
    {
        return (linear == t.linear) && (angularRaw == t.angularRaw) && (angular == t.angular);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_TWIST_H
