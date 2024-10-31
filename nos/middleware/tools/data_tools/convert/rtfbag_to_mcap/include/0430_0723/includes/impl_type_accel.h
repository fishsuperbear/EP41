/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_ACCEL_H
#define IMPL_TYPE_ACCEL_H
#include <cfloat>
#include <cmath>
#include "impl_type_vector3.h"

struct Accel {
    ::Vector3 linear;
    ::Vector3 angular;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(linear);
        fun(angular);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(linear);
        fun(angular);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("linear", linear);
        fun("angular", angular);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("linear", linear);
        fun("angular", angular);
    }

    bool operator==(const ::Accel& t) const
    {
        return (linear == t.linear) && (angular == t.angular);
    }
};


#endif // IMPL_TYPE_ACCEL_H
