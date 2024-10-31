/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LANELINE_IMPL_TYPE_LANEPARAM_H
#define HOZON_LANELINE_IMPL_TYPE_LANEPARAM_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"

namespace hozon {
namespace laneline {
struct LaneParam {
    ::Float a;
    ::Float b;
    ::Float c;
    ::Float d;
    ::Float dev_a;
    ::Float dev_b;
    ::Float dev_c;
    ::Float dev_d;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(a);
        fun(b);
        fun(c);
        fun(d);
        fun(dev_a);
        fun(dev_b);
        fun(dev_c);
        fun(dev_d);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(a);
        fun(b);
        fun(c);
        fun(d);
        fun(dev_a);
        fun(dev_b);
        fun(dev_c);
        fun(dev_d);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("a", a);
        fun("b", b);
        fun("c", c);
        fun("d", d);
        fun("dev_a", dev_a);
        fun("dev_b", dev_b);
        fun("dev_c", dev_c);
        fun("dev_d", dev_d);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("a", a);
        fun("b", b);
        fun("c", c);
        fun("d", d);
        fun("dev_a", dev_a);
        fun("dev_b", dev_b);
        fun("dev_c", dev_c);
        fun("dev_d", dev_d);
    }

    bool operator==(const ::hozon::laneline::LaneParam& t) const
    {
        return (fabs(static_cast<double>(a - t.a)) < DBL_EPSILON) && (fabs(static_cast<double>(b - t.b)) < DBL_EPSILON) && (fabs(static_cast<double>(c - t.c)) < DBL_EPSILON) && (fabs(static_cast<double>(d - t.d)) < DBL_EPSILON) && (fabs(static_cast<double>(dev_a - t.dev_a)) < DBL_EPSILON) && (fabs(static_cast<double>(dev_b - t.dev_b)) < DBL_EPSILON) && (fabs(static_cast<double>(dev_c - t.dev_c)) < DBL_EPSILON) && (fabs(static_cast<double>(dev_d - t.dev_d)) < DBL_EPSILON);
    }
};
} // namespace laneline
} // namespace hozon


#endif // HOZON_LANELINE_IMPL_TYPE_LANEPARAM_H
