/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_LANELINEFITPARAM_H
#define ARA_ADSFI_IMPL_TYPE_LANELINEFITPARAM_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"

namespace ara {
namespace adsfi {
struct LaneLineFitParam {
    ::Double a;
    ::Double b;
    ::Double c;
    ::Double d;

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
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(a);
        fun(b);
        fun(c);
        fun(d);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("a", a);
        fun("b", b);
        fun("c", c);
        fun("d", d);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("a", a);
        fun("b", b);
        fun("c", c);
        fun("d", d);
    }

    bool operator==(const ::ara::adsfi::LaneLineFitParam& t) const
    {
        return (fabs(static_cast<double>(a - t.a)) < DBL_EPSILON) && (fabs(static_cast<double>(b - t.b)) < DBL_EPSILON) && (fabs(static_cast<double>(c - t.c)) < DBL_EPSILON) && (fabs(static_cast<double>(d - t.d)) < DBL_EPSILON);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_LANELINEFITPARAM_H