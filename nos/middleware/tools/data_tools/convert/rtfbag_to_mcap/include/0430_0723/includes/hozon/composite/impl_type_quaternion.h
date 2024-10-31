/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_QUATERNION_H
#define HOZON_COMPOSITE_IMPL_TYPE_QUATERNION_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"

namespace hozon {
namespace composite {
struct Quaternion {
    ::Float x;
    ::Float y;
    ::Float z;
    ::Float w;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(x);
        fun(y);
        fun(z);
        fun(w);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(z);
        fun(w);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("w", w);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("w", w);
    }

    bool operator==(const ::hozon::composite::Quaternion& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(w - t.w)) < DBL_EPSILON);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_QUATERNION_H
