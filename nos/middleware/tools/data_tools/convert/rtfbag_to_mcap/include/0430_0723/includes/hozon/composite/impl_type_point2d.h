/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_POINT2D_H
#define HOZON_COMPOSITE_IMPL_TYPE_POINT2D_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"

namespace hozon {
namespace composite {
struct Point2D {
    ::Float x;
    ::Float y;

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
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
    }

    bool operator==(const ::hozon::composite::Point2D& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_POINT2D_H
