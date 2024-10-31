/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_POINT2I_H
#define HOZON_COMPOSITE_IMPL_TYPE_POINT2I_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"

namespace hozon {
namespace composite {
struct Point2I {
    ::Int32 x;
    ::Int32 y;

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

    bool operator==(const ::hozon::composite::Point2I& t) const
    {
        return (x == t.x) && (y == t.y);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_POINT2I_H
