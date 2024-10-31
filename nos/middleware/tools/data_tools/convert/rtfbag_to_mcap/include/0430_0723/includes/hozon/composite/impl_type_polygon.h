/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_POLYGON_H
#define HOZON_COMPOSITE_IMPL_TYPE_POLYGON_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_pointarray.h"

namespace hozon {
namespace composite {
struct Polygon {
    ::hozon::composite::PointArray points;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(points);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(points);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("points", points);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("points", points);
    }

    bool operator==(const ::hozon::composite::Polygon& t) const
    {
        return (points == t.points);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_POLYGON_H
