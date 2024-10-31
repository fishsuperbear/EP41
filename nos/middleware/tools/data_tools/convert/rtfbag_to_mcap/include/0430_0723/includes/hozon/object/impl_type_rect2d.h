/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_RECT2D_H
#define HOZON_OBJECT_IMPL_TYPE_RECT2D_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_point2d.h"

namespace hozon {
namespace object {
struct Rect2D {
    ::hozon::composite::Point2D Center;
    ::hozon::composite::Point2D SizeLW;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Center);
        fun(SizeLW);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Center);
        fun(SizeLW);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Center", Center);
        fun("SizeLW", SizeLW);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Center", Center);
        fun("SizeLW", SizeLW);
    }

    bool operator==(const ::hozon::object::Rect2D& t) const
    {
        return (Center == t.Center) && (SizeLW == t.SizeLW);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_RECT2D_H
