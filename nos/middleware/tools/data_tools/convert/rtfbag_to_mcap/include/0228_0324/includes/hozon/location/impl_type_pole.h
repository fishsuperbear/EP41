/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_POLE_H
#define HOZON_LOCATION_IMPL_TYPE_POLE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "hozon/composite/impl_type_point3d_double.h"

namespace hozon {
namespace location {
struct Pole {
    ::UInt8 type;
    ::Float relativeHeight;
    ::hozon::composite::Point3D_double position;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(type);
        fun(relativeHeight);
        fun(position);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(type);
        fun(relativeHeight);
        fun(position);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("type", type);
        fun("relativeHeight", relativeHeight);
        fun("position", position);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("type", type);
        fun("relativeHeight", relativeHeight);
        fun("position", position);
    }

    bool operator==(const ::hozon::location::Pole& t) const
    {
        return (type == t.type) && (fabs(static_cast<double>(relativeHeight - t.relativeHeight)) < DBL_EPSILON) && (position == t.position);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_POLE_H
