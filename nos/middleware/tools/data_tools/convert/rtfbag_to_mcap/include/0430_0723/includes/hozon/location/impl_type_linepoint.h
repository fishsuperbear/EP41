/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_LINEPOINT_H
#define HOZON_LOCATION_IMPL_TYPE_LINEPOINT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_point3ddoublevec.h"

namespace hozon {
namespace location {
struct LinePoint {
    ::UInt8 type;
    ::hozon::composite::Point3DDoubleVec linePoint;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(type);
        fun(linePoint);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(type);
        fun(linePoint);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("type", type);
        fun("linePoint", linePoint);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("type", type);
        fun("linePoint", linePoint);
    }

    bool operator==(const ::hozon::location::LinePoint& t) const
    {
        return (type == t.type) && (linePoint == t.linePoint);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_LINEPOINT_H
