/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_TRAFFICSIGN_H
#define HOZON_LOCATION_IMPL_TYPE_TRAFFICSIGN_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_point3ddoublevec.h"

namespace hozon {
namespace location {
struct TrafficSign {
    ::UInt8 shape;
    ::hozon::composite::Point3DDoubleVec position;
    ::hozon::composite::Point3DDoubleVec boundingBox3;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(shape);
        fun(position);
        fun(boundingBox3);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(shape);
        fun(position);
        fun(boundingBox3);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("shape", shape);
        fun("position", position);
        fun("boundingBox3", boundingBox3);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("shape", shape);
        fun("position", position);
        fun("boundingBox3", boundingBox3);
    }

    bool operator==(const ::hozon::location::TrafficSign& t) const
    {
        return (shape == t.shape) && (position == t.position) && (boundingBox3 == t.boundingBox3);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_TRAFFICSIGN_H
