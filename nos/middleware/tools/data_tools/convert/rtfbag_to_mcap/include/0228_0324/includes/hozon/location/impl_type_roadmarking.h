/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_ROADMARKING_H
#define HOZON_LOCATION_IMPL_TYPE_ROADMARKING_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_point3d_double.h"
#include "hozon/composite/impl_type_point3ddoublevec.h"

namespace hozon {
namespace location {
struct RoadMarking {
    ::UInt8 type;
    ::hozon::composite::Point3D_double position;
    ::hozon::composite::Point3DDoubleVec boundingBox;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(type);
        fun(position);
        fun(boundingBox);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(type);
        fun(position);
        fun(boundingBox);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("type", type);
        fun("position", position);
        fun("boundingBox", boundingBox);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("type", type);
        fun("position", position);
        fun("boundingBox", boundingBox);
    }

    bool operator==(const ::hozon::location::RoadMarking& t) const
    {
        return (type == t.type) && (position == t.position) && (boundingBox == t.boundingBox);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_ROADMARKING_H
