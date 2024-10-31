/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PARKINGLOT_IMPL_TYPE_LOTPSPOINT_H
#define HOZON_PARKINGLOT_IMPL_TYPE_LOTPSPOINT_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_point.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace parkinglot {
struct LotPsPoint {
    ::hozon::composite::Point point;
    ::UInt8 position;
    ::UInt8 quality;
    ::hozon::composite::Point point_vehicle;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(point);
        fun(position);
        fun(quality);
        fun(point_vehicle);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(point);
        fun(position);
        fun(quality);
        fun(point_vehicle);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("point", point);
        fun("position", position);
        fun("quality", quality);
        fun("point_vehicle", point_vehicle);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("point", point);
        fun("position", position);
        fun("quality", quality);
        fun("point_vehicle", point_vehicle);
    }

    bool operator==(const ::hozon::parkinglot::LotPsPoint& t) const
    {
        return (point == t.point) && (position == t.position) && (quality == t.quality) && (point_vehicle == t.point_vehicle);
    }
};
} // namespace parkinglot
} // namespace hozon


#endif // HOZON_PARKINGLOT_IMPL_TYPE_LOTPSPOINT_H
