/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_POINTFIELD_H
#define HOZON_SENSORS_IMPL_TYPE_POINTFIELD_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_uint32.h"
#include "impl_type_uint16.h"

namespace hozon {
namespace sensors {
struct PointField {
    ::Float x;
    ::Float y;
    ::Float z;
    ::UInt32 time;
    ::Float distance;
    ::Float pitch;
    ::Float yaw;
    ::UInt16 intensity;
    ::UInt16 ring;
    ::UInt16 block;

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
        fun(time);
        fun(distance);
        fun(pitch);
        fun(yaw);
        fun(intensity);
        fun(ring);
        fun(block);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(z);
        fun(time);
        fun(distance);
        fun(pitch);
        fun(yaw);
        fun(intensity);
        fun(ring);
        fun(block);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("time", time);
        fun("distance", distance);
        fun("pitch", pitch);
        fun("yaw", yaw);
        fun("intensity", intensity);
        fun("ring", ring);
        fun("block", block);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("time", time);
        fun("distance", distance);
        fun("pitch", pitch);
        fun("yaw", yaw);
        fun("intensity", intensity);
        fun("ring", ring);
        fun("block", block);
    }

    bool operator==(const ::hozon::sensors::PointField& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (time == t.time) && (fabs(static_cast<double>(distance - t.distance)) < DBL_EPSILON) && (fabs(static_cast<double>(pitch - t.pitch)) < DBL_EPSILON) && (fabs(static_cast<double>(yaw - t.yaw)) < DBL_EPSILON) && (intensity == t.intensity) && (ring == t.ring) && (block == t.block);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_POINTFIELD_H
