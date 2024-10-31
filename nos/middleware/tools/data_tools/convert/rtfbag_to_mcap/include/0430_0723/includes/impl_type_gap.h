/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_GAP_H
#define IMPL_TYPE_GAP_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_int32.h"

struct Gap {
    ::Float speed_front;
    ::Float speed_back;
    ::Float length;
    ::Float distance;
    ::Int32 lane_id;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(speed_front);
        fun(speed_back);
        fun(length);
        fun(distance);
        fun(lane_id);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(speed_front);
        fun(speed_back);
        fun(length);
        fun(distance);
        fun(lane_id);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("speed_front", speed_front);
        fun("speed_back", speed_back);
        fun("length", length);
        fun("distance", distance);
        fun("lane_id", lane_id);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("speed_front", speed_front);
        fun("speed_back", speed_back);
        fun("length", length);
        fun("distance", distance);
        fun("lane_id", lane_id);
    }

    bool operator==(const ::Gap& t) const
    {
        return (fabs(static_cast<double>(speed_front - t.speed_front)) < DBL_EPSILON) && (fabs(static_cast<double>(speed_back - t.speed_back)) < DBL_EPSILON) && (fabs(static_cast<double>(length - t.length)) < DBL_EPSILON) && (fabs(static_cast<double>(distance - t.distance)) < DBL_EPSILON) && (lane_id == t.lane_id);
    }
};


#endif // IMPL_TYPE_GAP_H
