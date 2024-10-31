/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_VIRTUALWALL_H
#define IMPL_TYPE_VIRTUALWALL_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_int32.h"

struct VirtualWall {
    ::Float target_speed;
    ::Float target_distance;
    ::Float desire_follow_seconds;
    ::Int32 id;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(target_speed);
        fun(target_distance);
        fun(desire_follow_seconds);
        fun(id);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(target_speed);
        fun(target_distance);
        fun(desire_follow_seconds);
        fun(id);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("target_speed", target_speed);
        fun("target_distance", target_distance);
        fun("desire_follow_seconds", desire_follow_seconds);
        fun("id", id);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("target_speed", target_speed);
        fun("target_distance", target_distance);
        fun("desire_follow_seconds", desire_follow_seconds);
        fun("id", id);
    }

    bool operator==(const ::VirtualWall& t) const
    {
        return (fabs(static_cast<double>(target_speed - t.target_speed)) < DBL_EPSILON) && (fabs(static_cast<double>(target_distance - t.target_distance)) < DBL_EPSILON) && (fabs(static_cast<double>(desire_follow_seconds - t.desire_follow_seconds)) < DBL_EPSILON) && (id == t.id);
    }
};


#endif // IMPL_TYPE_VIRTUALWALL_H
