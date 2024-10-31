/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_POSE_H
#define IMPL_TYPE_POSE_H
#include <cfloat>
#include <cmath>
#include "impl_type_point.h"
#include "impl_type_quaternion.h"

struct Pose {
    ::Point position;
    ::Quaternion orientation;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(position);
        fun(orientation);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(position);
        fun(orientation);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("position", position);
        fun("orientation", orientation);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("position", position);
        fun("orientation", orientation);
    }

    bool operator==(const ::Pose& t) const
    {
        return (position == t.position) && (orientation == t.orientation);
    }
};


#endif // IMPL_TYPE_POSE_H
