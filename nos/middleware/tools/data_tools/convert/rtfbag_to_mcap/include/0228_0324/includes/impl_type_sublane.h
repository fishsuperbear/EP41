/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_SUBLANE_H
#define IMPL_TYPE_SUBLANE_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"
#include "impl_type_uint8.h"
#include "impl_type_pose2d.h"

struct SubLane {
    ::Int32 lane_id;
    ::UInt8 priority;
    ::Pose2D cross_point;
    ::Pose2D stop_point;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(lane_id);
        fun(priority);
        fun(cross_point);
        fun(stop_point);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(lane_id);
        fun(priority);
        fun(cross_point);
        fun(stop_point);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("lane_id", lane_id);
        fun("priority", priority);
        fun("cross_point", cross_point);
        fun("stop_point", stop_point);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("lane_id", lane_id);
        fun("priority", priority);
        fun("cross_point", cross_point);
        fun("stop_point", stop_point);
    }

    bool operator==(const ::SubLane& t) const
    {
        return (lane_id == t.lane_id) && (priority == t.priority) && (cross_point == t.cross_point) && (stop_point == t.stop_point);
    }
};


#endif // IMPL_TYPE_SUBLANE_H
