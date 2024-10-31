/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_LANE_H
#define IMPL_TYPE_LANE_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"
#include "impl_type_float.h"
#include "impl_type_uint8.h"
#include "impl_type_pose2darray.h"
#include "impl_type_sublanearray.h"

struct Lane {
    ::Int32 id;
    ::Float speed_limit;
    ::UInt8 left_changable;
    ::Float left_changable_distance;
    ::Float right_changable_distance;
    ::UInt8 right_changable;
    ::Pose2DArray road_points;
    ::Pose2DArray left_edge;
    ::Pose2DArray right_edge;
    ::Int32 next_lane_id;
    ::Int32 previous_lane_id;
    ::Int32 left_lane_id;
    ::Int32 right_lane_id;
    ::UInt8 traffic_light_state;
    ::Float traffic_light_distance;
    ::Float stop_sign_distance;
    ::Float parking_pole_distance;
    ::Float cross_walk_distance;
    ::UInt8 type;
    ::SubLaneArray sub_lanes;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(speed_limit);
        fun(left_changable);
        fun(left_changable_distance);
        fun(right_changable_distance);
        fun(right_changable);
        fun(road_points);
        fun(left_edge);
        fun(right_edge);
        fun(next_lane_id);
        fun(previous_lane_id);
        fun(left_lane_id);
        fun(right_lane_id);
        fun(traffic_light_state);
        fun(traffic_light_distance);
        fun(stop_sign_distance);
        fun(parking_pole_distance);
        fun(cross_walk_distance);
        fun(type);
        fun(sub_lanes);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(speed_limit);
        fun(left_changable);
        fun(left_changable_distance);
        fun(right_changable_distance);
        fun(right_changable);
        fun(road_points);
        fun(left_edge);
        fun(right_edge);
        fun(next_lane_id);
        fun(previous_lane_id);
        fun(left_lane_id);
        fun(right_lane_id);
        fun(traffic_light_state);
        fun(traffic_light_distance);
        fun(stop_sign_distance);
        fun(parking_pole_distance);
        fun(cross_walk_distance);
        fun(type);
        fun(sub_lanes);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("speed_limit", speed_limit);
        fun("left_changable", left_changable);
        fun("left_changable_distance", left_changable_distance);
        fun("right_changable_distance", right_changable_distance);
        fun("right_changable", right_changable);
        fun("road_points", road_points);
        fun("left_edge", left_edge);
        fun("right_edge", right_edge);
        fun("next_lane_id", next_lane_id);
        fun("previous_lane_id", previous_lane_id);
        fun("left_lane_id", left_lane_id);
        fun("right_lane_id", right_lane_id);
        fun("traffic_light_state", traffic_light_state);
        fun("traffic_light_distance", traffic_light_distance);
        fun("stop_sign_distance", stop_sign_distance);
        fun("parking_pole_distance", parking_pole_distance);
        fun("cross_walk_distance", cross_walk_distance);
        fun("type", type);
        fun("sub_lanes", sub_lanes);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("speed_limit", speed_limit);
        fun("left_changable", left_changable);
        fun("left_changable_distance", left_changable_distance);
        fun("right_changable_distance", right_changable_distance);
        fun("right_changable", right_changable);
        fun("road_points", road_points);
        fun("left_edge", left_edge);
        fun("right_edge", right_edge);
        fun("next_lane_id", next_lane_id);
        fun("previous_lane_id", previous_lane_id);
        fun("left_lane_id", left_lane_id);
        fun("right_lane_id", right_lane_id);
        fun("traffic_light_state", traffic_light_state);
        fun("traffic_light_distance", traffic_light_distance);
        fun("stop_sign_distance", stop_sign_distance);
        fun("parking_pole_distance", parking_pole_distance);
        fun("cross_walk_distance", cross_walk_distance);
        fun("type", type);
        fun("sub_lanes", sub_lanes);
    }

    bool operator==(const ::Lane& t) const
    {
        return (id == t.id) && (fabs(static_cast<double>(speed_limit - t.speed_limit)) < DBL_EPSILON) && (left_changable == t.left_changable) && (fabs(static_cast<double>(left_changable_distance - t.left_changable_distance)) < DBL_EPSILON) && (fabs(static_cast<double>(right_changable_distance - t.right_changable_distance)) < DBL_EPSILON) && (right_changable == t.right_changable) && (road_points == t.road_points) && (left_edge == t.left_edge) && (right_edge == t.right_edge) && (next_lane_id == t.next_lane_id) && (previous_lane_id == t.previous_lane_id) && (left_lane_id == t.left_lane_id) && (right_lane_id == t.right_lane_id) && (traffic_light_state == t.traffic_light_state) && (fabs(static_cast<double>(traffic_light_distance - t.traffic_light_distance)) < DBL_EPSILON) && (fabs(static_cast<double>(stop_sign_distance - t.stop_sign_distance)) < DBL_EPSILON) && (fabs(static_cast<double>(parking_pole_distance - t.parking_pole_distance)) < DBL_EPSILON) && (fabs(static_cast<double>(cross_walk_distance - t.cross_walk_distance)) < DBL_EPSILON) && (type == t.type) && (sub_lanes == t.sub_lanes);
    }
};


#endif // IMPL_TYPE_LANE_H
