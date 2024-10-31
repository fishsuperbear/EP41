/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STCAMERA_IMPL_TYPE_ABDINFO_H
#define HOZON_STCAMERA_IMPL_TYPE_ABDINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace stcamera {
struct ABDInfo {
    ::Float pitch_angle_online;
    ::Float roll_angle_online;
    ::UInt8 ac_stage;
    ::UInt8 ego_lane_right_first_boundary;
    ::UInt8 ego_lane_right_second_boundary;
    ::UInt8 ego_lane_right_third_boundary;
    ::UInt8 ego_lane_left_first_boundary;
    ::UInt8 ego_lane_left_second_boundary;
    ::UInt8 ego_lane_left_third_boundary;
    ::UInt8 parallel_model;
    ::UInt8 lane_change;
    ::UInt8 construction_site;
    ::UInt8 overall_confidence;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(pitch_angle_online);
        fun(roll_angle_online);
        fun(ac_stage);
        fun(ego_lane_right_first_boundary);
        fun(ego_lane_right_second_boundary);
        fun(ego_lane_right_third_boundary);
        fun(ego_lane_left_first_boundary);
        fun(ego_lane_left_second_boundary);
        fun(ego_lane_left_third_boundary);
        fun(parallel_model);
        fun(lane_change);
        fun(construction_site);
        fun(overall_confidence);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(pitch_angle_online);
        fun(roll_angle_online);
        fun(ac_stage);
        fun(ego_lane_right_first_boundary);
        fun(ego_lane_right_second_boundary);
        fun(ego_lane_right_third_boundary);
        fun(ego_lane_left_first_boundary);
        fun(ego_lane_left_second_boundary);
        fun(ego_lane_left_third_boundary);
        fun(parallel_model);
        fun(lane_change);
        fun(construction_site);
        fun(overall_confidence);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("pitch_angle_online", pitch_angle_online);
        fun("roll_angle_online", roll_angle_online);
        fun("ac_stage", ac_stage);
        fun("ego_lane_right_first_boundary", ego_lane_right_first_boundary);
        fun("ego_lane_right_second_boundary", ego_lane_right_second_boundary);
        fun("ego_lane_right_third_boundary", ego_lane_right_third_boundary);
        fun("ego_lane_left_first_boundary", ego_lane_left_first_boundary);
        fun("ego_lane_left_second_boundary", ego_lane_left_second_boundary);
        fun("ego_lane_left_third_boundary", ego_lane_left_third_boundary);
        fun("parallel_model", parallel_model);
        fun("lane_change", lane_change);
        fun("construction_site", construction_site);
        fun("overall_confidence", overall_confidence);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("pitch_angle_online", pitch_angle_online);
        fun("roll_angle_online", roll_angle_online);
        fun("ac_stage", ac_stage);
        fun("ego_lane_right_first_boundary", ego_lane_right_first_boundary);
        fun("ego_lane_right_second_boundary", ego_lane_right_second_boundary);
        fun("ego_lane_right_third_boundary", ego_lane_right_third_boundary);
        fun("ego_lane_left_first_boundary", ego_lane_left_first_boundary);
        fun("ego_lane_left_second_boundary", ego_lane_left_second_boundary);
        fun("ego_lane_left_third_boundary", ego_lane_left_third_boundary);
        fun("parallel_model", parallel_model);
        fun("lane_change", lane_change);
        fun("construction_site", construction_site);
        fun("overall_confidence", overall_confidence);
    }

    bool operator==(const ::hozon::stcamera::ABDInfo& t) const
    {
        return (fabs(static_cast<double>(pitch_angle_online - t.pitch_angle_online)) < DBL_EPSILON) && (fabs(static_cast<double>(roll_angle_online - t.roll_angle_online)) < DBL_EPSILON) && (ac_stage == t.ac_stage) && (ego_lane_right_first_boundary == t.ego_lane_right_first_boundary) && (ego_lane_right_second_boundary == t.ego_lane_right_second_boundary) && (ego_lane_right_third_boundary == t.ego_lane_right_third_boundary) && (ego_lane_left_first_boundary == t.ego_lane_left_first_boundary) && (ego_lane_left_second_boundary == t.ego_lane_left_second_boundary) && (ego_lane_left_third_boundary == t.ego_lane_left_third_boundary) && (parallel_model == t.parallel_model) && (lane_change == t.lane_change) && (construction_site == t.construction_site) && (overall_confidence == t.overall_confidence);
    }
};
} // namespace stcamera
} // namespace hozon


#endif // HOZON_STCAMERA_IMPL_TYPE_ABDINFO_H
