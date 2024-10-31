/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_PEDESTRIANINFO_H
#define HOZON_EQ3_IMPL_TYPE_PEDESTRIANINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace eq3 {
struct PedestrianInfo {
    ::uint8_t pedestrian_pa_trackid;
    double pedestrian_pa_l_long_rel;
    double pedestrian_pa_v_long_obj;
    double pedestrian_pa_a_long_obj;
    double pedestrian_pa_l_lat_rel;
    double pedestrian_pb_v_lat_obj;
    double pedestrian_pb_a_lat_obj;
    ::uint8_t pedestrian_pb_cmbb_primary_confidence;
    ::uint8_t pedestrian_pb_fcw_confidence;
    ::uint8_t pedestrian_pb_status;
    ::uint8_t pedestrian_pb_age;
    ::uint8_t pedestrian_pc_stage_age;
    ::uint8_t pedestrian_pc_object_class;
    ::uint8_t pedestrian_pc_vistrkid;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(pedestrian_pa_trackid);
        fun(pedestrian_pa_l_long_rel);
        fun(pedestrian_pa_v_long_obj);
        fun(pedestrian_pa_a_long_obj);
        fun(pedestrian_pa_l_lat_rel);
        fun(pedestrian_pb_v_lat_obj);
        fun(pedestrian_pb_a_lat_obj);
        fun(pedestrian_pb_cmbb_primary_confidence);
        fun(pedestrian_pb_fcw_confidence);
        fun(pedestrian_pb_status);
        fun(pedestrian_pb_age);
        fun(pedestrian_pc_stage_age);
        fun(pedestrian_pc_object_class);
        fun(pedestrian_pc_vistrkid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(pedestrian_pa_trackid);
        fun(pedestrian_pa_l_long_rel);
        fun(pedestrian_pa_v_long_obj);
        fun(pedestrian_pa_a_long_obj);
        fun(pedestrian_pa_l_lat_rel);
        fun(pedestrian_pb_v_lat_obj);
        fun(pedestrian_pb_a_lat_obj);
        fun(pedestrian_pb_cmbb_primary_confidence);
        fun(pedestrian_pb_fcw_confidence);
        fun(pedestrian_pb_status);
        fun(pedestrian_pb_age);
        fun(pedestrian_pc_stage_age);
        fun(pedestrian_pc_object_class);
        fun(pedestrian_pc_vistrkid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("pedestrian_pa_trackid", pedestrian_pa_trackid);
        fun("pedestrian_pa_l_long_rel", pedestrian_pa_l_long_rel);
        fun("pedestrian_pa_v_long_obj", pedestrian_pa_v_long_obj);
        fun("pedestrian_pa_a_long_obj", pedestrian_pa_a_long_obj);
        fun("pedestrian_pa_l_lat_rel", pedestrian_pa_l_lat_rel);
        fun("pedestrian_pb_v_lat_obj", pedestrian_pb_v_lat_obj);
        fun("pedestrian_pb_a_lat_obj", pedestrian_pb_a_lat_obj);
        fun("pedestrian_pb_cmbb_primary_confidence", pedestrian_pb_cmbb_primary_confidence);
        fun("pedestrian_pb_fcw_confidence", pedestrian_pb_fcw_confidence);
        fun("pedestrian_pb_status", pedestrian_pb_status);
        fun("pedestrian_pb_age", pedestrian_pb_age);
        fun("pedestrian_pc_stage_age", pedestrian_pc_stage_age);
        fun("pedestrian_pc_object_class", pedestrian_pc_object_class);
        fun("pedestrian_pc_vistrkid", pedestrian_pc_vistrkid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("pedestrian_pa_trackid", pedestrian_pa_trackid);
        fun("pedestrian_pa_l_long_rel", pedestrian_pa_l_long_rel);
        fun("pedestrian_pa_v_long_obj", pedestrian_pa_v_long_obj);
        fun("pedestrian_pa_a_long_obj", pedestrian_pa_a_long_obj);
        fun("pedestrian_pa_l_lat_rel", pedestrian_pa_l_lat_rel);
        fun("pedestrian_pb_v_lat_obj", pedestrian_pb_v_lat_obj);
        fun("pedestrian_pb_a_lat_obj", pedestrian_pb_a_lat_obj);
        fun("pedestrian_pb_cmbb_primary_confidence", pedestrian_pb_cmbb_primary_confidence);
        fun("pedestrian_pb_fcw_confidence", pedestrian_pb_fcw_confidence);
        fun("pedestrian_pb_status", pedestrian_pb_status);
        fun("pedestrian_pb_age", pedestrian_pb_age);
        fun("pedestrian_pc_stage_age", pedestrian_pc_stage_age);
        fun("pedestrian_pc_object_class", pedestrian_pc_object_class);
        fun("pedestrian_pc_vistrkid", pedestrian_pc_vistrkid);
    }

    bool operator==(const ::hozon::eq3::PedestrianInfo& t) const
    {
        return (pedestrian_pa_trackid == t.pedestrian_pa_trackid) && (fabs(static_cast<double>(pedestrian_pa_l_long_rel - t.pedestrian_pa_l_long_rel)) < DBL_EPSILON) && (fabs(static_cast<double>(pedestrian_pa_v_long_obj - t.pedestrian_pa_v_long_obj)) < DBL_EPSILON) && (fabs(static_cast<double>(pedestrian_pa_a_long_obj - t.pedestrian_pa_a_long_obj)) < DBL_EPSILON) && (fabs(static_cast<double>(pedestrian_pa_l_lat_rel - t.pedestrian_pa_l_lat_rel)) < DBL_EPSILON) && (fabs(static_cast<double>(pedestrian_pb_v_lat_obj - t.pedestrian_pb_v_lat_obj)) < DBL_EPSILON) && (fabs(static_cast<double>(pedestrian_pb_a_lat_obj - t.pedestrian_pb_a_lat_obj)) < DBL_EPSILON) && (pedestrian_pb_cmbb_primary_confidence == t.pedestrian_pb_cmbb_primary_confidence) && (pedestrian_pb_fcw_confidence == t.pedestrian_pb_fcw_confidence) && (pedestrian_pb_status == t.pedestrian_pb_status) && (pedestrian_pb_age == t.pedestrian_pb_age) && (pedestrian_pc_stage_age == t.pedestrian_pc_stage_age) && (pedestrian_pc_object_class == t.pedestrian_pc_object_class) && (pedestrian_pc_vistrkid == t.pedestrian_pc_vistrkid);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_PEDESTRIANINFO_H
