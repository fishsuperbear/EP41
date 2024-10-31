/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_RTSDISINFO_H
#define HOZON_EQ3_IMPL_TYPE_RTSDISINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace eq3 {
struct RtsDisInfo {
    float rtsa_l_long_rel;
    float rtsa_l_lat_rel;
    ::uint8_t rtsa_trackid;
    ::uint8_t rtsa_status;
    float rtsa_a_long_obj;
    ::uint8_t rtsa_detection_sensor;
    ::uint8_t rtsa_message_counter;
    double rtsb_v_long_obj;
    double rtsb_v_lat_obj;
    float rtsb_xohp_etsel_cmbb_fcw;
    float rtsb_xolc_etsel_cmbb_fcw;
    float rtsb_a_lat_obj;
    ::uint8_t rtsb_movement;
    ::uint8_t rtsb_message_counter;
    float rtsc_width;
    ::uint8_t rtsc_mc_position_confidence;
    ::uint8_t rtsc_mc_age;
    ::uint8_t rtsc_mc_stage_age;
    ::uint8_t rtsc_mc_object_class;
    ::uint8_t rtsc_left_corner_long;
    float rtsc_vistrkid;
    ::uint8_t rtsc_far_left_right;
    ::uint8_t rtsc_message_counter;
    float rtsd_left_corner_lat;
    float rtsd_right_corner_long;
    float rtsd_right_corner_lat;
    float rtsd_far_corner_long;
    float rtsd_far_corner_lat;
    ::uint8_t rtsd_message_counter;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(rtsa_l_long_rel);
        fun(rtsa_l_lat_rel);
        fun(rtsa_trackid);
        fun(rtsa_status);
        fun(rtsa_a_long_obj);
        fun(rtsa_detection_sensor);
        fun(rtsa_message_counter);
        fun(rtsb_v_long_obj);
        fun(rtsb_v_lat_obj);
        fun(rtsb_xohp_etsel_cmbb_fcw);
        fun(rtsb_xolc_etsel_cmbb_fcw);
        fun(rtsb_a_lat_obj);
        fun(rtsb_movement);
        fun(rtsb_message_counter);
        fun(rtsc_width);
        fun(rtsc_mc_position_confidence);
        fun(rtsc_mc_age);
        fun(rtsc_mc_stage_age);
        fun(rtsc_mc_object_class);
        fun(rtsc_left_corner_long);
        fun(rtsc_vistrkid);
        fun(rtsc_far_left_right);
        fun(rtsc_message_counter);
        fun(rtsd_left_corner_lat);
        fun(rtsd_right_corner_long);
        fun(rtsd_right_corner_lat);
        fun(rtsd_far_corner_long);
        fun(rtsd_far_corner_lat);
        fun(rtsd_message_counter);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(rtsa_l_long_rel);
        fun(rtsa_l_lat_rel);
        fun(rtsa_trackid);
        fun(rtsa_status);
        fun(rtsa_a_long_obj);
        fun(rtsa_detection_sensor);
        fun(rtsa_message_counter);
        fun(rtsb_v_long_obj);
        fun(rtsb_v_lat_obj);
        fun(rtsb_xohp_etsel_cmbb_fcw);
        fun(rtsb_xolc_etsel_cmbb_fcw);
        fun(rtsb_a_lat_obj);
        fun(rtsb_movement);
        fun(rtsb_message_counter);
        fun(rtsc_width);
        fun(rtsc_mc_position_confidence);
        fun(rtsc_mc_age);
        fun(rtsc_mc_stage_age);
        fun(rtsc_mc_object_class);
        fun(rtsc_left_corner_long);
        fun(rtsc_vistrkid);
        fun(rtsc_far_left_right);
        fun(rtsc_message_counter);
        fun(rtsd_left_corner_lat);
        fun(rtsd_right_corner_long);
        fun(rtsd_right_corner_lat);
        fun(rtsd_far_corner_long);
        fun(rtsd_far_corner_lat);
        fun(rtsd_message_counter);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("rtsa_l_long_rel", rtsa_l_long_rel);
        fun("rtsa_l_lat_rel", rtsa_l_lat_rel);
        fun("rtsa_trackid", rtsa_trackid);
        fun("rtsa_status", rtsa_status);
        fun("rtsa_a_long_obj", rtsa_a_long_obj);
        fun("rtsa_detection_sensor", rtsa_detection_sensor);
        fun("rtsa_message_counter", rtsa_message_counter);
        fun("rtsb_v_long_obj", rtsb_v_long_obj);
        fun("rtsb_v_lat_obj", rtsb_v_lat_obj);
        fun("rtsb_xohp_etsel_cmbb_fcw", rtsb_xohp_etsel_cmbb_fcw);
        fun("rtsb_xolc_etsel_cmbb_fcw", rtsb_xolc_etsel_cmbb_fcw);
        fun("rtsb_a_lat_obj", rtsb_a_lat_obj);
        fun("rtsb_movement", rtsb_movement);
        fun("rtsb_message_counter", rtsb_message_counter);
        fun("rtsc_width", rtsc_width);
        fun("rtsc_mc_position_confidence", rtsc_mc_position_confidence);
        fun("rtsc_mc_age", rtsc_mc_age);
        fun("rtsc_mc_stage_age", rtsc_mc_stage_age);
        fun("rtsc_mc_object_class", rtsc_mc_object_class);
        fun("rtsc_left_corner_long", rtsc_left_corner_long);
        fun("rtsc_vistrkid", rtsc_vistrkid);
        fun("rtsc_far_left_right", rtsc_far_left_right);
        fun("rtsc_message_counter", rtsc_message_counter);
        fun("rtsd_left_corner_lat", rtsd_left_corner_lat);
        fun("rtsd_right_corner_long", rtsd_right_corner_long);
        fun("rtsd_right_corner_lat", rtsd_right_corner_lat);
        fun("rtsd_far_corner_long", rtsd_far_corner_long);
        fun("rtsd_far_corner_lat", rtsd_far_corner_lat);
        fun("rtsd_message_counter", rtsd_message_counter);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("rtsa_l_long_rel", rtsa_l_long_rel);
        fun("rtsa_l_lat_rel", rtsa_l_lat_rel);
        fun("rtsa_trackid", rtsa_trackid);
        fun("rtsa_status", rtsa_status);
        fun("rtsa_a_long_obj", rtsa_a_long_obj);
        fun("rtsa_detection_sensor", rtsa_detection_sensor);
        fun("rtsa_message_counter", rtsa_message_counter);
        fun("rtsb_v_long_obj", rtsb_v_long_obj);
        fun("rtsb_v_lat_obj", rtsb_v_lat_obj);
        fun("rtsb_xohp_etsel_cmbb_fcw", rtsb_xohp_etsel_cmbb_fcw);
        fun("rtsb_xolc_etsel_cmbb_fcw", rtsb_xolc_etsel_cmbb_fcw);
        fun("rtsb_a_lat_obj", rtsb_a_lat_obj);
        fun("rtsb_movement", rtsb_movement);
        fun("rtsb_message_counter", rtsb_message_counter);
        fun("rtsc_width", rtsc_width);
        fun("rtsc_mc_position_confidence", rtsc_mc_position_confidence);
        fun("rtsc_mc_age", rtsc_mc_age);
        fun("rtsc_mc_stage_age", rtsc_mc_stage_age);
        fun("rtsc_mc_object_class", rtsc_mc_object_class);
        fun("rtsc_left_corner_long", rtsc_left_corner_long);
        fun("rtsc_vistrkid", rtsc_vistrkid);
        fun("rtsc_far_left_right", rtsc_far_left_right);
        fun("rtsc_message_counter", rtsc_message_counter);
        fun("rtsd_left_corner_lat", rtsd_left_corner_lat);
        fun("rtsd_right_corner_long", rtsd_right_corner_long);
        fun("rtsd_right_corner_lat", rtsd_right_corner_lat);
        fun("rtsd_far_corner_long", rtsd_far_corner_long);
        fun("rtsd_far_corner_lat", rtsd_far_corner_lat);
        fun("rtsd_message_counter", rtsd_message_counter);
    }

    bool operator==(const ::hozon::eq3::RtsDisInfo& t) const
    {
        return (fabs(static_cast<double>(rtsa_l_long_rel - t.rtsa_l_long_rel)) < DBL_EPSILON) && (fabs(static_cast<double>(rtsa_l_lat_rel - t.rtsa_l_lat_rel)) < DBL_EPSILON) && (rtsa_trackid == t.rtsa_trackid) && (rtsa_status == t.rtsa_status) && (fabs(static_cast<double>(rtsa_a_long_obj - t.rtsa_a_long_obj)) < DBL_EPSILON) && (rtsa_detection_sensor == t.rtsa_detection_sensor) && (rtsa_message_counter == t.rtsa_message_counter) && (fabs(static_cast<double>(rtsb_v_long_obj - t.rtsb_v_long_obj)) < DBL_EPSILON) && (fabs(static_cast<double>(rtsb_v_lat_obj - t.rtsb_v_lat_obj)) < DBL_EPSILON) && (fabs(static_cast<double>(rtsb_xohp_etsel_cmbb_fcw - t.rtsb_xohp_etsel_cmbb_fcw)) < DBL_EPSILON) && (fabs(static_cast<double>(rtsb_xolc_etsel_cmbb_fcw - t.rtsb_xolc_etsel_cmbb_fcw)) < DBL_EPSILON) && (fabs(static_cast<double>(rtsb_a_lat_obj - t.rtsb_a_lat_obj)) < DBL_EPSILON) && (rtsb_movement == t.rtsb_movement) && (rtsb_message_counter == t.rtsb_message_counter) && (fabs(static_cast<double>(rtsc_width - t.rtsc_width)) < DBL_EPSILON) && (rtsc_mc_position_confidence == t.rtsc_mc_position_confidence) && (rtsc_mc_age == t.rtsc_mc_age) && (rtsc_mc_stage_age == t.rtsc_mc_stage_age) && (rtsc_mc_object_class == t.rtsc_mc_object_class) && (rtsc_left_corner_long == t.rtsc_left_corner_long) && (fabs(static_cast<double>(rtsc_vistrkid - t.rtsc_vistrkid)) < DBL_EPSILON) && (rtsc_far_left_right == t.rtsc_far_left_right) && (rtsc_message_counter == t.rtsc_message_counter) && (fabs(static_cast<double>(rtsd_left_corner_lat - t.rtsd_left_corner_lat)) < DBL_EPSILON) && (fabs(static_cast<double>(rtsd_right_corner_long - t.rtsd_right_corner_long)) < DBL_EPSILON) && (fabs(static_cast<double>(rtsd_right_corner_lat - t.rtsd_right_corner_lat)) < DBL_EPSILON) && (fabs(static_cast<double>(rtsd_far_corner_long - t.rtsd_far_corner_long)) < DBL_EPSILON) && (fabs(static_cast<double>(rtsd_far_corner_lat - t.rtsd_far_corner_lat)) < DBL_EPSILON) && (rtsd_message_counter == t.rtsd_message_counter);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_RTSDISINFO_H
