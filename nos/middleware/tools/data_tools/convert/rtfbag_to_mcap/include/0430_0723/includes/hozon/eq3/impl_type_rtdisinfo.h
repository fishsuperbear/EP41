/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_RTDISINFO_H
#define HOZON_EQ3_IMPL_TYPE_RTDISINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace eq3 {
struct RtDisInfo {
    double rta_l_long_rel;
    double rta_v_long_obj;
    double rta_l_lat_rel;
    double rta_v_lat_obj;
    float rta_a_long_obj;
    ::uint8_t rta_detection_sensor;
    ::uint8_t rta_message_counter;
    ::uint8_t rtb_trackid;
    ::uint8_t rtb_status;
    float rtb_xohp_etsel_cmbb_fcw;
    float rtb_xolc_etsel_cmbb_fcw;
    float rtb_a_lat_obj;
    ::uint8_t rtbrtc_width_movement;
    ::uint8_t rtb_message_counter;
    float rtc_width;
    ::uint8_t rtc_mc_position_confidence;
    ::uint8_t rtc_mc_age;
    ::uint8_t rtc_mc_stage_age;
    ::uint8_t rtc_mc_object_class;
    float rtc_left_corner_long;
    ::uint8_t rtc_vistrkid;
    ::uint8_t rtc_farleftright;
    ::uint8_t rtc_message_counter;
    float rtd_left_corner_lat;
    float rtd_right_corner_long;
    float rtd_right_corner_lat;
    float rt1d_far_corner_long;
    float rt1d_far_corner_lat;
    ::uint8_t rtd_message_counter;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(rta_l_long_rel);
        fun(rta_v_long_obj);
        fun(rta_l_lat_rel);
        fun(rta_v_lat_obj);
        fun(rta_a_long_obj);
        fun(rta_detection_sensor);
        fun(rta_message_counter);
        fun(rtb_trackid);
        fun(rtb_status);
        fun(rtb_xohp_etsel_cmbb_fcw);
        fun(rtb_xolc_etsel_cmbb_fcw);
        fun(rtb_a_lat_obj);
        fun(rtbrtc_width_movement);
        fun(rtb_message_counter);
        fun(rtc_width);
        fun(rtc_mc_position_confidence);
        fun(rtc_mc_age);
        fun(rtc_mc_stage_age);
        fun(rtc_mc_object_class);
        fun(rtc_left_corner_long);
        fun(rtc_vistrkid);
        fun(rtc_farleftright);
        fun(rtc_message_counter);
        fun(rtd_left_corner_lat);
        fun(rtd_right_corner_long);
        fun(rtd_right_corner_lat);
        fun(rt1d_far_corner_long);
        fun(rt1d_far_corner_lat);
        fun(rtd_message_counter);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(rta_l_long_rel);
        fun(rta_v_long_obj);
        fun(rta_l_lat_rel);
        fun(rta_v_lat_obj);
        fun(rta_a_long_obj);
        fun(rta_detection_sensor);
        fun(rta_message_counter);
        fun(rtb_trackid);
        fun(rtb_status);
        fun(rtb_xohp_etsel_cmbb_fcw);
        fun(rtb_xolc_etsel_cmbb_fcw);
        fun(rtb_a_lat_obj);
        fun(rtbrtc_width_movement);
        fun(rtb_message_counter);
        fun(rtc_width);
        fun(rtc_mc_position_confidence);
        fun(rtc_mc_age);
        fun(rtc_mc_stage_age);
        fun(rtc_mc_object_class);
        fun(rtc_left_corner_long);
        fun(rtc_vistrkid);
        fun(rtc_farleftright);
        fun(rtc_message_counter);
        fun(rtd_left_corner_lat);
        fun(rtd_right_corner_long);
        fun(rtd_right_corner_lat);
        fun(rt1d_far_corner_long);
        fun(rt1d_far_corner_lat);
        fun(rtd_message_counter);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("rta_l_long_rel", rta_l_long_rel);
        fun("rta_v_long_obj", rta_v_long_obj);
        fun("rta_l_lat_rel", rta_l_lat_rel);
        fun("rta_v_lat_obj", rta_v_lat_obj);
        fun("rta_a_long_obj", rta_a_long_obj);
        fun("rta_detection_sensor", rta_detection_sensor);
        fun("rta_message_counter", rta_message_counter);
        fun("rtb_trackid", rtb_trackid);
        fun("rtb_status", rtb_status);
        fun("rtb_xohp_etsel_cmbb_fcw", rtb_xohp_etsel_cmbb_fcw);
        fun("rtb_xolc_etsel_cmbb_fcw", rtb_xolc_etsel_cmbb_fcw);
        fun("rtb_a_lat_obj", rtb_a_lat_obj);
        fun("rtbrtc_width_movement", rtbrtc_width_movement);
        fun("rtb_message_counter", rtb_message_counter);
        fun("rtc_width", rtc_width);
        fun("rtc_mc_position_confidence", rtc_mc_position_confidence);
        fun("rtc_mc_age", rtc_mc_age);
        fun("rtc_mc_stage_age", rtc_mc_stage_age);
        fun("rtc_mc_object_class", rtc_mc_object_class);
        fun("rtc_left_corner_long", rtc_left_corner_long);
        fun("rtc_vistrkid", rtc_vistrkid);
        fun("rtc_farleftright", rtc_farleftright);
        fun("rtc_message_counter", rtc_message_counter);
        fun("rtd_left_corner_lat", rtd_left_corner_lat);
        fun("rtd_right_corner_long", rtd_right_corner_long);
        fun("rtd_right_corner_lat", rtd_right_corner_lat);
        fun("rt1d_far_corner_long", rt1d_far_corner_long);
        fun("rt1d_far_corner_lat", rt1d_far_corner_lat);
        fun("rtd_message_counter", rtd_message_counter);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("rta_l_long_rel", rta_l_long_rel);
        fun("rta_v_long_obj", rta_v_long_obj);
        fun("rta_l_lat_rel", rta_l_lat_rel);
        fun("rta_v_lat_obj", rta_v_lat_obj);
        fun("rta_a_long_obj", rta_a_long_obj);
        fun("rta_detection_sensor", rta_detection_sensor);
        fun("rta_message_counter", rta_message_counter);
        fun("rtb_trackid", rtb_trackid);
        fun("rtb_status", rtb_status);
        fun("rtb_xohp_etsel_cmbb_fcw", rtb_xohp_etsel_cmbb_fcw);
        fun("rtb_xolc_etsel_cmbb_fcw", rtb_xolc_etsel_cmbb_fcw);
        fun("rtb_a_lat_obj", rtb_a_lat_obj);
        fun("rtbrtc_width_movement", rtbrtc_width_movement);
        fun("rtb_message_counter", rtb_message_counter);
        fun("rtc_width", rtc_width);
        fun("rtc_mc_position_confidence", rtc_mc_position_confidence);
        fun("rtc_mc_age", rtc_mc_age);
        fun("rtc_mc_stage_age", rtc_mc_stage_age);
        fun("rtc_mc_object_class", rtc_mc_object_class);
        fun("rtc_left_corner_long", rtc_left_corner_long);
        fun("rtc_vistrkid", rtc_vistrkid);
        fun("rtc_farleftright", rtc_farleftright);
        fun("rtc_message_counter", rtc_message_counter);
        fun("rtd_left_corner_lat", rtd_left_corner_lat);
        fun("rtd_right_corner_long", rtd_right_corner_long);
        fun("rtd_right_corner_lat", rtd_right_corner_lat);
        fun("rt1d_far_corner_long", rt1d_far_corner_long);
        fun("rt1d_far_corner_lat", rt1d_far_corner_lat);
        fun("rtd_message_counter", rtd_message_counter);
    }

    bool operator==(const ::hozon::eq3::RtDisInfo& t) const
    {
        return (fabs(static_cast<double>(rta_l_long_rel - t.rta_l_long_rel)) < DBL_EPSILON) && (fabs(static_cast<double>(rta_v_long_obj - t.rta_v_long_obj)) < DBL_EPSILON) && (fabs(static_cast<double>(rta_l_lat_rel - t.rta_l_lat_rel)) < DBL_EPSILON) && (fabs(static_cast<double>(rta_v_lat_obj - t.rta_v_lat_obj)) < DBL_EPSILON) && (fabs(static_cast<double>(rta_a_long_obj - t.rta_a_long_obj)) < DBL_EPSILON) && (rta_detection_sensor == t.rta_detection_sensor) && (rta_message_counter == t.rta_message_counter) && (rtb_trackid == t.rtb_trackid) && (rtb_status == t.rtb_status) && (fabs(static_cast<double>(rtb_xohp_etsel_cmbb_fcw - t.rtb_xohp_etsel_cmbb_fcw)) < DBL_EPSILON) && (fabs(static_cast<double>(rtb_xolc_etsel_cmbb_fcw - t.rtb_xolc_etsel_cmbb_fcw)) < DBL_EPSILON) && (fabs(static_cast<double>(rtb_a_lat_obj - t.rtb_a_lat_obj)) < DBL_EPSILON) && (rtbrtc_width_movement == t.rtbrtc_width_movement) && (rtb_message_counter == t.rtb_message_counter) && (fabs(static_cast<double>(rtc_width - t.rtc_width)) < DBL_EPSILON) && (rtc_mc_position_confidence == t.rtc_mc_position_confidence) && (rtc_mc_age == t.rtc_mc_age) && (rtc_mc_stage_age == t.rtc_mc_stage_age) && (rtc_mc_object_class == t.rtc_mc_object_class) && (fabs(static_cast<double>(rtc_left_corner_long - t.rtc_left_corner_long)) < DBL_EPSILON) && (rtc_vistrkid == t.rtc_vistrkid) && (rtc_farleftright == t.rtc_farleftright) && (rtc_message_counter == t.rtc_message_counter) && (fabs(static_cast<double>(rtd_left_corner_lat - t.rtd_left_corner_lat)) < DBL_EPSILON) && (fabs(static_cast<double>(rtd_right_corner_long - t.rtd_right_corner_long)) < DBL_EPSILON) && (fabs(static_cast<double>(rtd_right_corner_lat - t.rtd_right_corner_lat)) < DBL_EPSILON) && (fabs(static_cast<double>(rt1d_far_corner_long - t.rt1d_far_corner_long)) < DBL_EPSILON) && (fabs(static_cast<double>(rt1d_far_corner_lat - t.rt1d_far_corner_lat)) < DBL_EPSILON) && (rtd_message_counter == t.rtd_message_counter);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_RTDISINFO_H
