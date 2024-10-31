/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MBDESTONLINEDEBUG_H
#define HOZON_SOC_MCU_IMPL_TYPE_MBDESTONLINEDEBUG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "impl_type_float.h"

namespace hozon {
namespace soc_mcu {
struct MbdEstOnlineDebug {
    ::UInt8 freq_err_flag;
    ::UInt8 draw_yawrate_flag;
    ::UInt8 draw_steer_flag;
    ::UInt8 limit_yawrate_flag;
    ::UInt8 draw_pose_flag;
    ::UInt8 lat_track_err_flag;
    ::UInt8 limit_lat_poserr_flag;
    ::UInt8 limit_lat_angerr_flag;
    ::UInt8 yawctrl_leadlagfilter_err_flag;
    ::UInt8 ratectrl_leadlagfilter_err_flag;
    ::UInt8 latctrl_leadlagfilter_err_flag;
    ::UInt8 lat_act_err_flag;
    ::UInt8 limit_steer_err_flag;
    ::UInt8 limit_steerrate_err_flag;
    ::UInt8 lon_follow_err_flag;
    ::UInt8 lon_follow_pos_err_flag;
    ::UInt8 lon_follow_vel_err_flag;
    ::UInt8 lon_filter_err_flag;
    ::UInt8 lon_act_err_flag;
    ::UInt8 nan_err_flag;
    ::UInt8 pos_traj_err_flag;
    ::UInt8 postrajdisable_err_flag;
    ::UInt8 vx_err_avp_flag;
    ::UInt8 chassisinfo_err_flag;
    ::UInt8 ctrl_err;
    ::UInt32 diag_common_flag;
    ::Float diag_reserve_r0;
    ::Float diag_reserve_r1;
    ::Float diag_reserve_r2;
    ::Float diag_reserve_r3;
    ::Float diag_reserve_r4;
    ::Float diag_reserve_r5;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(freq_err_flag);
        fun(draw_yawrate_flag);
        fun(draw_steer_flag);
        fun(limit_yawrate_flag);
        fun(draw_pose_flag);
        fun(lat_track_err_flag);
        fun(limit_lat_poserr_flag);
        fun(limit_lat_angerr_flag);
        fun(yawctrl_leadlagfilter_err_flag);
        fun(ratectrl_leadlagfilter_err_flag);
        fun(latctrl_leadlagfilter_err_flag);
        fun(lat_act_err_flag);
        fun(limit_steer_err_flag);
        fun(limit_steerrate_err_flag);
        fun(lon_follow_err_flag);
        fun(lon_follow_pos_err_flag);
        fun(lon_follow_vel_err_flag);
        fun(lon_filter_err_flag);
        fun(lon_act_err_flag);
        fun(nan_err_flag);
        fun(pos_traj_err_flag);
        fun(postrajdisable_err_flag);
        fun(vx_err_avp_flag);
        fun(chassisinfo_err_flag);
        fun(ctrl_err);
        fun(diag_common_flag);
        fun(diag_reserve_r0);
        fun(diag_reserve_r1);
        fun(diag_reserve_r2);
        fun(diag_reserve_r3);
        fun(diag_reserve_r4);
        fun(diag_reserve_r5);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(freq_err_flag);
        fun(draw_yawrate_flag);
        fun(draw_steer_flag);
        fun(limit_yawrate_flag);
        fun(draw_pose_flag);
        fun(lat_track_err_flag);
        fun(limit_lat_poserr_flag);
        fun(limit_lat_angerr_flag);
        fun(yawctrl_leadlagfilter_err_flag);
        fun(ratectrl_leadlagfilter_err_flag);
        fun(latctrl_leadlagfilter_err_flag);
        fun(lat_act_err_flag);
        fun(limit_steer_err_flag);
        fun(limit_steerrate_err_flag);
        fun(lon_follow_err_flag);
        fun(lon_follow_pos_err_flag);
        fun(lon_follow_vel_err_flag);
        fun(lon_filter_err_flag);
        fun(lon_act_err_flag);
        fun(nan_err_flag);
        fun(pos_traj_err_flag);
        fun(postrajdisable_err_flag);
        fun(vx_err_avp_flag);
        fun(chassisinfo_err_flag);
        fun(ctrl_err);
        fun(diag_common_flag);
        fun(diag_reserve_r0);
        fun(diag_reserve_r1);
        fun(diag_reserve_r2);
        fun(diag_reserve_r3);
        fun(diag_reserve_r4);
        fun(diag_reserve_r5);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("freq_err_flag", freq_err_flag);
        fun("draw_yawrate_flag", draw_yawrate_flag);
        fun("draw_steer_flag", draw_steer_flag);
        fun("limit_yawrate_flag", limit_yawrate_flag);
        fun("draw_pose_flag", draw_pose_flag);
        fun("lat_track_err_flag", lat_track_err_flag);
        fun("limit_lat_poserr_flag", limit_lat_poserr_flag);
        fun("limit_lat_angerr_flag", limit_lat_angerr_flag);
        fun("yawctrl_leadlagfilter_err_flag", yawctrl_leadlagfilter_err_flag);
        fun("ratectrl_leadlagfilter_err_flag", ratectrl_leadlagfilter_err_flag);
        fun("latctrl_leadlagfilter_err_flag", latctrl_leadlagfilter_err_flag);
        fun("lat_act_err_flag", lat_act_err_flag);
        fun("limit_steer_err_flag", limit_steer_err_flag);
        fun("limit_steerrate_err_flag", limit_steerrate_err_flag);
        fun("lon_follow_err_flag", lon_follow_err_flag);
        fun("lon_follow_pos_err_flag", lon_follow_pos_err_flag);
        fun("lon_follow_vel_err_flag", lon_follow_vel_err_flag);
        fun("lon_filter_err_flag", lon_filter_err_flag);
        fun("lon_act_err_flag", lon_act_err_flag);
        fun("nan_err_flag", nan_err_flag);
        fun("pos_traj_err_flag", pos_traj_err_flag);
        fun("postrajdisable_err_flag", postrajdisable_err_flag);
        fun("vx_err_avp_flag", vx_err_avp_flag);
        fun("chassisinfo_err_flag", chassisinfo_err_flag);
        fun("ctrl_err", ctrl_err);
        fun("diag_common_flag", diag_common_flag);
        fun("diag_reserve_r0", diag_reserve_r0);
        fun("diag_reserve_r1", diag_reserve_r1);
        fun("diag_reserve_r2", diag_reserve_r2);
        fun("diag_reserve_r3", diag_reserve_r3);
        fun("diag_reserve_r4", diag_reserve_r4);
        fun("diag_reserve_r5", diag_reserve_r5);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("freq_err_flag", freq_err_flag);
        fun("draw_yawrate_flag", draw_yawrate_flag);
        fun("draw_steer_flag", draw_steer_flag);
        fun("limit_yawrate_flag", limit_yawrate_flag);
        fun("draw_pose_flag", draw_pose_flag);
        fun("lat_track_err_flag", lat_track_err_flag);
        fun("limit_lat_poserr_flag", limit_lat_poserr_flag);
        fun("limit_lat_angerr_flag", limit_lat_angerr_flag);
        fun("yawctrl_leadlagfilter_err_flag", yawctrl_leadlagfilter_err_flag);
        fun("ratectrl_leadlagfilter_err_flag", ratectrl_leadlagfilter_err_flag);
        fun("latctrl_leadlagfilter_err_flag", latctrl_leadlagfilter_err_flag);
        fun("lat_act_err_flag", lat_act_err_flag);
        fun("limit_steer_err_flag", limit_steer_err_flag);
        fun("limit_steerrate_err_flag", limit_steerrate_err_flag);
        fun("lon_follow_err_flag", lon_follow_err_flag);
        fun("lon_follow_pos_err_flag", lon_follow_pos_err_flag);
        fun("lon_follow_vel_err_flag", lon_follow_vel_err_flag);
        fun("lon_filter_err_flag", lon_filter_err_flag);
        fun("lon_act_err_flag", lon_act_err_flag);
        fun("nan_err_flag", nan_err_flag);
        fun("pos_traj_err_flag", pos_traj_err_flag);
        fun("postrajdisable_err_flag", postrajdisable_err_flag);
        fun("vx_err_avp_flag", vx_err_avp_flag);
        fun("chassisinfo_err_flag", chassisinfo_err_flag);
        fun("ctrl_err", ctrl_err);
        fun("diag_common_flag", diag_common_flag);
        fun("diag_reserve_r0", diag_reserve_r0);
        fun("diag_reserve_r1", diag_reserve_r1);
        fun("diag_reserve_r2", diag_reserve_r2);
        fun("diag_reserve_r3", diag_reserve_r3);
        fun("diag_reserve_r4", diag_reserve_r4);
        fun("diag_reserve_r5", diag_reserve_r5);
    }

    bool operator==(const ::hozon::soc_mcu::MbdEstOnlineDebug& t) const
    {
        return (freq_err_flag == t.freq_err_flag) && (draw_yawrate_flag == t.draw_yawrate_flag) && (draw_steer_flag == t.draw_steer_flag) && (limit_yawrate_flag == t.limit_yawrate_flag) && (draw_pose_flag == t.draw_pose_flag) && (lat_track_err_flag == t.lat_track_err_flag) && (limit_lat_poserr_flag == t.limit_lat_poserr_flag) && (limit_lat_angerr_flag == t.limit_lat_angerr_flag) && (yawctrl_leadlagfilter_err_flag == t.yawctrl_leadlagfilter_err_flag) && (ratectrl_leadlagfilter_err_flag == t.ratectrl_leadlagfilter_err_flag) && (latctrl_leadlagfilter_err_flag == t.latctrl_leadlagfilter_err_flag) && (lat_act_err_flag == t.lat_act_err_flag) && (limit_steer_err_flag == t.limit_steer_err_flag) && (limit_steerrate_err_flag == t.limit_steerrate_err_flag) && (lon_follow_err_flag == t.lon_follow_err_flag) && (lon_follow_pos_err_flag == t.lon_follow_pos_err_flag) && (lon_follow_vel_err_flag == t.lon_follow_vel_err_flag) && (lon_filter_err_flag == t.lon_filter_err_flag) && (lon_act_err_flag == t.lon_act_err_flag) && (nan_err_flag == t.nan_err_flag) && (pos_traj_err_flag == t.pos_traj_err_flag) && (postrajdisable_err_flag == t.postrajdisable_err_flag) && (vx_err_avp_flag == t.vx_err_avp_flag) && (chassisinfo_err_flag == t.chassisinfo_err_flag) && (ctrl_err == t.ctrl_err) && (diag_common_flag == t.diag_common_flag) && (fabs(static_cast<double>(diag_reserve_r0 - t.diag_reserve_r0)) < DBL_EPSILON) && (fabs(static_cast<double>(diag_reserve_r1 - t.diag_reserve_r1)) < DBL_EPSILON) && (fabs(static_cast<double>(diag_reserve_r2 - t.diag_reserve_r2)) < DBL_EPSILON) && (fabs(static_cast<double>(diag_reserve_r3 - t.diag_reserve_r3)) < DBL_EPSILON) && (fabs(static_cast<double>(diag_reserve_r4 - t.diag_reserve_r4)) < DBL_EPSILON) && (fabs(static_cast<double>(diag_reserve_r5 - t.diag_reserve_r5)) < DBL_EPSILON);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MBDESTONLINEDEBUG_H
