/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MBDPOSCALCDEBUG_H
#define HOZON_SOC_MCU_IMPL_TYPE_MBDPOSCALCDEBUG_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "impl_type_boolean.h"
#include "impl_type_float.h"

namespace hozon {
namespace soc_mcu {
struct MbdPosCalcDebug {
    ::Double posecalc_posedata_timestamp;
    ::Boolean posecalc_inputdata_valid;
    ::Boolean posecalc_posedata_timecheck;
    ::Boolean posecalc_enable;
    ::Float posecalc_timedelay;
    ::Double posedata_world_pos_x;
    ::Double posedata_world_pos_y;
    ::Float posecalc_vrf_vel_x;
    ::Float posecalc_vrf_vel_y;
    ::Float posecalc_vrf_acc_x;
    ::Float posecalc_vrf_acc_y;
    ::Float posedata_world_heading;
    ::Float posedata_vrf_yawrate;
    ::Double posecalc_world_pos_x;
    ::Double posecalc_world_pos_y;
    ::Float posedata_vrf_vel_x;
    ::Float posedata_vrf_vel_y;
    ::Float posedata_vrf_acc_x;
    ::Float posedata_vrf_acc_y;
    ::Float posecalc_world_vel_x;
    ::Float posecalc_world_vel_y;
    ::Float posecalc_world_acc_x;
    ::Float posecalc_world_acc_y;
    ::Float posecalc_world_heading;
    ::Float posecalc_world_pitch;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(posecalc_posedata_timestamp);
        fun(posecalc_inputdata_valid);
        fun(posecalc_posedata_timecheck);
        fun(posecalc_enable);
        fun(posecalc_timedelay);
        fun(posedata_world_pos_x);
        fun(posedata_world_pos_y);
        fun(posecalc_vrf_vel_x);
        fun(posecalc_vrf_vel_y);
        fun(posecalc_vrf_acc_x);
        fun(posecalc_vrf_acc_y);
        fun(posedata_world_heading);
        fun(posedata_vrf_yawrate);
        fun(posecalc_world_pos_x);
        fun(posecalc_world_pos_y);
        fun(posedata_vrf_vel_x);
        fun(posedata_vrf_vel_y);
        fun(posedata_vrf_acc_x);
        fun(posedata_vrf_acc_y);
        fun(posecalc_world_vel_x);
        fun(posecalc_world_vel_y);
        fun(posecalc_world_acc_x);
        fun(posecalc_world_acc_y);
        fun(posecalc_world_heading);
        fun(posecalc_world_pitch);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(posecalc_posedata_timestamp);
        fun(posecalc_inputdata_valid);
        fun(posecalc_posedata_timecheck);
        fun(posecalc_enable);
        fun(posecalc_timedelay);
        fun(posedata_world_pos_x);
        fun(posedata_world_pos_y);
        fun(posecalc_vrf_vel_x);
        fun(posecalc_vrf_vel_y);
        fun(posecalc_vrf_acc_x);
        fun(posecalc_vrf_acc_y);
        fun(posedata_world_heading);
        fun(posedata_vrf_yawrate);
        fun(posecalc_world_pos_x);
        fun(posecalc_world_pos_y);
        fun(posedata_vrf_vel_x);
        fun(posedata_vrf_vel_y);
        fun(posedata_vrf_acc_x);
        fun(posedata_vrf_acc_y);
        fun(posecalc_world_vel_x);
        fun(posecalc_world_vel_y);
        fun(posecalc_world_acc_x);
        fun(posecalc_world_acc_y);
        fun(posecalc_world_heading);
        fun(posecalc_world_pitch);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("posecalc_posedata_timestamp", posecalc_posedata_timestamp);
        fun("posecalc_inputdata_valid", posecalc_inputdata_valid);
        fun("posecalc_posedata_timecheck", posecalc_posedata_timecheck);
        fun("posecalc_enable", posecalc_enable);
        fun("posecalc_timedelay", posecalc_timedelay);
        fun("posedata_world_pos_x", posedata_world_pos_x);
        fun("posedata_world_pos_y", posedata_world_pos_y);
        fun("posecalc_vrf_vel_x", posecalc_vrf_vel_x);
        fun("posecalc_vrf_vel_y", posecalc_vrf_vel_y);
        fun("posecalc_vrf_acc_x", posecalc_vrf_acc_x);
        fun("posecalc_vrf_acc_y", posecalc_vrf_acc_y);
        fun("posedata_world_heading", posedata_world_heading);
        fun("posedata_vrf_yawrate", posedata_vrf_yawrate);
        fun("posecalc_world_pos_x", posecalc_world_pos_x);
        fun("posecalc_world_pos_y", posecalc_world_pos_y);
        fun("posedata_vrf_vel_x", posedata_vrf_vel_x);
        fun("posedata_vrf_vel_y", posedata_vrf_vel_y);
        fun("posedata_vrf_acc_x", posedata_vrf_acc_x);
        fun("posedata_vrf_acc_y", posedata_vrf_acc_y);
        fun("posecalc_world_vel_x", posecalc_world_vel_x);
        fun("posecalc_world_vel_y", posecalc_world_vel_y);
        fun("posecalc_world_acc_x", posecalc_world_acc_x);
        fun("posecalc_world_acc_y", posecalc_world_acc_y);
        fun("posecalc_world_heading", posecalc_world_heading);
        fun("posecalc_world_pitch", posecalc_world_pitch);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("posecalc_posedata_timestamp", posecalc_posedata_timestamp);
        fun("posecalc_inputdata_valid", posecalc_inputdata_valid);
        fun("posecalc_posedata_timecheck", posecalc_posedata_timecheck);
        fun("posecalc_enable", posecalc_enable);
        fun("posecalc_timedelay", posecalc_timedelay);
        fun("posedata_world_pos_x", posedata_world_pos_x);
        fun("posedata_world_pos_y", posedata_world_pos_y);
        fun("posecalc_vrf_vel_x", posecalc_vrf_vel_x);
        fun("posecalc_vrf_vel_y", posecalc_vrf_vel_y);
        fun("posecalc_vrf_acc_x", posecalc_vrf_acc_x);
        fun("posecalc_vrf_acc_y", posecalc_vrf_acc_y);
        fun("posedata_world_heading", posedata_world_heading);
        fun("posedata_vrf_yawrate", posedata_vrf_yawrate);
        fun("posecalc_world_pos_x", posecalc_world_pos_x);
        fun("posecalc_world_pos_y", posecalc_world_pos_y);
        fun("posedata_vrf_vel_x", posedata_vrf_vel_x);
        fun("posedata_vrf_vel_y", posedata_vrf_vel_y);
        fun("posedata_vrf_acc_x", posedata_vrf_acc_x);
        fun("posedata_vrf_acc_y", posedata_vrf_acc_y);
        fun("posecalc_world_vel_x", posecalc_world_vel_x);
        fun("posecalc_world_vel_y", posecalc_world_vel_y);
        fun("posecalc_world_acc_x", posecalc_world_acc_x);
        fun("posecalc_world_acc_y", posecalc_world_acc_y);
        fun("posecalc_world_heading", posecalc_world_heading);
        fun("posecalc_world_pitch", posecalc_world_pitch);
    }

    bool operator==(const ::hozon::soc_mcu::MbdPosCalcDebug& t) const
    {
        return (fabs(static_cast<double>(posecalc_posedata_timestamp - t.posecalc_posedata_timestamp)) < DBL_EPSILON) && (posecalc_inputdata_valid == t.posecalc_inputdata_valid) && (posecalc_posedata_timecheck == t.posecalc_posedata_timecheck) && (posecalc_enable == t.posecalc_enable) && (fabs(static_cast<double>(posecalc_timedelay - t.posecalc_timedelay)) < DBL_EPSILON) && (fabs(static_cast<double>(posedata_world_pos_x - t.posedata_world_pos_x)) < DBL_EPSILON) && (fabs(static_cast<double>(posedata_world_pos_y - t.posedata_world_pos_y)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_vrf_vel_x - t.posecalc_vrf_vel_x)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_vrf_vel_y - t.posecalc_vrf_vel_y)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_vrf_acc_x - t.posecalc_vrf_acc_x)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_vrf_acc_y - t.posecalc_vrf_acc_y)) < DBL_EPSILON) && (fabs(static_cast<double>(posedata_world_heading - t.posedata_world_heading)) < DBL_EPSILON) && (fabs(static_cast<double>(posedata_vrf_yawrate - t.posedata_vrf_yawrate)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_world_pos_x - t.posecalc_world_pos_x)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_world_pos_y - t.posecalc_world_pos_y)) < DBL_EPSILON) && (fabs(static_cast<double>(posedata_vrf_vel_x - t.posedata_vrf_vel_x)) < DBL_EPSILON) && (fabs(static_cast<double>(posedata_vrf_vel_y - t.posedata_vrf_vel_y)) < DBL_EPSILON) && (fabs(static_cast<double>(posedata_vrf_acc_x - t.posedata_vrf_acc_x)) < DBL_EPSILON) && (fabs(static_cast<double>(posedata_vrf_acc_y - t.posedata_vrf_acc_y)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_world_vel_x - t.posecalc_world_vel_x)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_world_vel_y - t.posecalc_world_vel_y)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_world_acc_x - t.posecalc_world_acc_x)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_world_acc_y - t.posecalc_world_acc_y)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_world_heading - t.posecalc_world_heading)) < DBL_EPSILON) && (fabs(static_cast<double>(posecalc_world_pitch - t.posecalc_world_pitch)) < DBL_EPSILON);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MBDPOSCALCDEBUG_H
