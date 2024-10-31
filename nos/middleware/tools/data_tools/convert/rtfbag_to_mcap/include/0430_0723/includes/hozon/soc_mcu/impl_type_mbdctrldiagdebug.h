/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MBDCTRLDIAGDEBUG_H
#define HOZON_SOC_MCU_IMPL_TYPE_MBDCTRLDIAGDEBUG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "impl_type_uint32.h"

namespace hozon {
namespace soc_mcu {
struct MbdCtrlDiagDebug {
    ::UInt16 calibration_counter;
    ::UInt8 condition_bend_flag;
    ::UInt8 condition_standstill_flag;
    ::UInt8 condition_straight_flag;
    ::UInt8 condition_straight_spd_flag;
    ::UInt8 steeroffset_condition;
    ::UInt8 condition_straight_yawrate_flag;
    ::UInt8 condition_straight_steer_flag;
    ::UInt8 condition_straight_latposerr_flag;
    ::UInt8 condition_straight_curvcmd_flag;
    ::UInt8 condition_straight_ayfdbk_flag;
    ::UInt8 condition_acc_flag;
    ::UInt8 condition_brk_flag;
    ::UInt8 condition_cruise_flag;
    ::UInt8 steeroffset_eps_learning_flag;
    ::Float steeroffset_to_closeloop;
    ::Float yawrate_bias_to_closeloop;
    ::Float heading_bias_to_closeloop;
    ::Float ax_bias_to_closeloop;
    ::Float ay_bias_to_closeloop;
    ::Float united_est_yawrate;
    ::Float united_est_beta;
    ::Float united_est_vy;
    ::Float united_est_lat_vx;
    ::Float united_est_jerk;
    ::Float united_est_ax;
    ::Float united_est_lon_vx;
    ::Float united_est_mass;
    ::Float united_est_slope;
    ::UInt32 estol_common_flag;
    ::UInt32 condition_type;
    ::Float estol_reserve_r0;
    ::Float estol_reserve_r1;
    ::Float estol_reserve_r2;
    ::Float estol_reserve_r3;
    ::Float estol_reserve_r4;
    ::Float estol_reserve_r5;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(calibration_counter);
        fun(condition_bend_flag);
        fun(condition_standstill_flag);
        fun(condition_straight_flag);
        fun(condition_straight_spd_flag);
        fun(steeroffset_condition);
        fun(condition_straight_yawrate_flag);
        fun(condition_straight_steer_flag);
        fun(condition_straight_latposerr_flag);
        fun(condition_straight_curvcmd_flag);
        fun(condition_straight_ayfdbk_flag);
        fun(condition_acc_flag);
        fun(condition_brk_flag);
        fun(condition_cruise_flag);
        fun(steeroffset_eps_learning_flag);
        fun(steeroffset_to_closeloop);
        fun(yawrate_bias_to_closeloop);
        fun(heading_bias_to_closeloop);
        fun(ax_bias_to_closeloop);
        fun(ay_bias_to_closeloop);
        fun(united_est_yawrate);
        fun(united_est_beta);
        fun(united_est_vy);
        fun(united_est_lat_vx);
        fun(united_est_jerk);
        fun(united_est_ax);
        fun(united_est_lon_vx);
        fun(united_est_mass);
        fun(united_est_slope);
        fun(estol_common_flag);
        fun(condition_type);
        fun(estol_reserve_r0);
        fun(estol_reserve_r1);
        fun(estol_reserve_r2);
        fun(estol_reserve_r3);
        fun(estol_reserve_r4);
        fun(estol_reserve_r5);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(calibration_counter);
        fun(condition_bend_flag);
        fun(condition_standstill_flag);
        fun(condition_straight_flag);
        fun(condition_straight_spd_flag);
        fun(steeroffset_condition);
        fun(condition_straight_yawrate_flag);
        fun(condition_straight_steer_flag);
        fun(condition_straight_latposerr_flag);
        fun(condition_straight_curvcmd_flag);
        fun(condition_straight_ayfdbk_flag);
        fun(condition_acc_flag);
        fun(condition_brk_flag);
        fun(condition_cruise_flag);
        fun(steeroffset_eps_learning_flag);
        fun(steeroffset_to_closeloop);
        fun(yawrate_bias_to_closeloop);
        fun(heading_bias_to_closeloop);
        fun(ax_bias_to_closeloop);
        fun(ay_bias_to_closeloop);
        fun(united_est_yawrate);
        fun(united_est_beta);
        fun(united_est_vy);
        fun(united_est_lat_vx);
        fun(united_est_jerk);
        fun(united_est_ax);
        fun(united_est_lon_vx);
        fun(united_est_mass);
        fun(united_est_slope);
        fun(estol_common_flag);
        fun(condition_type);
        fun(estol_reserve_r0);
        fun(estol_reserve_r1);
        fun(estol_reserve_r2);
        fun(estol_reserve_r3);
        fun(estol_reserve_r4);
        fun(estol_reserve_r5);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("calibration_counter", calibration_counter);
        fun("condition_bend_flag", condition_bend_flag);
        fun("condition_standstill_flag", condition_standstill_flag);
        fun("condition_straight_flag", condition_straight_flag);
        fun("condition_straight_spd_flag", condition_straight_spd_flag);
        fun("steeroffset_condition", steeroffset_condition);
        fun("condition_straight_yawrate_flag", condition_straight_yawrate_flag);
        fun("condition_straight_steer_flag", condition_straight_steer_flag);
        fun("condition_straight_latposerr_flag", condition_straight_latposerr_flag);
        fun("condition_straight_curvcmd_flag", condition_straight_curvcmd_flag);
        fun("condition_straight_ayfdbk_flag", condition_straight_ayfdbk_flag);
        fun("condition_acc_flag", condition_acc_flag);
        fun("condition_brk_flag", condition_brk_flag);
        fun("condition_cruise_flag", condition_cruise_flag);
        fun("steeroffset_eps_learning_flag", steeroffset_eps_learning_flag);
        fun("steeroffset_to_closeloop", steeroffset_to_closeloop);
        fun("yawrate_bias_to_closeloop", yawrate_bias_to_closeloop);
        fun("heading_bias_to_closeloop", heading_bias_to_closeloop);
        fun("ax_bias_to_closeloop", ax_bias_to_closeloop);
        fun("ay_bias_to_closeloop", ay_bias_to_closeloop);
        fun("united_est_yawrate", united_est_yawrate);
        fun("united_est_beta", united_est_beta);
        fun("united_est_vy", united_est_vy);
        fun("united_est_lat_vx", united_est_lat_vx);
        fun("united_est_jerk", united_est_jerk);
        fun("united_est_ax", united_est_ax);
        fun("united_est_lon_vx", united_est_lon_vx);
        fun("united_est_mass", united_est_mass);
        fun("united_est_slope", united_est_slope);
        fun("estol_common_flag", estol_common_flag);
        fun("condition_type", condition_type);
        fun("estol_reserve_r0", estol_reserve_r0);
        fun("estol_reserve_r1", estol_reserve_r1);
        fun("estol_reserve_r2", estol_reserve_r2);
        fun("estol_reserve_r3", estol_reserve_r3);
        fun("estol_reserve_r4", estol_reserve_r4);
        fun("estol_reserve_r5", estol_reserve_r5);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("calibration_counter", calibration_counter);
        fun("condition_bend_flag", condition_bend_flag);
        fun("condition_standstill_flag", condition_standstill_flag);
        fun("condition_straight_flag", condition_straight_flag);
        fun("condition_straight_spd_flag", condition_straight_spd_flag);
        fun("steeroffset_condition", steeroffset_condition);
        fun("condition_straight_yawrate_flag", condition_straight_yawrate_flag);
        fun("condition_straight_steer_flag", condition_straight_steer_flag);
        fun("condition_straight_latposerr_flag", condition_straight_latposerr_flag);
        fun("condition_straight_curvcmd_flag", condition_straight_curvcmd_flag);
        fun("condition_straight_ayfdbk_flag", condition_straight_ayfdbk_flag);
        fun("condition_acc_flag", condition_acc_flag);
        fun("condition_brk_flag", condition_brk_flag);
        fun("condition_cruise_flag", condition_cruise_flag);
        fun("steeroffset_eps_learning_flag", steeroffset_eps_learning_flag);
        fun("steeroffset_to_closeloop", steeroffset_to_closeloop);
        fun("yawrate_bias_to_closeloop", yawrate_bias_to_closeloop);
        fun("heading_bias_to_closeloop", heading_bias_to_closeloop);
        fun("ax_bias_to_closeloop", ax_bias_to_closeloop);
        fun("ay_bias_to_closeloop", ay_bias_to_closeloop);
        fun("united_est_yawrate", united_est_yawrate);
        fun("united_est_beta", united_est_beta);
        fun("united_est_vy", united_est_vy);
        fun("united_est_lat_vx", united_est_lat_vx);
        fun("united_est_jerk", united_est_jerk);
        fun("united_est_ax", united_est_ax);
        fun("united_est_lon_vx", united_est_lon_vx);
        fun("united_est_mass", united_est_mass);
        fun("united_est_slope", united_est_slope);
        fun("estol_common_flag", estol_common_flag);
        fun("condition_type", condition_type);
        fun("estol_reserve_r0", estol_reserve_r0);
        fun("estol_reserve_r1", estol_reserve_r1);
        fun("estol_reserve_r2", estol_reserve_r2);
        fun("estol_reserve_r3", estol_reserve_r3);
        fun("estol_reserve_r4", estol_reserve_r4);
        fun("estol_reserve_r5", estol_reserve_r5);
    }

    bool operator==(const ::hozon::soc_mcu::MbdCtrlDiagDebug& t) const
    {
        return (calibration_counter == t.calibration_counter) && (condition_bend_flag == t.condition_bend_flag) && (condition_standstill_flag == t.condition_standstill_flag) && (condition_straight_flag == t.condition_straight_flag) && (condition_straight_spd_flag == t.condition_straight_spd_flag) && (steeroffset_condition == t.steeroffset_condition) && (condition_straight_yawrate_flag == t.condition_straight_yawrate_flag) && (condition_straight_steer_flag == t.condition_straight_steer_flag) && (condition_straight_latposerr_flag == t.condition_straight_latposerr_flag) && (condition_straight_curvcmd_flag == t.condition_straight_curvcmd_flag) && (condition_straight_ayfdbk_flag == t.condition_straight_ayfdbk_flag) && (condition_acc_flag == t.condition_acc_flag) && (condition_brk_flag == t.condition_brk_flag) && (condition_cruise_flag == t.condition_cruise_flag) && (steeroffset_eps_learning_flag == t.steeroffset_eps_learning_flag) && (fabs(static_cast<double>(steeroffset_to_closeloop - t.steeroffset_to_closeloop)) < DBL_EPSILON) && (fabs(static_cast<double>(yawrate_bias_to_closeloop - t.yawrate_bias_to_closeloop)) < DBL_EPSILON) && (fabs(static_cast<double>(heading_bias_to_closeloop - t.heading_bias_to_closeloop)) < DBL_EPSILON) && (fabs(static_cast<double>(ax_bias_to_closeloop - t.ax_bias_to_closeloop)) < DBL_EPSILON) && (fabs(static_cast<double>(ay_bias_to_closeloop - t.ay_bias_to_closeloop)) < DBL_EPSILON) && (fabs(static_cast<double>(united_est_yawrate - t.united_est_yawrate)) < DBL_EPSILON) && (fabs(static_cast<double>(united_est_beta - t.united_est_beta)) < DBL_EPSILON) && (fabs(static_cast<double>(united_est_vy - t.united_est_vy)) < DBL_EPSILON) && (fabs(static_cast<double>(united_est_lat_vx - t.united_est_lat_vx)) < DBL_EPSILON) && (fabs(static_cast<double>(united_est_jerk - t.united_est_jerk)) < DBL_EPSILON) && (fabs(static_cast<double>(united_est_ax - t.united_est_ax)) < DBL_EPSILON) && (fabs(static_cast<double>(united_est_lon_vx - t.united_est_lon_vx)) < DBL_EPSILON) && (fabs(static_cast<double>(united_est_mass - t.united_est_mass)) < DBL_EPSILON) && (fabs(static_cast<double>(united_est_slope - t.united_est_slope)) < DBL_EPSILON) && (estol_common_flag == t.estol_common_flag) && (condition_type == t.condition_type) && (fabs(static_cast<double>(estol_reserve_r0 - t.estol_reserve_r0)) < DBL_EPSILON) && (fabs(static_cast<double>(estol_reserve_r1 - t.estol_reserve_r1)) < DBL_EPSILON) && (fabs(static_cast<double>(estol_reserve_r2 - t.estol_reserve_r2)) < DBL_EPSILON) && (fabs(static_cast<double>(estol_reserve_r3 - t.estol_reserve_r3)) < DBL_EPSILON) && (fabs(static_cast<double>(estol_reserve_r4 - t.estol_reserve_r4)) < DBL_EPSILON) && (fabs(static_cast<double>(estol_reserve_r5 - t.estol_reserve_r5)) < DBL_EPSILON);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MBDCTRLDIAGDEBUG_H
