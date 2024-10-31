/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MBDLONCTRLDEBUG_H
#define HOZON_SOC_MCU_IMPL_TYPE_MBDLONCTRLDEBUG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "impl_type_float.h"

namespace hozon {
namespace soc_mcu {
struct MbdLonCtrlDebug {
    ::UInt8 lonctrl_modecmd;
    ::UInt32 lonctrl_resetflag;
    ::Float lonctrl_sys_poserr;
    ::Float lonctrl_sys_velff;
    ::Float lonctrl_sys_accff;
    ::UInt32 lonctrl_sys_gearcmd;
    ::UInt32 lonctrl_sys_gearena;
    ::UInt32 lonctrl_sys_brkemerg;
    ::Float lonctrl_api_poscmd;
    ::Float lonctrl_api_velcmd;
    ::Float lonctrl_api_acccmd;
    ::Float lonctrl_api_thrcmd;
    ::Float lonctrl_api_brkcmd;
    ::Float lonctrl_fdbk_vx;
    ::Float lonctrl_fdbk_ax;
    ::Float lonctrl_fdbk_pitch;
    ::Float lonctrl_fdbk_gear;
    ::Float lonctrl_fdbk_reverse;
    ::Float lonctrl_pos_dyn;
    ::Float lonctrl_posctrl_p;
    ::Float lonctrl_posctrl_i;
    ::Float lonctrl_pos_poserr_filter;
    ::Float lonctrl_pos_pi_velcmd;
    ::Float lonctrl_pos_output_velcmd;
    ::Float lonctrl_vel_dyn;
    ::Float lonctrl_vel_velcmd_lmt;
    ::Float lonctrl_vel_vel_err;
    ::Float lonctrl_velctrl_p;
    ::Float lonctrl_velctrl_i;
    ::Float lonctrl_vel_pi_acc_cmd;
    ::Float lonctrl_vel_pi_acccmd_filter;
    ::Float lonctrl_vel_accpitch;
    ::Float lonctrl_vel_accdamper;
    ::Float lonctrl_vel_accff_filter;
    ::Float lonctrl_vel_output_accCmd;
    ::Float lonctrl_vel_output_accCmd_filter;
    ::Float lonctrl_thrust_thr_dyn;
    ::Float lonctrl_thrust_thr_accerr;
    ::Float lonctrl_thrust_brk_dyn;
    ::Float lonctrl_thrust_brk_accerr;
    ::Float lonctrl_thrust_fdbk_ax_filter;
    ::Float lonctrl_thrust_thr_acc_cmd_filter;
    ::Float lonctrl_thrust_brk_acc_cmd_filter;
    ::Float lonctrl_thrustctrl_thr_p;
    ::Float lonctrl_thrustctrl_thr_i;
    ::Float lonctrl_thrustctrl_brk_p;
    ::Float lonctrl_thrustctrl_brk_i;
    ::Float lonctrl_thrust_pi_thr_acc_cmd;
    ::Float lonctrl_thrust_pi_brk_acc_cmd;
    ::Float lonctrl_thrust_acc_cmd_filter_lmt;
    ::Float lonctrl_thrust_acctothr_gain;
    ::Float lonctrl_thrust_throut_throcmd;
    ::Float lonctrl_thrust_acctobrk_gain;
    ::Float lonctrl_thrust_brkout_brkcmd;
    ::Float lonctrl_analog_autput_throtcmd;
    ::Float lonctrl_analog_autput_brkcmd;
    ::Float lonctrl_vel_mrac_cmd;
    ::Float lonctrl_vel_reference_model;
    ::Float lonctrl_accel_eso_cmd;
    ::Float lonctrl_deccel_eso_cmd;
    ::Float lonctrl_slope_estimate;
    ::Float lonctrl_mass_estimate;
    ::UInt8 lonctrl_error;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(lonctrl_modecmd);
        fun(lonctrl_resetflag);
        fun(lonctrl_sys_poserr);
        fun(lonctrl_sys_velff);
        fun(lonctrl_sys_accff);
        fun(lonctrl_sys_gearcmd);
        fun(lonctrl_sys_gearena);
        fun(lonctrl_sys_brkemerg);
        fun(lonctrl_api_poscmd);
        fun(lonctrl_api_velcmd);
        fun(lonctrl_api_acccmd);
        fun(lonctrl_api_thrcmd);
        fun(lonctrl_api_brkcmd);
        fun(lonctrl_fdbk_vx);
        fun(lonctrl_fdbk_ax);
        fun(lonctrl_fdbk_pitch);
        fun(lonctrl_fdbk_gear);
        fun(lonctrl_fdbk_reverse);
        fun(lonctrl_pos_dyn);
        fun(lonctrl_posctrl_p);
        fun(lonctrl_posctrl_i);
        fun(lonctrl_pos_poserr_filter);
        fun(lonctrl_pos_pi_velcmd);
        fun(lonctrl_pos_output_velcmd);
        fun(lonctrl_vel_dyn);
        fun(lonctrl_vel_velcmd_lmt);
        fun(lonctrl_vel_vel_err);
        fun(lonctrl_velctrl_p);
        fun(lonctrl_velctrl_i);
        fun(lonctrl_vel_pi_acc_cmd);
        fun(lonctrl_vel_pi_acccmd_filter);
        fun(lonctrl_vel_accpitch);
        fun(lonctrl_vel_accdamper);
        fun(lonctrl_vel_accff_filter);
        fun(lonctrl_vel_output_accCmd);
        fun(lonctrl_vel_output_accCmd_filter);
        fun(lonctrl_thrust_thr_dyn);
        fun(lonctrl_thrust_thr_accerr);
        fun(lonctrl_thrust_brk_dyn);
        fun(lonctrl_thrust_brk_accerr);
        fun(lonctrl_thrust_fdbk_ax_filter);
        fun(lonctrl_thrust_thr_acc_cmd_filter);
        fun(lonctrl_thrust_brk_acc_cmd_filter);
        fun(lonctrl_thrustctrl_thr_p);
        fun(lonctrl_thrustctrl_thr_i);
        fun(lonctrl_thrustctrl_brk_p);
        fun(lonctrl_thrustctrl_brk_i);
        fun(lonctrl_thrust_pi_thr_acc_cmd);
        fun(lonctrl_thrust_pi_brk_acc_cmd);
        fun(lonctrl_thrust_acc_cmd_filter_lmt);
        fun(lonctrl_thrust_acctothr_gain);
        fun(lonctrl_thrust_throut_throcmd);
        fun(lonctrl_thrust_acctobrk_gain);
        fun(lonctrl_thrust_brkout_brkcmd);
        fun(lonctrl_analog_autput_throtcmd);
        fun(lonctrl_analog_autput_brkcmd);
        fun(lonctrl_vel_mrac_cmd);
        fun(lonctrl_vel_reference_model);
        fun(lonctrl_accel_eso_cmd);
        fun(lonctrl_deccel_eso_cmd);
        fun(lonctrl_slope_estimate);
        fun(lonctrl_mass_estimate);
        fun(lonctrl_error);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(lonctrl_modecmd);
        fun(lonctrl_resetflag);
        fun(lonctrl_sys_poserr);
        fun(lonctrl_sys_velff);
        fun(lonctrl_sys_accff);
        fun(lonctrl_sys_gearcmd);
        fun(lonctrl_sys_gearena);
        fun(lonctrl_sys_brkemerg);
        fun(lonctrl_api_poscmd);
        fun(lonctrl_api_velcmd);
        fun(lonctrl_api_acccmd);
        fun(lonctrl_api_thrcmd);
        fun(lonctrl_api_brkcmd);
        fun(lonctrl_fdbk_vx);
        fun(lonctrl_fdbk_ax);
        fun(lonctrl_fdbk_pitch);
        fun(lonctrl_fdbk_gear);
        fun(lonctrl_fdbk_reverse);
        fun(lonctrl_pos_dyn);
        fun(lonctrl_posctrl_p);
        fun(lonctrl_posctrl_i);
        fun(lonctrl_pos_poserr_filter);
        fun(lonctrl_pos_pi_velcmd);
        fun(lonctrl_pos_output_velcmd);
        fun(lonctrl_vel_dyn);
        fun(lonctrl_vel_velcmd_lmt);
        fun(lonctrl_vel_vel_err);
        fun(lonctrl_velctrl_p);
        fun(lonctrl_velctrl_i);
        fun(lonctrl_vel_pi_acc_cmd);
        fun(lonctrl_vel_pi_acccmd_filter);
        fun(lonctrl_vel_accpitch);
        fun(lonctrl_vel_accdamper);
        fun(lonctrl_vel_accff_filter);
        fun(lonctrl_vel_output_accCmd);
        fun(lonctrl_vel_output_accCmd_filter);
        fun(lonctrl_thrust_thr_dyn);
        fun(lonctrl_thrust_thr_accerr);
        fun(lonctrl_thrust_brk_dyn);
        fun(lonctrl_thrust_brk_accerr);
        fun(lonctrl_thrust_fdbk_ax_filter);
        fun(lonctrl_thrust_thr_acc_cmd_filter);
        fun(lonctrl_thrust_brk_acc_cmd_filter);
        fun(lonctrl_thrustctrl_thr_p);
        fun(lonctrl_thrustctrl_thr_i);
        fun(lonctrl_thrustctrl_brk_p);
        fun(lonctrl_thrustctrl_brk_i);
        fun(lonctrl_thrust_pi_thr_acc_cmd);
        fun(lonctrl_thrust_pi_brk_acc_cmd);
        fun(lonctrl_thrust_acc_cmd_filter_lmt);
        fun(lonctrl_thrust_acctothr_gain);
        fun(lonctrl_thrust_throut_throcmd);
        fun(lonctrl_thrust_acctobrk_gain);
        fun(lonctrl_thrust_brkout_brkcmd);
        fun(lonctrl_analog_autput_throtcmd);
        fun(lonctrl_analog_autput_brkcmd);
        fun(lonctrl_vel_mrac_cmd);
        fun(lonctrl_vel_reference_model);
        fun(lonctrl_accel_eso_cmd);
        fun(lonctrl_deccel_eso_cmd);
        fun(lonctrl_slope_estimate);
        fun(lonctrl_mass_estimate);
        fun(lonctrl_error);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("lonctrl_modecmd", lonctrl_modecmd);
        fun("lonctrl_resetflag", lonctrl_resetflag);
        fun("lonctrl_sys_poserr", lonctrl_sys_poserr);
        fun("lonctrl_sys_velff", lonctrl_sys_velff);
        fun("lonctrl_sys_accff", lonctrl_sys_accff);
        fun("lonctrl_sys_gearcmd", lonctrl_sys_gearcmd);
        fun("lonctrl_sys_gearena", lonctrl_sys_gearena);
        fun("lonctrl_sys_brkemerg", lonctrl_sys_brkemerg);
        fun("lonctrl_api_poscmd", lonctrl_api_poscmd);
        fun("lonctrl_api_velcmd", lonctrl_api_velcmd);
        fun("lonctrl_api_acccmd", lonctrl_api_acccmd);
        fun("lonctrl_api_thrcmd", lonctrl_api_thrcmd);
        fun("lonctrl_api_brkcmd", lonctrl_api_brkcmd);
        fun("lonctrl_fdbk_vx", lonctrl_fdbk_vx);
        fun("lonctrl_fdbk_ax", lonctrl_fdbk_ax);
        fun("lonctrl_fdbk_pitch", lonctrl_fdbk_pitch);
        fun("lonctrl_fdbk_gear", lonctrl_fdbk_gear);
        fun("lonctrl_fdbk_reverse", lonctrl_fdbk_reverse);
        fun("lonctrl_pos_dyn", lonctrl_pos_dyn);
        fun("lonctrl_posctrl_p", lonctrl_posctrl_p);
        fun("lonctrl_posctrl_i", lonctrl_posctrl_i);
        fun("lonctrl_pos_poserr_filter", lonctrl_pos_poserr_filter);
        fun("lonctrl_pos_pi_velcmd", lonctrl_pos_pi_velcmd);
        fun("lonctrl_pos_output_velcmd", lonctrl_pos_output_velcmd);
        fun("lonctrl_vel_dyn", lonctrl_vel_dyn);
        fun("lonctrl_vel_velcmd_lmt", lonctrl_vel_velcmd_lmt);
        fun("lonctrl_vel_vel_err", lonctrl_vel_vel_err);
        fun("lonctrl_velctrl_p", lonctrl_velctrl_p);
        fun("lonctrl_velctrl_i", lonctrl_velctrl_i);
        fun("lonctrl_vel_pi_acc_cmd", lonctrl_vel_pi_acc_cmd);
        fun("lonctrl_vel_pi_acccmd_filter", lonctrl_vel_pi_acccmd_filter);
        fun("lonctrl_vel_accpitch", lonctrl_vel_accpitch);
        fun("lonctrl_vel_accdamper", lonctrl_vel_accdamper);
        fun("lonctrl_vel_accff_filter", lonctrl_vel_accff_filter);
        fun("lonctrl_vel_output_accCmd", lonctrl_vel_output_accCmd);
        fun("lonctrl_vel_output_accCmd_filter", lonctrl_vel_output_accCmd_filter);
        fun("lonctrl_thrust_thr_dyn", lonctrl_thrust_thr_dyn);
        fun("lonctrl_thrust_thr_accerr", lonctrl_thrust_thr_accerr);
        fun("lonctrl_thrust_brk_dyn", lonctrl_thrust_brk_dyn);
        fun("lonctrl_thrust_brk_accerr", lonctrl_thrust_brk_accerr);
        fun("lonctrl_thrust_fdbk_ax_filter", lonctrl_thrust_fdbk_ax_filter);
        fun("lonctrl_thrust_thr_acc_cmd_filter", lonctrl_thrust_thr_acc_cmd_filter);
        fun("lonctrl_thrust_brk_acc_cmd_filter", lonctrl_thrust_brk_acc_cmd_filter);
        fun("lonctrl_thrustctrl_thr_p", lonctrl_thrustctrl_thr_p);
        fun("lonctrl_thrustctrl_thr_i", lonctrl_thrustctrl_thr_i);
        fun("lonctrl_thrustctrl_brk_p", lonctrl_thrustctrl_brk_p);
        fun("lonctrl_thrustctrl_brk_i", lonctrl_thrustctrl_brk_i);
        fun("lonctrl_thrust_pi_thr_acc_cmd", lonctrl_thrust_pi_thr_acc_cmd);
        fun("lonctrl_thrust_pi_brk_acc_cmd", lonctrl_thrust_pi_brk_acc_cmd);
        fun("lonctrl_thrust_acc_cmd_filter_lmt", lonctrl_thrust_acc_cmd_filter_lmt);
        fun("lonctrl_thrust_acctothr_gain", lonctrl_thrust_acctothr_gain);
        fun("lonctrl_thrust_throut_throcmd", lonctrl_thrust_throut_throcmd);
        fun("lonctrl_thrust_acctobrk_gain", lonctrl_thrust_acctobrk_gain);
        fun("lonctrl_thrust_brkout_brkcmd", lonctrl_thrust_brkout_brkcmd);
        fun("lonctrl_analog_autput_throtcmd", lonctrl_analog_autput_throtcmd);
        fun("lonctrl_analog_autput_brkcmd", lonctrl_analog_autput_brkcmd);
        fun("lonctrl_vel_mrac_cmd", lonctrl_vel_mrac_cmd);
        fun("lonctrl_vel_reference_model", lonctrl_vel_reference_model);
        fun("lonctrl_accel_eso_cmd", lonctrl_accel_eso_cmd);
        fun("lonctrl_deccel_eso_cmd", lonctrl_deccel_eso_cmd);
        fun("lonctrl_slope_estimate", lonctrl_slope_estimate);
        fun("lonctrl_mass_estimate", lonctrl_mass_estimate);
        fun("lonctrl_error", lonctrl_error);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("lonctrl_modecmd", lonctrl_modecmd);
        fun("lonctrl_resetflag", lonctrl_resetflag);
        fun("lonctrl_sys_poserr", lonctrl_sys_poserr);
        fun("lonctrl_sys_velff", lonctrl_sys_velff);
        fun("lonctrl_sys_accff", lonctrl_sys_accff);
        fun("lonctrl_sys_gearcmd", lonctrl_sys_gearcmd);
        fun("lonctrl_sys_gearena", lonctrl_sys_gearena);
        fun("lonctrl_sys_brkemerg", lonctrl_sys_brkemerg);
        fun("lonctrl_api_poscmd", lonctrl_api_poscmd);
        fun("lonctrl_api_velcmd", lonctrl_api_velcmd);
        fun("lonctrl_api_acccmd", lonctrl_api_acccmd);
        fun("lonctrl_api_thrcmd", lonctrl_api_thrcmd);
        fun("lonctrl_api_brkcmd", lonctrl_api_brkcmd);
        fun("lonctrl_fdbk_vx", lonctrl_fdbk_vx);
        fun("lonctrl_fdbk_ax", lonctrl_fdbk_ax);
        fun("lonctrl_fdbk_pitch", lonctrl_fdbk_pitch);
        fun("lonctrl_fdbk_gear", lonctrl_fdbk_gear);
        fun("lonctrl_fdbk_reverse", lonctrl_fdbk_reverse);
        fun("lonctrl_pos_dyn", lonctrl_pos_dyn);
        fun("lonctrl_posctrl_p", lonctrl_posctrl_p);
        fun("lonctrl_posctrl_i", lonctrl_posctrl_i);
        fun("lonctrl_pos_poserr_filter", lonctrl_pos_poserr_filter);
        fun("lonctrl_pos_pi_velcmd", lonctrl_pos_pi_velcmd);
        fun("lonctrl_pos_output_velcmd", lonctrl_pos_output_velcmd);
        fun("lonctrl_vel_dyn", lonctrl_vel_dyn);
        fun("lonctrl_vel_velcmd_lmt", lonctrl_vel_velcmd_lmt);
        fun("lonctrl_vel_vel_err", lonctrl_vel_vel_err);
        fun("lonctrl_velctrl_p", lonctrl_velctrl_p);
        fun("lonctrl_velctrl_i", lonctrl_velctrl_i);
        fun("lonctrl_vel_pi_acc_cmd", lonctrl_vel_pi_acc_cmd);
        fun("lonctrl_vel_pi_acccmd_filter", lonctrl_vel_pi_acccmd_filter);
        fun("lonctrl_vel_accpitch", lonctrl_vel_accpitch);
        fun("lonctrl_vel_accdamper", lonctrl_vel_accdamper);
        fun("lonctrl_vel_accff_filter", lonctrl_vel_accff_filter);
        fun("lonctrl_vel_output_accCmd", lonctrl_vel_output_accCmd);
        fun("lonctrl_vel_output_accCmd_filter", lonctrl_vel_output_accCmd_filter);
        fun("lonctrl_thrust_thr_dyn", lonctrl_thrust_thr_dyn);
        fun("lonctrl_thrust_thr_accerr", lonctrl_thrust_thr_accerr);
        fun("lonctrl_thrust_brk_dyn", lonctrl_thrust_brk_dyn);
        fun("lonctrl_thrust_brk_accerr", lonctrl_thrust_brk_accerr);
        fun("lonctrl_thrust_fdbk_ax_filter", lonctrl_thrust_fdbk_ax_filter);
        fun("lonctrl_thrust_thr_acc_cmd_filter", lonctrl_thrust_thr_acc_cmd_filter);
        fun("lonctrl_thrust_brk_acc_cmd_filter", lonctrl_thrust_brk_acc_cmd_filter);
        fun("lonctrl_thrustctrl_thr_p", lonctrl_thrustctrl_thr_p);
        fun("lonctrl_thrustctrl_thr_i", lonctrl_thrustctrl_thr_i);
        fun("lonctrl_thrustctrl_brk_p", lonctrl_thrustctrl_brk_p);
        fun("lonctrl_thrustctrl_brk_i", lonctrl_thrustctrl_brk_i);
        fun("lonctrl_thrust_pi_thr_acc_cmd", lonctrl_thrust_pi_thr_acc_cmd);
        fun("lonctrl_thrust_pi_brk_acc_cmd", lonctrl_thrust_pi_brk_acc_cmd);
        fun("lonctrl_thrust_acc_cmd_filter_lmt", lonctrl_thrust_acc_cmd_filter_lmt);
        fun("lonctrl_thrust_acctothr_gain", lonctrl_thrust_acctothr_gain);
        fun("lonctrl_thrust_throut_throcmd", lonctrl_thrust_throut_throcmd);
        fun("lonctrl_thrust_acctobrk_gain", lonctrl_thrust_acctobrk_gain);
        fun("lonctrl_thrust_brkout_brkcmd", lonctrl_thrust_brkout_brkcmd);
        fun("lonctrl_analog_autput_throtcmd", lonctrl_analog_autput_throtcmd);
        fun("lonctrl_analog_autput_brkcmd", lonctrl_analog_autput_brkcmd);
        fun("lonctrl_vel_mrac_cmd", lonctrl_vel_mrac_cmd);
        fun("lonctrl_vel_reference_model", lonctrl_vel_reference_model);
        fun("lonctrl_accel_eso_cmd", lonctrl_accel_eso_cmd);
        fun("lonctrl_deccel_eso_cmd", lonctrl_deccel_eso_cmd);
        fun("lonctrl_slope_estimate", lonctrl_slope_estimate);
        fun("lonctrl_mass_estimate", lonctrl_mass_estimate);
        fun("lonctrl_error", lonctrl_error);
    }

    bool operator==(const ::hozon::soc_mcu::MbdLonCtrlDebug& t) const
    {
        return (lonctrl_modecmd == t.lonctrl_modecmd) && (lonctrl_resetflag == t.lonctrl_resetflag) && (fabs(static_cast<double>(lonctrl_sys_poserr - t.lonctrl_sys_poserr)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_sys_velff - t.lonctrl_sys_velff)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_sys_accff - t.lonctrl_sys_accff)) < DBL_EPSILON) && (lonctrl_sys_gearcmd == t.lonctrl_sys_gearcmd) && (lonctrl_sys_gearena == t.lonctrl_sys_gearena) && (lonctrl_sys_brkemerg == t.lonctrl_sys_brkemerg) && (fabs(static_cast<double>(lonctrl_api_poscmd - t.lonctrl_api_poscmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_api_velcmd - t.lonctrl_api_velcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_api_acccmd - t.lonctrl_api_acccmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_api_thrcmd - t.lonctrl_api_thrcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_api_brkcmd - t.lonctrl_api_brkcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_fdbk_vx - t.lonctrl_fdbk_vx)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_fdbk_ax - t.lonctrl_fdbk_ax)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_fdbk_pitch - t.lonctrl_fdbk_pitch)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_fdbk_gear - t.lonctrl_fdbk_gear)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_fdbk_reverse - t.lonctrl_fdbk_reverse)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_pos_dyn - t.lonctrl_pos_dyn)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_posctrl_p - t.lonctrl_posctrl_p)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_posctrl_i - t.lonctrl_posctrl_i)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_pos_poserr_filter - t.lonctrl_pos_poserr_filter)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_pos_pi_velcmd - t.lonctrl_pos_pi_velcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_pos_output_velcmd - t.lonctrl_pos_output_velcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_dyn - t.lonctrl_vel_dyn)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_velcmd_lmt - t.lonctrl_vel_velcmd_lmt)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_vel_err - t.lonctrl_vel_vel_err)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_velctrl_p - t.lonctrl_velctrl_p)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_velctrl_i - t.lonctrl_velctrl_i)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_pi_acc_cmd - t.lonctrl_vel_pi_acc_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_pi_acccmd_filter - t.lonctrl_vel_pi_acccmd_filter)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_accpitch - t.lonctrl_vel_accpitch)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_accdamper - t.lonctrl_vel_accdamper)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_accff_filter - t.lonctrl_vel_accff_filter)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_output_accCmd - t.lonctrl_vel_output_accCmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_output_accCmd_filter - t.lonctrl_vel_output_accCmd_filter)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_thr_dyn - t.lonctrl_thrust_thr_dyn)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_thr_accerr - t.lonctrl_thrust_thr_accerr)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_brk_dyn - t.lonctrl_thrust_brk_dyn)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_brk_accerr - t.lonctrl_thrust_brk_accerr)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_fdbk_ax_filter - t.lonctrl_thrust_fdbk_ax_filter)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_thr_acc_cmd_filter - t.lonctrl_thrust_thr_acc_cmd_filter)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_brk_acc_cmd_filter - t.lonctrl_thrust_brk_acc_cmd_filter)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrustctrl_thr_p - t.lonctrl_thrustctrl_thr_p)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrustctrl_thr_i - t.lonctrl_thrustctrl_thr_i)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrustctrl_brk_p - t.lonctrl_thrustctrl_brk_p)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrustctrl_brk_i - t.lonctrl_thrustctrl_brk_i)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_pi_thr_acc_cmd - t.lonctrl_thrust_pi_thr_acc_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_pi_brk_acc_cmd - t.lonctrl_thrust_pi_brk_acc_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_acc_cmd_filter_lmt - t.lonctrl_thrust_acc_cmd_filter_lmt)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_acctothr_gain - t.lonctrl_thrust_acctothr_gain)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_throut_throcmd - t.lonctrl_thrust_throut_throcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_acctobrk_gain - t.lonctrl_thrust_acctobrk_gain)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_thrust_brkout_brkcmd - t.lonctrl_thrust_brkout_brkcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_analog_autput_throtcmd - t.lonctrl_analog_autput_throtcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_analog_autput_brkcmd - t.lonctrl_analog_autput_brkcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_mrac_cmd - t.lonctrl_vel_mrac_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_vel_reference_model - t.lonctrl_vel_reference_model)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_accel_eso_cmd - t.lonctrl_accel_eso_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_deccel_eso_cmd - t.lonctrl_deccel_eso_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_slope_estimate - t.lonctrl_slope_estimate)) < DBL_EPSILON) && (fabs(static_cast<double>(lonctrl_mass_estimate - t.lonctrl_mass_estimate)) < DBL_EPSILON) && (lonctrl_error == t.lonctrl_error);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MBDLONCTRLDEBUG_H
