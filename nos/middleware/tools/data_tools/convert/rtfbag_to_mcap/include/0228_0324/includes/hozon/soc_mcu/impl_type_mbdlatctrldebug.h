/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MBDLATCTRLDEBUG_H
#define HOZON_SOC_MCU_IMPL_TYPE_MBDLATCTRLDEBUG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "impl_type_float.h"

namespace hozon {
namespace soc_mcu {
struct MbdLatCtrlDebug {
    ::UInt8 latctrl_modecmd;
    ::UInt32 latctrl_resetflag;
    ::Float latctrl_sys_poserr;
    ::Float latctrl_sys_yawff;
    ::Float latctrl_sys_velff;
    ::Float latctrl_sys_curvff;
    ::Float latctrl_api_poscmd;
    ::Float latctrl_api_yawcmd;
    ::Float latctrl_api_curvcmd;
    ::Float latctrl_api_steercmd;
    ::Float latictrl_fdbk_vxb;
    ::Float latictrl_fdbk_heading;
    ::Float latictrl_fdbk_yawrate;
    ::Float latictrl_fdbk_steer;
    ::UInt8 latictrl_fdbk_gear;
    ::UInt32 latictrl_fdbk_rvsflag;
    ::Float latictrl_offset_dyn;
    ::Float latictrl_offset_offseterr;
    ::Float latictrl_offset_pi_torscmd;
    ::Float latictrl_offset_torsrateffcmd;
    ::Float latictrl_offset_output_yawcmd;
    ::Float latictrl_offsetctrl_i;
    ::Float latictrl_offsetctrl_p;
    ::Float latictrl_tors_dyn;
    ::Float latictrl_tors_pure_yawerr;
    ::Float latictrl_tors_yawerr;
    ::Float latictrl_yawctrl_p;
    ::Float latictrl_yawctrl_i;
    ::Float latictrl_tors_pi_torsrate;
    ::Float latictrl_tors_pi_leadfilter_torsrate;
    ::Float latictrl_tors_torsrateff;
    ::Float latictrl_tors_output_yawratecmd;
    ::Float latictrl_rate_dyn;
    ::Float latictrl_rate_p;
    ::Float latictrl_rate_i;
    ::Float latictrl_rate_yawratecmd_lmt;
    ::Float latictrl_rate_filter_yawratecmd_lmt;
    ::Float latictrl_rate_pi_steer;
    ::Float latictrl_rate_pi_filter_steer;
    ::Float latictrl_rate_steerff;
    ::Float latictrl_rate_output_front_steercmd;
    ::Float latictrl_rate_output_front_steercmd_offset;
    ::Float latictrl_rate_output_sw_steercmd;
    ::Float latictrl_steer_steercmd_filter;
    ::Float latictrl_steer_max_steerrate_value;
    ::Float latictrl_steer_steercmd_lmt_filter;
    ::Float latictrl_steer_output_steercmd;
    ::Float latictrl_yaw_curve_compsate;
    ::Float latictrl_rate_reference_model;
    ::Float latictrl_rate_mrac_cmd;
    ::Float latictrl_rate_eso_cmd;
    ::Float latictrl_rate_steer_offset;
    ::Float latictrl_rate_ramp_estimate;
    ::UInt8 latictrl_error;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(latctrl_modecmd);
        fun(latctrl_resetflag);
        fun(latctrl_sys_poserr);
        fun(latctrl_sys_yawff);
        fun(latctrl_sys_velff);
        fun(latctrl_sys_curvff);
        fun(latctrl_api_poscmd);
        fun(latctrl_api_yawcmd);
        fun(latctrl_api_curvcmd);
        fun(latctrl_api_steercmd);
        fun(latictrl_fdbk_vxb);
        fun(latictrl_fdbk_heading);
        fun(latictrl_fdbk_yawrate);
        fun(latictrl_fdbk_steer);
        fun(latictrl_fdbk_gear);
        fun(latictrl_fdbk_rvsflag);
        fun(latictrl_offset_dyn);
        fun(latictrl_offset_offseterr);
        fun(latictrl_offset_pi_torscmd);
        fun(latictrl_offset_torsrateffcmd);
        fun(latictrl_offset_output_yawcmd);
        fun(latictrl_offsetctrl_i);
        fun(latictrl_offsetctrl_p);
        fun(latictrl_tors_dyn);
        fun(latictrl_tors_pure_yawerr);
        fun(latictrl_tors_yawerr);
        fun(latictrl_yawctrl_p);
        fun(latictrl_yawctrl_i);
        fun(latictrl_tors_pi_torsrate);
        fun(latictrl_tors_pi_leadfilter_torsrate);
        fun(latictrl_tors_torsrateff);
        fun(latictrl_tors_output_yawratecmd);
        fun(latictrl_rate_dyn);
        fun(latictrl_rate_p);
        fun(latictrl_rate_i);
        fun(latictrl_rate_yawratecmd_lmt);
        fun(latictrl_rate_filter_yawratecmd_lmt);
        fun(latictrl_rate_pi_steer);
        fun(latictrl_rate_pi_filter_steer);
        fun(latictrl_rate_steerff);
        fun(latictrl_rate_output_front_steercmd);
        fun(latictrl_rate_output_front_steercmd_offset);
        fun(latictrl_rate_output_sw_steercmd);
        fun(latictrl_steer_steercmd_filter);
        fun(latictrl_steer_max_steerrate_value);
        fun(latictrl_steer_steercmd_lmt_filter);
        fun(latictrl_steer_output_steercmd);
        fun(latictrl_yaw_curve_compsate);
        fun(latictrl_rate_reference_model);
        fun(latictrl_rate_mrac_cmd);
        fun(latictrl_rate_eso_cmd);
        fun(latictrl_rate_steer_offset);
        fun(latictrl_rate_ramp_estimate);
        fun(latictrl_error);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(latctrl_modecmd);
        fun(latctrl_resetflag);
        fun(latctrl_sys_poserr);
        fun(latctrl_sys_yawff);
        fun(latctrl_sys_velff);
        fun(latctrl_sys_curvff);
        fun(latctrl_api_poscmd);
        fun(latctrl_api_yawcmd);
        fun(latctrl_api_curvcmd);
        fun(latctrl_api_steercmd);
        fun(latictrl_fdbk_vxb);
        fun(latictrl_fdbk_heading);
        fun(latictrl_fdbk_yawrate);
        fun(latictrl_fdbk_steer);
        fun(latictrl_fdbk_gear);
        fun(latictrl_fdbk_rvsflag);
        fun(latictrl_offset_dyn);
        fun(latictrl_offset_offseterr);
        fun(latictrl_offset_pi_torscmd);
        fun(latictrl_offset_torsrateffcmd);
        fun(latictrl_offset_output_yawcmd);
        fun(latictrl_offsetctrl_i);
        fun(latictrl_offsetctrl_p);
        fun(latictrl_tors_dyn);
        fun(latictrl_tors_pure_yawerr);
        fun(latictrl_tors_yawerr);
        fun(latictrl_yawctrl_p);
        fun(latictrl_yawctrl_i);
        fun(latictrl_tors_pi_torsrate);
        fun(latictrl_tors_pi_leadfilter_torsrate);
        fun(latictrl_tors_torsrateff);
        fun(latictrl_tors_output_yawratecmd);
        fun(latictrl_rate_dyn);
        fun(latictrl_rate_p);
        fun(latictrl_rate_i);
        fun(latictrl_rate_yawratecmd_lmt);
        fun(latictrl_rate_filter_yawratecmd_lmt);
        fun(latictrl_rate_pi_steer);
        fun(latictrl_rate_pi_filter_steer);
        fun(latictrl_rate_steerff);
        fun(latictrl_rate_output_front_steercmd);
        fun(latictrl_rate_output_front_steercmd_offset);
        fun(latictrl_rate_output_sw_steercmd);
        fun(latictrl_steer_steercmd_filter);
        fun(latictrl_steer_max_steerrate_value);
        fun(latictrl_steer_steercmd_lmt_filter);
        fun(latictrl_steer_output_steercmd);
        fun(latictrl_yaw_curve_compsate);
        fun(latictrl_rate_reference_model);
        fun(latictrl_rate_mrac_cmd);
        fun(latictrl_rate_eso_cmd);
        fun(latictrl_rate_steer_offset);
        fun(latictrl_rate_ramp_estimate);
        fun(latictrl_error);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("latctrl_modecmd", latctrl_modecmd);
        fun("latctrl_resetflag", latctrl_resetflag);
        fun("latctrl_sys_poserr", latctrl_sys_poserr);
        fun("latctrl_sys_yawff", latctrl_sys_yawff);
        fun("latctrl_sys_velff", latctrl_sys_velff);
        fun("latctrl_sys_curvff", latctrl_sys_curvff);
        fun("latctrl_api_poscmd", latctrl_api_poscmd);
        fun("latctrl_api_yawcmd", latctrl_api_yawcmd);
        fun("latctrl_api_curvcmd", latctrl_api_curvcmd);
        fun("latctrl_api_steercmd", latctrl_api_steercmd);
        fun("latictrl_fdbk_vxb", latictrl_fdbk_vxb);
        fun("latictrl_fdbk_heading", latictrl_fdbk_heading);
        fun("latictrl_fdbk_yawrate", latictrl_fdbk_yawrate);
        fun("latictrl_fdbk_steer", latictrl_fdbk_steer);
        fun("latictrl_fdbk_gear", latictrl_fdbk_gear);
        fun("latictrl_fdbk_rvsflag", latictrl_fdbk_rvsflag);
        fun("latictrl_offset_dyn", latictrl_offset_dyn);
        fun("latictrl_offset_offseterr", latictrl_offset_offseterr);
        fun("latictrl_offset_pi_torscmd", latictrl_offset_pi_torscmd);
        fun("latictrl_offset_torsrateffcmd", latictrl_offset_torsrateffcmd);
        fun("latictrl_offset_output_yawcmd", latictrl_offset_output_yawcmd);
        fun("latictrl_offsetctrl_i", latictrl_offsetctrl_i);
        fun("latictrl_offsetctrl_p", latictrl_offsetctrl_p);
        fun("latictrl_tors_dyn", latictrl_tors_dyn);
        fun("latictrl_tors_pure_yawerr", latictrl_tors_pure_yawerr);
        fun("latictrl_tors_yawerr", latictrl_tors_yawerr);
        fun("latictrl_yawctrl_p", latictrl_yawctrl_p);
        fun("latictrl_yawctrl_i", latictrl_yawctrl_i);
        fun("latictrl_tors_pi_torsrate", latictrl_tors_pi_torsrate);
        fun("latictrl_tors_pi_leadfilter_torsrate", latictrl_tors_pi_leadfilter_torsrate);
        fun("latictrl_tors_torsrateff", latictrl_tors_torsrateff);
        fun("latictrl_tors_output_yawratecmd", latictrl_tors_output_yawratecmd);
        fun("latictrl_rate_dyn", latictrl_rate_dyn);
        fun("latictrl_rate_p", latictrl_rate_p);
        fun("latictrl_rate_i", latictrl_rate_i);
        fun("latictrl_rate_yawratecmd_lmt", latictrl_rate_yawratecmd_lmt);
        fun("latictrl_rate_filter_yawratecmd_lmt", latictrl_rate_filter_yawratecmd_lmt);
        fun("latictrl_rate_pi_steer", latictrl_rate_pi_steer);
        fun("latictrl_rate_pi_filter_steer", latictrl_rate_pi_filter_steer);
        fun("latictrl_rate_steerff", latictrl_rate_steerff);
        fun("latictrl_rate_output_front_steercmd", latictrl_rate_output_front_steercmd);
        fun("latictrl_rate_output_front_steercmd_offset", latictrl_rate_output_front_steercmd_offset);
        fun("latictrl_rate_output_sw_steercmd", latictrl_rate_output_sw_steercmd);
        fun("latictrl_steer_steercmd_filter", latictrl_steer_steercmd_filter);
        fun("latictrl_steer_max_steerrate_value", latictrl_steer_max_steerrate_value);
        fun("latictrl_steer_steercmd_lmt_filter", latictrl_steer_steercmd_lmt_filter);
        fun("latictrl_steer_output_steercmd", latictrl_steer_output_steercmd);
        fun("latictrl_yaw_curve_compsate", latictrl_yaw_curve_compsate);
        fun("latictrl_rate_reference_model", latictrl_rate_reference_model);
        fun("latictrl_rate_mrac_cmd", latictrl_rate_mrac_cmd);
        fun("latictrl_rate_eso_cmd", latictrl_rate_eso_cmd);
        fun("latictrl_rate_steer_offset", latictrl_rate_steer_offset);
        fun("latictrl_rate_ramp_estimate", latictrl_rate_ramp_estimate);
        fun("latictrl_error", latictrl_error);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("latctrl_modecmd", latctrl_modecmd);
        fun("latctrl_resetflag", latctrl_resetflag);
        fun("latctrl_sys_poserr", latctrl_sys_poserr);
        fun("latctrl_sys_yawff", latctrl_sys_yawff);
        fun("latctrl_sys_velff", latctrl_sys_velff);
        fun("latctrl_sys_curvff", latctrl_sys_curvff);
        fun("latctrl_api_poscmd", latctrl_api_poscmd);
        fun("latctrl_api_yawcmd", latctrl_api_yawcmd);
        fun("latctrl_api_curvcmd", latctrl_api_curvcmd);
        fun("latctrl_api_steercmd", latctrl_api_steercmd);
        fun("latictrl_fdbk_vxb", latictrl_fdbk_vxb);
        fun("latictrl_fdbk_heading", latictrl_fdbk_heading);
        fun("latictrl_fdbk_yawrate", latictrl_fdbk_yawrate);
        fun("latictrl_fdbk_steer", latictrl_fdbk_steer);
        fun("latictrl_fdbk_gear", latictrl_fdbk_gear);
        fun("latictrl_fdbk_rvsflag", latictrl_fdbk_rvsflag);
        fun("latictrl_offset_dyn", latictrl_offset_dyn);
        fun("latictrl_offset_offseterr", latictrl_offset_offseterr);
        fun("latictrl_offset_pi_torscmd", latictrl_offset_pi_torscmd);
        fun("latictrl_offset_torsrateffcmd", latictrl_offset_torsrateffcmd);
        fun("latictrl_offset_output_yawcmd", latictrl_offset_output_yawcmd);
        fun("latictrl_offsetctrl_i", latictrl_offsetctrl_i);
        fun("latictrl_offsetctrl_p", latictrl_offsetctrl_p);
        fun("latictrl_tors_dyn", latictrl_tors_dyn);
        fun("latictrl_tors_pure_yawerr", latictrl_tors_pure_yawerr);
        fun("latictrl_tors_yawerr", latictrl_tors_yawerr);
        fun("latictrl_yawctrl_p", latictrl_yawctrl_p);
        fun("latictrl_yawctrl_i", latictrl_yawctrl_i);
        fun("latictrl_tors_pi_torsrate", latictrl_tors_pi_torsrate);
        fun("latictrl_tors_pi_leadfilter_torsrate", latictrl_tors_pi_leadfilter_torsrate);
        fun("latictrl_tors_torsrateff", latictrl_tors_torsrateff);
        fun("latictrl_tors_output_yawratecmd", latictrl_tors_output_yawratecmd);
        fun("latictrl_rate_dyn", latictrl_rate_dyn);
        fun("latictrl_rate_p", latictrl_rate_p);
        fun("latictrl_rate_i", latictrl_rate_i);
        fun("latictrl_rate_yawratecmd_lmt", latictrl_rate_yawratecmd_lmt);
        fun("latictrl_rate_filter_yawratecmd_lmt", latictrl_rate_filter_yawratecmd_lmt);
        fun("latictrl_rate_pi_steer", latictrl_rate_pi_steer);
        fun("latictrl_rate_pi_filter_steer", latictrl_rate_pi_filter_steer);
        fun("latictrl_rate_steerff", latictrl_rate_steerff);
        fun("latictrl_rate_output_front_steercmd", latictrl_rate_output_front_steercmd);
        fun("latictrl_rate_output_front_steercmd_offset", latictrl_rate_output_front_steercmd_offset);
        fun("latictrl_rate_output_sw_steercmd", latictrl_rate_output_sw_steercmd);
        fun("latictrl_steer_steercmd_filter", latictrl_steer_steercmd_filter);
        fun("latictrl_steer_max_steerrate_value", latictrl_steer_max_steerrate_value);
        fun("latictrl_steer_steercmd_lmt_filter", latictrl_steer_steercmd_lmt_filter);
        fun("latictrl_steer_output_steercmd", latictrl_steer_output_steercmd);
        fun("latictrl_yaw_curve_compsate", latictrl_yaw_curve_compsate);
        fun("latictrl_rate_reference_model", latictrl_rate_reference_model);
        fun("latictrl_rate_mrac_cmd", latictrl_rate_mrac_cmd);
        fun("latictrl_rate_eso_cmd", latictrl_rate_eso_cmd);
        fun("latictrl_rate_steer_offset", latictrl_rate_steer_offset);
        fun("latictrl_rate_ramp_estimate", latictrl_rate_ramp_estimate);
        fun("latictrl_error", latictrl_error);
    }

    bool operator==(const ::hozon::soc_mcu::MbdLatCtrlDebug& t) const
    {
        return (latctrl_modecmd == t.latctrl_modecmd) && (latctrl_resetflag == t.latctrl_resetflag) && (fabs(static_cast<double>(latctrl_sys_poserr - t.latctrl_sys_poserr)) < DBL_EPSILON) && (fabs(static_cast<double>(latctrl_sys_yawff - t.latctrl_sys_yawff)) < DBL_EPSILON) && (fabs(static_cast<double>(latctrl_sys_velff - t.latctrl_sys_velff)) < DBL_EPSILON) && (fabs(static_cast<double>(latctrl_sys_curvff - t.latctrl_sys_curvff)) < DBL_EPSILON) && (fabs(static_cast<double>(latctrl_api_poscmd - t.latctrl_api_poscmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latctrl_api_yawcmd - t.latctrl_api_yawcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latctrl_api_curvcmd - t.latctrl_api_curvcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latctrl_api_steercmd - t.latctrl_api_steercmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_fdbk_vxb - t.latictrl_fdbk_vxb)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_fdbk_heading - t.latictrl_fdbk_heading)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_fdbk_yawrate - t.latictrl_fdbk_yawrate)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_fdbk_steer - t.latictrl_fdbk_steer)) < DBL_EPSILON) && (latictrl_fdbk_gear == t.latictrl_fdbk_gear) && (latictrl_fdbk_rvsflag == t.latictrl_fdbk_rvsflag) && (fabs(static_cast<double>(latictrl_offset_dyn - t.latictrl_offset_dyn)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_offset_offseterr - t.latictrl_offset_offseterr)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_offset_pi_torscmd - t.latictrl_offset_pi_torscmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_offset_torsrateffcmd - t.latictrl_offset_torsrateffcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_offset_output_yawcmd - t.latictrl_offset_output_yawcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_offsetctrl_i - t.latictrl_offsetctrl_i)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_offsetctrl_p - t.latictrl_offsetctrl_p)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_tors_dyn - t.latictrl_tors_dyn)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_tors_pure_yawerr - t.latictrl_tors_pure_yawerr)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_tors_yawerr - t.latictrl_tors_yawerr)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_yawctrl_p - t.latictrl_yawctrl_p)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_yawctrl_i - t.latictrl_yawctrl_i)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_tors_pi_torsrate - t.latictrl_tors_pi_torsrate)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_tors_pi_leadfilter_torsrate - t.latictrl_tors_pi_leadfilter_torsrate)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_tors_torsrateff - t.latictrl_tors_torsrateff)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_tors_output_yawratecmd - t.latictrl_tors_output_yawratecmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_dyn - t.latictrl_rate_dyn)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_p - t.latictrl_rate_p)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_i - t.latictrl_rate_i)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_yawratecmd_lmt - t.latictrl_rate_yawratecmd_lmt)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_filter_yawratecmd_lmt - t.latictrl_rate_filter_yawratecmd_lmt)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_pi_steer - t.latictrl_rate_pi_steer)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_pi_filter_steer - t.latictrl_rate_pi_filter_steer)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_steerff - t.latictrl_rate_steerff)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_output_front_steercmd - t.latictrl_rate_output_front_steercmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_output_front_steercmd_offset - t.latictrl_rate_output_front_steercmd_offset)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_output_sw_steercmd - t.latictrl_rate_output_sw_steercmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_steer_steercmd_filter - t.latictrl_steer_steercmd_filter)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_steer_max_steerrate_value - t.latictrl_steer_max_steerrate_value)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_steer_steercmd_lmt_filter - t.latictrl_steer_steercmd_lmt_filter)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_steer_output_steercmd - t.latictrl_steer_output_steercmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_yaw_curve_compsate - t.latictrl_yaw_curve_compsate)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_reference_model - t.latictrl_rate_reference_model)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_mrac_cmd - t.latictrl_rate_mrac_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_eso_cmd - t.latictrl_rate_eso_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_steer_offset - t.latictrl_rate_steer_offset)) < DBL_EPSILON) && (fabs(static_cast<double>(latictrl_rate_ramp_estimate - t.latictrl_rate_ramp_estimate)) < DBL_EPSILON) && (latictrl_error == t.latictrl_error);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MBDLATCTRLDEBUG_H
