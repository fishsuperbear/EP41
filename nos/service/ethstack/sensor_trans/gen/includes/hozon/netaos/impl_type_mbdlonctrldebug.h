/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file impl_type_mbdlonctrldebug.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_MBDLONCTRLDEBUG_H_
#define HOZON_NETAOS_IMPL_TYPE_MBDLONCTRLDEBUG_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct MbdLonCtrlDebug {
    std::uint8_t lonctrl_modecmd;
    std::uint32_t lonctrl_resetflag;
    float lonctrl_sys_poserr;
    float lonctrl_sys_velff;
    float lonctrl_sys_accff;
    std::uint32_t lonctrl_sys_gearcmd;
    std::uint32_t lonctrl_sys_gearena;
    std::uint32_t lonctrl_sys_brkemerg;
    float lonctrl_api_poscmd;
    float lonctrl_api_velcmd;
    float lonctrl_api_acccmd;
    float lonctrl_api_thrcmd;
    float lonctrl_api_brkcmd;
    float lonctrl_fdbk_vx;
    float lonctrl_fdbk_ax;
    float lonctrl_fdbk_pitch;
    float lonctrl_fdbk_gear;
    float lonctrl_fdbk_reverse;
    float lonctrl_pos_dyn;
    float lonctrl_posctrl_p;
    float lonctrl_posctrl_i;
    float lonctrl_pos_poserr_filter;
    float lonctrl_pos_pi_velcmd;
    float lonctrl_pos_output_velcmd;
    float lonctrl_vel_dyn;
    float lonctrl_vel_velcmd_lmt;
    float lonctrl_vel_vel_err;
    float lonctrl_velctrl_p;
    float lonctrl_velctrl_i;
    float lonctrl_vel_pi_acc_cmd;
    float lonctrl_vel_pi_acccmd_filter;
    float lonctrl_vel_accpitch;
    float lonctrl_vel_accdamper;
    float lonctrl_vel_accff_filter;
    float lonctrl_vel_output_accCmd;
    float lonctrl_vel_output_accCmd_filter;
    float lonctrl_thrust_thr_dyn;
    float lonctrl_thrust_thr_accerr;
    float lonctrl_thrust_brk_dyn;
    float lonctrl_thrust_brk_accerr;
    float lonctrl_thrust_fdbk_ax_filter;
    float lonctrl_thrust_thr_acc_cmd_filter;
    float lonctrl_thrust_brk_acc_cmd_filter;
    float lonctrl_thrustctrl_thr_p;
    float lonctrl_thrustctrl_thr_i;
    float lonctrl_thrustctrl_brk_p;
    float lonctrl_thrustctrl_brk_i;
    float lonctrl_thrust_pi_thr_acc_cmd;
    float lonctrl_thrust_pi_brk_acc_cmd;
    float lonctrl_thrust_acc_cmd_filter_lmt;
    float lonctrl_thrust_acctothr_gain;
    float lonctrl_thrust_throut_throcmd;
    float lonctrl_thrust_acctobrk_gain;
    float lonctrl_thrust_brkout_brkcmd;
    float lonctrl_analog_autput_throtcmd;
    float lonctrl_analog_autput_brkcmd;
    float lonctrl_vel_mrac_cmd;
    float lonctrl_vel_reference_model;
    float lonctrl_accel_eso_cmd;
    float lonctrl_deccel_eso_cmd;
    float lonctrl_slope_estimate;
    float lonctrl_mass_estimate;
    std::uint8_t lonctrl_error;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::MbdLonCtrlDebug,lonctrl_modecmd,lonctrl_resetflag,lonctrl_sys_poserr,lonctrl_sys_velff,lonctrl_sys_accff,lonctrl_sys_gearcmd,lonctrl_sys_gearena,lonctrl_sys_brkemerg,lonctrl_api_poscmd,lonctrl_api_velcmd,lonctrl_api_acccmd,lonctrl_api_thrcmd,lonctrl_api_brkcmd,lonctrl_fdbk_vx,lonctrl_fdbk_ax,lonctrl_fdbk_pitch,lonctrl_fdbk_gear,lonctrl_fdbk_reverse,lonctrl_pos_dyn,lonctrl_posctrl_p,lonctrl_posctrl_i,lonctrl_pos_poserr_filter,lonctrl_pos_pi_velcmd,lonctrl_pos_output_velcmd,lonctrl_vel_dyn,lonctrl_vel_velcmd_lmt,lonctrl_vel_vel_err,lonctrl_velctrl_p,lonctrl_velctrl_i,lonctrl_vel_pi_acc_cmd,lonctrl_vel_pi_acccmd_filter,lonctrl_vel_accpitch,lonctrl_vel_accdamper,lonctrl_vel_accff_filter,lonctrl_vel_output_accCmd,lonctrl_vel_output_accCmd_filter,lonctrl_thrust_thr_dyn,lonctrl_thrust_thr_accerr,lonctrl_thrust_brk_dyn,lonctrl_thrust_brk_accerr,lonctrl_thrust_fdbk_ax_filter,lonctrl_thrust_thr_acc_cmd_filter,lonctrl_thrust_brk_acc_cmd_filter,lonctrl_thrustctrl_thr_p,lonctrl_thrustctrl_thr_i,lonctrl_thrustctrl_brk_p,lonctrl_thrustctrl_brk_i,lonctrl_thrust_pi_thr_acc_cmd,lonctrl_thrust_pi_brk_acc_cmd,lonctrl_thrust_acc_cmd_filter_lmt,lonctrl_thrust_acctothr_gain,lonctrl_thrust_throut_throcmd,lonctrl_thrust_acctobrk_gain,lonctrl_thrust_brkout_brkcmd,lonctrl_analog_autput_throtcmd,lonctrl_analog_autput_brkcmd,lonctrl_vel_mrac_cmd,lonctrl_vel_reference_model,lonctrl_accel_eso_cmd,lonctrl_deccel_eso_cmd,lonctrl_slope_estimate,lonctrl_mass_estimate,lonctrl_error);

#endif // HOZON_NETAOS_IMPL_TYPE_MBDLONCTRLDEBUG_H_
/* EOF */