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
 * @file impl_type_mbdlatctrldebug.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_MBDLATCTRLDEBUG_H_
#define HOZON_NETAOS_IMPL_TYPE_MBDLATCTRLDEBUG_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct MbdLatCtrlDebug {
    std::uint8_t latctrl_modecmd;
    std::uint32_t latctrl_resetflag;
    float latctrl_sys_poserr;
    float latctrl_sys_yawff;
    float latctrl_sys_velff;
    float latctrl_sys_curvff;
    float latctrl_api_poscmd;
    float latctrl_api_yawcmd;
    float latctrl_api_curvcmd;
    float latctrl_api_steercmd;
    float latictrl_fdbk_vxb;
    float latictrl_fdbk_heading;
    float latictrl_fdbk_yawrate;
    float latictrl_fdbk_steer;
    std::uint8_t latictrl_fdbk_gear;
    std::uint32_t latictrl_fdbk_rvsflag;
    float latictrl_offset_dyn;
    float latictrl_offset_offseterr;
    float latictrl_offset_pi_torscmd;
    float latictrl_offset_torsrateffcmd;
    float latictrl_offset_output_yawcmd;
    float latictrl_offsetctrl_i;
    float latictrl_offsetctrl_p;
    float latictrl_tors_dyn;
    float latictrl_tors_pure_yawerr;
    float latictrl_tors_yawerr;
    float latictrl_yawctrl_p;
    float latictrl_yawctrl_i;
    float latictrl_tors_pi_torsrate;
    float latictrl_tors_pi_leadfilter_torsrate;
    float latictrl_tors_torsrateff;
    float latictrl_tors_output_yawratecmd;
    float latictrl_rate_dyn;
    float latictrl_rate_p;
    float latictrl_rate_i;
    float latictrl_rate_yawratecmd_lmt;
    float latictrl_rate_filter_yawratecmd_lmt;
    float latictrl_rate_pi_steer;
    float latictrl_rate_pi_filter_steer;
    float latictrl_rate_steerff;
    float latictrl_rate_output_front_steercmd;
    float latictrl_rate_output_front_steercmd_offset;
    float latictrl_rate_output_sw_steercmd;
    float latictrl_steer_steercmd_filter;
    float latictrl_steer_max_steerrate_value;
    float latictrl_steer_steercmd_lmt_filter;
    float latictrl_steer_output_steercmd;
    float latictrl_yaw_curve_compsate;
    float latictrl_rate_reference_model;
    float latictrl_rate_mrac_cmd;
    float latictrl_rate_eso_cmd;
    float latictrl_rate_steer_offset;
    float latictrl_rate_ramp_estimate;
    std::uint8_t latictrl_error;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::MbdLatCtrlDebug,latctrl_modecmd,latctrl_resetflag,latctrl_sys_poserr,latctrl_sys_yawff,latctrl_sys_velff,latctrl_sys_curvff,latctrl_api_poscmd,latctrl_api_yawcmd,latctrl_api_curvcmd,latctrl_api_steercmd,latictrl_fdbk_vxb,latictrl_fdbk_heading,latictrl_fdbk_yawrate,latictrl_fdbk_steer,latictrl_fdbk_gear,latictrl_fdbk_rvsflag,latictrl_offset_dyn,latictrl_offset_offseterr,latictrl_offset_pi_torscmd,latictrl_offset_torsrateffcmd,latictrl_offset_output_yawcmd,latictrl_offsetctrl_i,latictrl_offsetctrl_p,latictrl_tors_dyn,latictrl_tors_pure_yawerr,latictrl_tors_yawerr,latictrl_yawctrl_p,latictrl_yawctrl_i,latictrl_tors_pi_torsrate,latictrl_tors_pi_leadfilter_torsrate,latictrl_tors_torsrateff,latictrl_tors_output_yawratecmd,latictrl_rate_dyn,latictrl_rate_p,latictrl_rate_i,latictrl_rate_yawratecmd_lmt,latictrl_rate_filter_yawratecmd_lmt,latictrl_rate_pi_steer,latictrl_rate_pi_filter_steer,latictrl_rate_steerff,latictrl_rate_output_front_steercmd,latictrl_rate_output_front_steercmd_offset,latictrl_rate_output_sw_steercmd,latictrl_steer_steercmd_filter,latictrl_steer_max_steerrate_value,latictrl_steer_steercmd_lmt_filter,latictrl_steer_output_steercmd,latictrl_yaw_curve_compsate,latictrl_rate_reference_model,latictrl_rate_mrac_cmd,latictrl_rate_eso_cmd,latictrl_rate_steer_offset,latictrl_rate_ramp_estimate,latictrl_error);

#endif // HOZON_NETAOS_IMPL_TYPE_MBDLATCTRLDEBUG_H_
/* EOF */