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
 * @file impl_type_mbdctrldecdebug.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_MBDCTRLDECDEBUG_H_
#define HOZON_NETAOS_IMPL_TYPE_MBDCTRLDECDEBUG_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct MbdCtrlDecDebug {
    std::uint32_t ctrldec_sysmode;
    bool ctrldec_req_auto;
    bool ctrldec_automode_is_estop;
    bool ctrldec_is_gear_change_req;
    bool ctrldec_is_stop_steer_ctrl;
    bool ctrldec_is_vehicle_standstill;
    bool ctrldec_is_steer_set_ok;
    std::uint32_t ctrldec_trajcalc_geargmd;
    std::uint32_t ctrldec_lat_sysmodecmd;
    std::uint32_t ctrldec_lat_resetflag;
    float ctrldec_lat_sys_poserr;
    float ctrldec_lat_sys_yawff;
    float ctrldec_lat_sys_velff;
    float ctrldec_lat_sys_curvff;
    float ctrldec_lat_api_poscmd;
    float ctrldec_lat_api_yawcmd;
    float ctrldec_lat_api_curvcmd;
    float ctrldec_lat_api_steercmd;
    std::uint32_t ctrldec_lon_sysmodecmd;
    std::uint32_t ctrldec_lon_resetflag;
    float ctrldec_lon_sys_poserr;
    float ctrldec_lon_sys_velff;
    float ctrldec_lon_sys_accff;
    std::uint32_t ctrldec_lon_sys_gearcmd;
    std::uint32_t ctrldec_lon_sys_gear_ena;
    std::uint32_t ctrldec_lon_sys_brk_emerg;
    float ctrldec_lon_api_poscmd;
    float ctrldec_lon_api_velff;
    float ctrldec_lon_api_acc_cmd;
    float ctrldec_lon_api_thrcmd;
    float ctrldec_lon_api_brkcmd;
    std::uint32_t ctrldec_adascalc_geargmd;
    std::uint8_t ctrldec_ctrl_err;
    std::uint8_t ctrldec_actor_error;
    std::uint8_t ctrldec_sensor_error;
    std::uint8_t ctrldec_algorithm_error;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::MbdCtrlDecDebug,ctrldec_sysmode,ctrldec_req_auto,ctrldec_automode_is_estop,ctrldec_is_gear_change_req,ctrldec_is_stop_steer_ctrl,ctrldec_is_vehicle_standstill,ctrldec_is_steer_set_ok,ctrldec_trajcalc_geargmd,ctrldec_lat_sysmodecmd,ctrldec_lat_resetflag,ctrldec_lat_sys_poserr,ctrldec_lat_sys_yawff,ctrldec_lat_sys_velff,ctrldec_lat_sys_curvff,ctrldec_lat_api_poscmd,ctrldec_lat_api_yawcmd,ctrldec_lat_api_curvcmd,ctrldec_lat_api_steercmd,ctrldec_lon_sysmodecmd,ctrldec_lon_resetflag,ctrldec_lon_sys_poserr,ctrldec_lon_sys_velff,ctrldec_lon_sys_accff,ctrldec_lon_sys_gearcmd,ctrldec_lon_sys_gear_ena,ctrldec_lon_sys_brk_emerg,ctrldec_lon_api_poscmd,ctrldec_lon_api_velff,ctrldec_lon_api_acc_cmd,ctrldec_lon_api_thrcmd,ctrldec_lon_api_brkcmd,ctrldec_adascalc_geargmd,ctrldec_ctrl_err,ctrldec_actor_error,ctrldec_sensor_error,ctrldec_algorithm_error);

#endif // HOZON_NETAOS_IMPL_TYPE_MBDCTRLDECDEBUG_H_
/* EOF */