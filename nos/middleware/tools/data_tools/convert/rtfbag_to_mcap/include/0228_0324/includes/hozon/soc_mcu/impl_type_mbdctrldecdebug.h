/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MBDCTRLDECDEBUG_H
#define HOZON_SOC_MCU_IMPL_TYPE_MBDCTRLDECDEBUG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_boolean.h"
#include "impl_type_float.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct MbdCtrlDecDebug {
    ::UInt32 ctrldec_sysmode;
    ::Boolean ctrldec_req_auto;
    ::Boolean ctrldec_automode_is_estop;
    ::Boolean ctrldec_is_gear_change_req;
    ::Boolean ctrldec_is_stop_steer_ctrl;
    ::Boolean ctrldec_is_vehicle_standstill;
    ::Boolean ctrldec_is_steer_set_ok;
    ::UInt32 ctrldec_trajcalc_geargmd;
    ::UInt32 ctrldec_lat_sysmodecmd;
    ::UInt32 ctrldec_lat_resetflag;
    ::Float ctrldec_lat_sys_poserr;
    ::Float ctrldec_lat_sys_yawff;
    ::Float ctrldec_lat_sys_velff;
    ::Float ctrldec_lat_sys_curvff;
    ::Float ctrldec_lat_api_poscmd;
    ::Float ctrldec_lat_api_yawcmd;
    ::Float ctrldec_lat_api_curvcmd;
    ::Float ctrldec_lat_api_steercmd;
    ::UInt32 ctrldec_lon_sysmodecmd;
    ::UInt32 ctrldec_lon_resetflag;
    ::Float ctrldec_lon_sys_poserr;
    ::Float ctrldec_lon_sys_velff;
    ::Float ctrldec_lon_sys_accff;
    ::UInt32 ctrldec_lon_sys_gearcmd;
    ::UInt32 ctrldec_lon_sys_gear_ena;
    ::UInt32 ctrldec_lon_sys_brk_emerg;
    ::Float ctrldec_lon_api_poscmd;
    ::Float ctrldec_lon_api_velff;
    ::Float ctrldec_lon_api_acc_cmd;
    ::Float ctrldec_lon_api_thrcmd;
    ::Float ctrldec_lon_api_brkcmd;
    ::UInt32 ctrldec_adascalc_geargmd;
    ::UInt8 ctrldec_ctrl_err;
    ::UInt8 ctrldec_actor_error;
    ::UInt8 ctrldec_sensor_error;
    ::UInt8 ctrldec_algorithm_error;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ctrldec_sysmode);
        fun(ctrldec_req_auto);
        fun(ctrldec_automode_is_estop);
        fun(ctrldec_is_gear_change_req);
        fun(ctrldec_is_stop_steer_ctrl);
        fun(ctrldec_is_vehicle_standstill);
        fun(ctrldec_is_steer_set_ok);
        fun(ctrldec_trajcalc_geargmd);
        fun(ctrldec_lat_sysmodecmd);
        fun(ctrldec_lat_resetflag);
        fun(ctrldec_lat_sys_poserr);
        fun(ctrldec_lat_sys_yawff);
        fun(ctrldec_lat_sys_velff);
        fun(ctrldec_lat_sys_curvff);
        fun(ctrldec_lat_api_poscmd);
        fun(ctrldec_lat_api_yawcmd);
        fun(ctrldec_lat_api_curvcmd);
        fun(ctrldec_lat_api_steercmd);
        fun(ctrldec_lon_sysmodecmd);
        fun(ctrldec_lon_resetflag);
        fun(ctrldec_lon_sys_poserr);
        fun(ctrldec_lon_sys_velff);
        fun(ctrldec_lon_sys_accff);
        fun(ctrldec_lon_sys_gearcmd);
        fun(ctrldec_lon_sys_gear_ena);
        fun(ctrldec_lon_sys_brk_emerg);
        fun(ctrldec_lon_api_poscmd);
        fun(ctrldec_lon_api_velff);
        fun(ctrldec_lon_api_acc_cmd);
        fun(ctrldec_lon_api_thrcmd);
        fun(ctrldec_lon_api_brkcmd);
        fun(ctrldec_adascalc_geargmd);
        fun(ctrldec_ctrl_err);
        fun(ctrldec_actor_error);
        fun(ctrldec_sensor_error);
        fun(ctrldec_algorithm_error);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ctrldec_sysmode);
        fun(ctrldec_req_auto);
        fun(ctrldec_automode_is_estop);
        fun(ctrldec_is_gear_change_req);
        fun(ctrldec_is_stop_steer_ctrl);
        fun(ctrldec_is_vehicle_standstill);
        fun(ctrldec_is_steer_set_ok);
        fun(ctrldec_trajcalc_geargmd);
        fun(ctrldec_lat_sysmodecmd);
        fun(ctrldec_lat_resetflag);
        fun(ctrldec_lat_sys_poserr);
        fun(ctrldec_lat_sys_yawff);
        fun(ctrldec_lat_sys_velff);
        fun(ctrldec_lat_sys_curvff);
        fun(ctrldec_lat_api_poscmd);
        fun(ctrldec_lat_api_yawcmd);
        fun(ctrldec_lat_api_curvcmd);
        fun(ctrldec_lat_api_steercmd);
        fun(ctrldec_lon_sysmodecmd);
        fun(ctrldec_lon_resetflag);
        fun(ctrldec_lon_sys_poserr);
        fun(ctrldec_lon_sys_velff);
        fun(ctrldec_lon_sys_accff);
        fun(ctrldec_lon_sys_gearcmd);
        fun(ctrldec_lon_sys_gear_ena);
        fun(ctrldec_lon_sys_brk_emerg);
        fun(ctrldec_lon_api_poscmd);
        fun(ctrldec_lon_api_velff);
        fun(ctrldec_lon_api_acc_cmd);
        fun(ctrldec_lon_api_thrcmd);
        fun(ctrldec_lon_api_brkcmd);
        fun(ctrldec_adascalc_geargmd);
        fun(ctrldec_ctrl_err);
        fun(ctrldec_actor_error);
        fun(ctrldec_sensor_error);
        fun(ctrldec_algorithm_error);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ctrldec_sysmode", ctrldec_sysmode);
        fun("ctrldec_req_auto", ctrldec_req_auto);
        fun("ctrldec_automode_is_estop", ctrldec_automode_is_estop);
        fun("ctrldec_is_gear_change_req", ctrldec_is_gear_change_req);
        fun("ctrldec_is_stop_steer_ctrl", ctrldec_is_stop_steer_ctrl);
        fun("ctrldec_is_vehicle_standstill", ctrldec_is_vehicle_standstill);
        fun("ctrldec_is_steer_set_ok", ctrldec_is_steer_set_ok);
        fun("ctrldec_trajcalc_geargmd", ctrldec_trajcalc_geargmd);
        fun("ctrldec_lat_sysmodecmd", ctrldec_lat_sysmodecmd);
        fun("ctrldec_lat_resetflag", ctrldec_lat_resetflag);
        fun("ctrldec_lat_sys_poserr", ctrldec_lat_sys_poserr);
        fun("ctrldec_lat_sys_yawff", ctrldec_lat_sys_yawff);
        fun("ctrldec_lat_sys_velff", ctrldec_lat_sys_velff);
        fun("ctrldec_lat_sys_curvff", ctrldec_lat_sys_curvff);
        fun("ctrldec_lat_api_poscmd", ctrldec_lat_api_poscmd);
        fun("ctrldec_lat_api_yawcmd", ctrldec_lat_api_yawcmd);
        fun("ctrldec_lat_api_curvcmd", ctrldec_lat_api_curvcmd);
        fun("ctrldec_lat_api_steercmd", ctrldec_lat_api_steercmd);
        fun("ctrldec_lon_sysmodecmd", ctrldec_lon_sysmodecmd);
        fun("ctrldec_lon_resetflag", ctrldec_lon_resetflag);
        fun("ctrldec_lon_sys_poserr", ctrldec_lon_sys_poserr);
        fun("ctrldec_lon_sys_velff", ctrldec_lon_sys_velff);
        fun("ctrldec_lon_sys_accff", ctrldec_lon_sys_accff);
        fun("ctrldec_lon_sys_gearcmd", ctrldec_lon_sys_gearcmd);
        fun("ctrldec_lon_sys_gear_ena", ctrldec_lon_sys_gear_ena);
        fun("ctrldec_lon_sys_brk_emerg", ctrldec_lon_sys_brk_emerg);
        fun("ctrldec_lon_api_poscmd", ctrldec_lon_api_poscmd);
        fun("ctrldec_lon_api_velff", ctrldec_lon_api_velff);
        fun("ctrldec_lon_api_acc_cmd", ctrldec_lon_api_acc_cmd);
        fun("ctrldec_lon_api_thrcmd", ctrldec_lon_api_thrcmd);
        fun("ctrldec_lon_api_brkcmd", ctrldec_lon_api_brkcmd);
        fun("ctrldec_adascalc_geargmd", ctrldec_adascalc_geargmd);
        fun("ctrldec_ctrl_err", ctrldec_ctrl_err);
        fun("ctrldec_actor_error", ctrldec_actor_error);
        fun("ctrldec_sensor_error", ctrldec_sensor_error);
        fun("ctrldec_algorithm_error", ctrldec_algorithm_error);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ctrldec_sysmode", ctrldec_sysmode);
        fun("ctrldec_req_auto", ctrldec_req_auto);
        fun("ctrldec_automode_is_estop", ctrldec_automode_is_estop);
        fun("ctrldec_is_gear_change_req", ctrldec_is_gear_change_req);
        fun("ctrldec_is_stop_steer_ctrl", ctrldec_is_stop_steer_ctrl);
        fun("ctrldec_is_vehicle_standstill", ctrldec_is_vehicle_standstill);
        fun("ctrldec_is_steer_set_ok", ctrldec_is_steer_set_ok);
        fun("ctrldec_trajcalc_geargmd", ctrldec_trajcalc_geargmd);
        fun("ctrldec_lat_sysmodecmd", ctrldec_lat_sysmodecmd);
        fun("ctrldec_lat_resetflag", ctrldec_lat_resetflag);
        fun("ctrldec_lat_sys_poserr", ctrldec_lat_sys_poserr);
        fun("ctrldec_lat_sys_yawff", ctrldec_lat_sys_yawff);
        fun("ctrldec_lat_sys_velff", ctrldec_lat_sys_velff);
        fun("ctrldec_lat_sys_curvff", ctrldec_lat_sys_curvff);
        fun("ctrldec_lat_api_poscmd", ctrldec_lat_api_poscmd);
        fun("ctrldec_lat_api_yawcmd", ctrldec_lat_api_yawcmd);
        fun("ctrldec_lat_api_curvcmd", ctrldec_lat_api_curvcmd);
        fun("ctrldec_lat_api_steercmd", ctrldec_lat_api_steercmd);
        fun("ctrldec_lon_sysmodecmd", ctrldec_lon_sysmodecmd);
        fun("ctrldec_lon_resetflag", ctrldec_lon_resetflag);
        fun("ctrldec_lon_sys_poserr", ctrldec_lon_sys_poserr);
        fun("ctrldec_lon_sys_velff", ctrldec_lon_sys_velff);
        fun("ctrldec_lon_sys_accff", ctrldec_lon_sys_accff);
        fun("ctrldec_lon_sys_gearcmd", ctrldec_lon_sys_gearcmd);
        fun("ctrldec_lon_sys_gear_ena", ctrldec_lon_sys_gear_ena);
        fun("ctrldec_lon_sys_brk_emerg", ctrldec_lon_sys_brk_emerg);
        fun("ctrldec_lon_api_poscmd", ctrldec_lon_api_poscmd);
        fun("ctrldec_lon_api_velff", ctrldec_lon_api_velff);
        fun("ctrldec_lon_api_acc_cmd", ctrldec_lon_api_acc_cmd);
        fun("ctrldec_lon_api_thrcmd", ctrldec_lon_api_thrcmd);
        fun("ctrldec_lon_api_brkcmd", ctrldec_lon_api_brkcmd);
        fun("ctrldec_adascalc_geargmd", ctrldec_adascalc_geargmd);
        fun("ctrldec_ctrl_err", ctrldec_ctrl_err);
        fun("ctrldec_actor_error", ctrldec_actor_error);
        fun("ctrldec_sensor_error", ctrldec_sensor_error);
        fun("ctrldec_algorithm_error", ctrldec_algorithm_error);
    }

    bool operator==(const ::hozon::soc_mcu::MbdCtrlDecDebug& t) const
    {
        return (ctrldec_sysmode == t.ctrldec_sysmode) && (ctrldec_req_auto == t.ctrldec_req_auto) && (ctrldec_automode_is_estop == t.ctrldec_automode_is_estop) && (ctrldec_is_gear_change_req == t.ctrldec_is_gear_change_req) && (ctrldec_is_stop_steer_ctrl == t.ctrldec_is_stop_steer_ctrl) && (ctrldec_is_vehicle_standstill == t.ctrldec_is_vehicle_standstill) && (ctrldec_is_steer_set_ok == t.ctrldec_is_steer_set_ok) && (ctrldec_trajcalc_geargmd == t.ctrldec_trajcalc_geargmd) && (ctrldec_lat_sysmodecmd == t.ctrldec_lat_sysmodecmd) && (ctrldec_lat_resetflag == t.ctrldec_lat_resetflag) && (fabs(static_cast<double>(ctrldec_lat_sys_poserr - t.ctrldec_lat_sys_poserr)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lat_sys_yawff - t.ctrldec_lat_sys_yawff)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lat_sys_velff - t.ctrldec_lat_sys_velff)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lat_sys_curvff - t.ctrldec_lat_sys_curvff)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lat_api_poscmd - t.ctrldec_lat_api_poscmd)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lat_api_yawcmd - t.ctrldec_lat_api_yawcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lat_api_curvcmd - t.ctrldec_lat_api_curvcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lat_api_steercmd - t.ctrldec_lat_api_steercmd)) < DBL_EPSILON) && (ctrldec_lon_sysmodecmd == t.ctrldec_lon_sysmodecmd) && (ctrldec_lon_resetflag == t.ctrldec_lon_resetflag) && (fabs(static_cast<double>(ctrldec_lon_sys_poserr - t.ctrldec_lon_sys_poserr)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lon_sys_velff - t.ctrldec_lon_sys_velff)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lon_sys_accff - t.ctrldec_lon_sys_accff)) < DBL_EPSILON) && (ctrldec_lon_sys_gearcmd == t.ctrldec_lon_sys_gearcmd) && (ctrldec_lon_sys_gear_ena == t.ctrldec_lon_sys_gear_ena) && (ctrldec_lon_sys_brk_emerg == t.ctrldec_lon_sys_brk_emerg) && (fabs(static_cast<double>(ctrldec_lon_api_poscmd - t.ctrldec_lon_api_poscmd)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lon_api_velff - t.ctrldec_lon_api_velff)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lon_api_acc_cmd - t.ctrldec_lon_api_acc_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lon_api_thrcmd - t.ctrldec_lon_api_thrcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrldec_lon_api_brkcmd - t.ctrldec_lon_api_brkcmd)) < DBL_EPSILON) && (ctrldec_adascalc_geargmd == t.ctrldec_adascalc_geargmd) && (ctrldec_ctrl_err == t.ctrldec_ctrl_err) && (ctrldec_actor_error == t.ctrldec_actor_error) && (ctrldec_sensor_error == t.ctrldec_sensor_error) && (ctrldec_algorithm_error == t.ctrldec_algorithm_error);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MBDCTRLDECDEBUG_H
