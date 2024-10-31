/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MBDADASCALCDEBUG_H
#define HOZON_SOC_MCU_IMPL_TYPE_MBDADASCALCDEBUG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "impl_type_boolean.h"
#include "impl_type_uint32.h"

namespace hozon {
namespace soc_mcu {
struct MbdADASCalcDebug {
    ::UInt8 adascalc_is_lccactive;
    ::Float adascalc_trajparam_a0;
    ::Float adascalc_trajparam_a1;
    ::Float adascalc_trajparam_a2;
    ::Float adascalc_trajparam_a3;
    ::UInt8 adascalc_trp_replanlevel;
    ::UInt8 adascalc_hostlindchgtoleft_bl;
    ::UInt8 adascalc_hostlindchgtorght_bl;
    ::UInt8 adascalc_accsystemstate;
    ::UInt8 adascalc_accstopreq;
    ::Float adascalc_deltavelocity;
    ::Float adascalc_deltadist;
    ::Float adascalc_drvrseltrgtspd_sg;
    ::UInt8 adascalc_acc_smrsts;
    ::UInt8 adascalc_enable;
    ::Boolean adascalc_replanningflag;
    ::UInt32 adascalc_gearcmd;
    ::UInt8 adascalc_estop;
    ::Float cal_adascalc_headdingoffset_rad;
    ::Float adascalc_lat_poserrcmd;
    ::Float adascalc_lat_headingcmd;
    ::Float adascalc_lat_velcmd;
    ::Float adascalc_latpre_curvcmd;
    ::Float adascalc_lon_poserrcmd;
    ::Float adascalc_lon_velcmd;
    ::Float adascalc_a_acctrajcmd;
    ::UInt8 adascalc_is_longtraj_replan;
    ::Float adascalc_m_strajerror;
    ::Float adascalc_trajparamlong_a0;
    ::Float adascalc_trajparamlong_a1;
    ::Float adascalc_trajparamlong_a2;
    ::Float adascalc_trajparamlong_a3;
    ::Float adascalc_trajparamlong_a4;
    ::Float adascalc_trajparamlong_a5;
    ::Float adascalc_v_spdtrajcmd;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(adascalc_is_lccactive);
        fun(adascalc_trajparam_a0);
        fun(adascalc_trajparam_a1);
        fun(adascalc_trajparam_a2);
        fun(adascalc_trajparam_a3);
        fun(adascalc_trp_replanlevel);
        fun(adascalc_hostlindchgtoleft_bl);
        fun(adascalc_hostlindchgtorght_bl);
        fun(adascalc_accsystemstate);
        fun(adascalc_accstopreq);
        fun(adascalc_deltavelocity);
        fun(adascalc_deltadist);
        fun(adascalc_drvrseltrgtspd_sg);
        fun(adascalc_acc_smrsts);
        fun(adascalc_enable);
        fun(adascalc_replanningflag);
        fun(adascalc_gearcmd);
        fun(adascalc_estop);
        fun(cal_adascalc_headdingoffset_rad);
        fun(adascalc_lat_poserrcmd);
        fun(adascalc_lat_headingcmd);
        fun(adascalc_lat_velcmd);
        fun(adascalc_latpre_curvcmd);
        fun(adascalc_lon_poserrcmd);
        fun(adascalc_lon_velcmd);
        fun(adascalc_a_acctrajcmd);
        fun(adascalc_is_longtraj_replan);
        fun(adascalc_m_strajerror);
        fun(adascalc_trajparamlong_a0);
        fun(adascalc_trajparamlong_a1);
        fun(adascalc_trajparamlong_a2);
        fun(adascalc_trajparamlong_a3);
        fun(adascalc_trajparamlong_a4);
        fun(adascalc_trajparamlong_a5);
        fun(adascalc_v_spdtrajcmd);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(adascalc_is_lccactive);
        fun(adascalc_trajparam_a0);
        fun(adascalc_trajparam_a1);
        fun(adascalc_trajparam_a2);
        fun(adascalc_trajparam_a3);
        fun(adascalc_trp_replanlevel);
        fun(adascalc_hostlindchgtoleft_bl);
        fun(adascalc_hostlindchgtorght_bl);
        fun(adascalc_accsystemstate);
        fun(adascalc_accstopreq);
        fun(adascalc_deltavelocity);
        fun(adascalc_deltadist);
        fun(adascalc_drvrseltrgtspd_sg);
        fun(adascalc_acc_smrsts);
        fun(adascalc_enable);
        fun(adascalc_replanningflag);
        fun(adascalc_gearcmd);
        fun(adascalc_estop);
        fun(cal_adascalc_headdingoffset_rad);
        fun(adascalc_lat_poserrcmd);
        fun(adascalc_lat_headingcmd);
        fun(adascalc_lat_velcmd);
        fun(adascalc_latpre_curvcmd);
        fun(adascalc_lon_poserrcmd);
        fun(adascalc_lon_velcmd);
        fun(adascalc_a_acctrajcmd);
        fun(adascalc_is_longtraj_replan);
        fun(adascalc_m_strajerror);
        fun(adascalc_trajparamlong_a0);
        fun(adascalc_trajparamlong_a1);
        fun(adascalc_trajparamlong_a2);
        fun(adascalc_trajparamlong_a3);
        fun(adascalc_trajparamlong_a4);
        fun(adascalc_trajparamlong_a5);
        fun(adascalc_v_spdtrajcmd);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("adascalc_is_lccactive", adascalc_is_lccactive);
        fun("adascalc_trajparam_a0", adascalc_trajparam_a0);
        fun("adascalc_trajparam_a1", adascalc_trajparam_a1);
        fun("adascalc_trajparam_a2", adascalc_trajparam_a2);
        fun("adascalc_trajparam_a3", adascalc_trajparam_a3);
        fun("adascalc_trp_replanlevel", adascalc_trp_replanlevel);
        fun("adascalc_hostlindchgtoleft_bl", adascalc_hostlindchgtoleft_bl);
        fun("adascalc_hostlindchgtorght_bl", adascalc_hostlindchgtorght_bl);
        fun("adascalc_accsystemstate", adascalc_accsystemstate);
        fun("adascalc_accstopreq", adascalc_accstopreq);
        fun("adascalc_deltavelocity", adascalc_deltavelocity);
        fun("adascalc_deltadist", adascalc_deltadist);
        fun("adascalc_drvrseltrgtspd_sg", adascalc_drvrseltrgtspd_sg);
        fun("adascalc_acc_smrsts", adascalc_acc_smrsts);
        fun("adascalc_enable", adascalc_enable);
        fun("adascalc_replanningflag", adascalc_replanningflag);
        fun("adascalc_gearcmd", adascalc_gearcmd);
        fun("adascalc_estop", adascalc_estop);
        fun("cal_adascalc_headdingoffset_rad", cal_adascalc_headdingoffset_rad);
        fun("adascalc_lat_poserrcmd", adascalc_lat_poserrcmd);
        fun("adascalc_lat_headingcmd", adascalc_lat_headingcmd);
        fun("adascalc_lat_velcmd", adascalc_lat_velcmd);
        fun("adascalc_latpre_curvcmd", adascalc_latpre_curvcmd);
        fun("adascalc_lon_poserrcmd", adascalc_lon_poserrcmd);
        fun("adascalc_lon_velcmd", adascalc_lon_velcmd);
        fun("adascalc_a_acctrajcmd", adascalc_a_acctrajcmd);
        fun("adascalc_is_longtraj_replan", adascalc_is_longtraj_replan);
        fun("adascalc_m_strajerror", adascalc_m_strajerror);
        fun("adascalc_trajparamlong_a0", adascalc_trajparamlong_a0);
        fun("adascalc_trajparamlong_a1", adascalc_trajparamlong_a1);
        fun("adascalc_trajparamlong_a2", adascalc_trajparamlong_a2);
        fun("adascalc_trajparamlong_a3", adascalc_trajparamlong_a3);
        fun("adascalc_trajparamlong_a4", adascalc_trajparamlong_a4);
        fun("adascalc_trajparamlong_a5", adascalc_trajparamlong_a5);
        fun("adascalc_v_spdtrajcmd", adascalc_v_spdtrajcmd);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("adascalc_is_lccactive", adascalc_is_lccactive);
        fun("adascalc_trajparam_a0", adascalc_trajparam_a0);
        fun("adascalc_trajparam_a1", adascalc_trajparam_a1);
        fun("adascalc_trajparam_a2", adascalc_trajparam_a2);
        fun("adascalc_trajparam_a3", adascalc_trajparam_a3);
        fun("adascalc_trp_replanlevel", adascalc_trp_replanlevel);
        fun("adascalc_hostlindchgtoleft_bl", adascalc_hostlindchgtoleft_bl);
        fun("adascalc_hostlindchgtorght_bl", adascalc_hostlindchgtorght_bl);
        fun("adascalc_accsystemstate", adascalc_accsystemstate);
        fun("adascalc_accstopreq", adascalc_accstopreq);
        fun("adascalc_deltavelocity", adascalc_deltavelocity);
        fun("adascalc_deltadist", adascalc_deltadist);
        fun("adascalc_drvrseltrgtspd_sg", adascalc_drvrseltrgtspd_sg);
        fun("adascalc_acc_smrsts", adascalc_acc_smrsts);
        fun("adascalc_enable", adascalc_enable);
        fun("adascalc_replanningflag", adascalc_replanningflag);
        fun("adascalc_gearcmd", adascalc_gearcmd);
        fun("adascalc_estop", adascalc_estop);
        fun("cal_adascalc_headdingoffset_rad", cal_adascalc_headdingoffset_rad);
        fun("adascalc_lat_poserrcmd", adascalc_lat_poserrcmd);
        fun("adascalc_lat_headingcmd", adascalc_lat_headingcmd);
        fun("adascalc_lat_velcmd", adascalc_lat_velcmd);
        fun("adascalc_latpre_curvcmd", adascalc_latpre_curvcmd);
        fun("adascalc_lon_poserrcmd", adascalc_lon_poserrcmd);
        fun("adascalc_lon_velcmd", adascalc_lon_velcmd);
        fun("adascalc_a_acctrajcmd", adascalc_a_acctrajcmd);
        fun("adascalc_is_longtraj_replan", adascalc_is_longtraj_replan);
        fun("adascalc_m_strajerror", adascalc_m_strajerror);
        fun("adascalc_trajparamlong_a0", adascalc_trajparamlong_a0);
        fun("adascalc_trajparamlong_a1", adascalc_trajparamlong_a1);
        fun("adascalc_trajparamlong_a2", adascalc_trajparamlong_a2);
        fun("adascalc_trajparamlong_a3", adascalc_trajparamlong_a3);
        fun("adascalc_trajparamlong_a4", adascalc_trajparamlong_a4);
        fun("adascalc_trajparamlong_a5", adascalc_trajparamlong_a5);
        fun("adascalc_v_spdtrajcmd", adascalc_v_spdtrajcmd);
    }

    bool operator==(const ::hozon::soc_mcu::MbdADASCalcDebug& t) const
    {
        return (adascalc_is_lccactive == t.adascalc_is_lccactive) && (fabs(static_cast<double>(adascalc_trajparam_a0 - t.adascalc_trajparam_a0)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_trajparam_a1 - t.adascalc_trajparam_a1)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_trajparam_a2 - t.adascalc_trajparam_a2)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_trajparam_a3 - t.adascalc_trajparam_a3)) < DBL_EPSILON) && (adascalc_trp_replanlevel == t.adascalc_trp_replanlevel) && (adascalc_hostlindchgtoleft_bl == t.adascalc_hostlindchgtoleft_bl) && (adascalc_hostlindchgtorght_bl == t.adascalc_hostlindchgtorght_bl) && (adascalc_accsystemstate == t.adascalc_accsystemstate) && (adascalc_accstopreq == t.adascalc_accstopreq) && (fabs(static_cast<double>(adascalc_deltavelocity - t.adascalc_deltavelocity)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_deltadist - t.adascalc_deltadist)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_drvrseltrgtspd_sg - t.adascalc_drvrseltrgtspd_sg)) < DBL_EPSILON) && (adascalc_acc_smrsts == t.adascalc_acc_smrsts) && (adascalc_enable == t.adascalc_enable) && (adascalc_replanningflag == t.adascalc_replanningflag) && (adascalc_gearcmd == t.adascalc_gearcmd) && (adascalc_estop == t.adascalc_estop) && (fabs(static_cast<double>(cal_adascalc_headdingoffset_rad - t.cal_adascalc_headdingoffset_rad)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_lat_poserrcmd - t.adascalc_lat_poserrcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_lat_headingcmd - t.adascalc_lat_headingcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_lat_velcmd - t.adascalc_lat_velcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_latpre_curvcmd - t.adascalc_latpre_curvcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_lon_poserrcmd - t.adascalc_lon_poserrcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_lon_velcmd - t.adascalc_lon_velcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_a_acctrajcmd - t.adascalc_a_acctrajcmd)) < DBL_EPSILON) && (adascalc_is_longtraj_replan == t.adascalc_is_longtraj_replan) && (fabs(static_cast<double>(adascalc_m_strajerror - t.adascalc_m_strajerror)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_trajparamlong_a0 - t.adascalc_trajparamlong_a0)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_trajparamlong_a1 - t.adascalc_trajparamlong_a1)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_trajparamlong_a2 - t.adascalc_trajparamlong_a2)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_trajparamlong_a3 - t.adascalc_trajparamlong_a3)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_trajparamlong_a4 - t.adascalc_trajparamlong_a4)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_trajparamlong_a5 - t.adascalc_trajparamlong_a5)) < DBL_EPSILON) && (fabs(static_cast<double>(adascalc_v_spdtrajcmd - t.adascalc_v_spdtrajcmd)) < DBL_EPSILON);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MBDADASCALCDEBUG_H
