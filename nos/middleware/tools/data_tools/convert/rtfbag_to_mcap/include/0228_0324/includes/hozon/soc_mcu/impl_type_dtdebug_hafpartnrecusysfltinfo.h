/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFPARTNRECUSYSFLTINFO_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFPARTNRECUSYSFLTINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_HafpartnrEcuSysFltInfo {
    ::UInt8 ESC_Sys_IDB_Flt;
    ::UInt8 EPB_Sys_Flt;
    ::UInt8 IDB_Sys_Flt;
    ::UInt8 ABS_Sys_IDB_Flt;
    ::UInt8 EBD_Sys_IDB_Flt;
    ::UInt8 TCS_Sys_IDB_Flt;
    ::UInt8 Str_A_Sys_Flt;
    ::UInt8 CDCS_Sys_Flt;
    ::UInt8 ESC_Sys_RCU7_Flt;
    ::UInt8 TCS_Sys_RCU_Flt;
    ::UInt8 HSA_Sys_RCU_Flt;
    ::UInt8 ABS_Sys_RCU_Flt;
    ::UInt8 ESC_Sys_RCU5_Flt;
    ::UInt8 BUCKLE_Sys_Flt;
    ::UInt8 EPS_ADAS_Sys_Flt;
    ::UInt8 EPS_PA_Sys_Flt;
    ::UInt8 WHEEL_rr_Sys_Flt;
    ::UInt8 WHEEL_lr_Sys_Flt;
    ::UInt8 WHEEL_rf_Sys_Flt;
    ::UInt8 WHEEL_lf_Sys_Flt;
    ::UInt8 IDB_PreFill_Sys_Flt;
    ::UInt8 IDB_PA_Sts_Sys_Flt;
    ::UInt8 IDB_Jerk_Sys_Flt;
    ::UInt8 ESC_Sys_Off_Flt;
    ::UInt8 HBA_Sys_Flt;
    ::UInt8 ROP_Sys_Flt;
    ::UInt8 HSA_Sys_Flt;
    ::UInt8 HDC_Sys_Flt;
    ::UInt8 GearPos_Sys_Flt;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ESC_Sys_IDB_Flt);
        fun(EPB_Sys_Flt);
        fun(IDB_Sys_Flt);
        fun(ABS_Sys_IDB_Flt);
        fun(EBD_Sys_IDB_Flt);
        fun(TCS_Sys_IDB_Flt);
        fun(Str_A_Sys_Flt);
        fun(CDCS_Sys_Flt);
        fun(ESC_Sys_RCU7_Flt);
        fun(TCS_Sys_RCU_Flt);
        fun(HSA_Sys_RCU_Flt);
        fun(ABS_Sys_RCU_Flt);
        fun(ESC_Sys_RCU5_Flt);
        fun(BUCKLE_Sys_Flt);
        fun(EPS_ADAS_Sys_Flt);
        fun(EPS_PA_Sys_Flt);
        fun(WHEEL_rr_Sys_Flt);
        fun(WHEEL_lr_Sys_Flt);
        fun(WHEEL_rf_Sys_Flt);
        fun(WHEEL_lf_Sys_Flt);
        fun(IDB_PreFill_Sys_Flt);
        fun(IDB_PA_Sts_Sys_Flt);
        fun(IDB_Jerk_Sys_Flt);
        fun(ESC_Sys_Off_Flt);
        fun(HBA_Sys_Flt);
        fun(ROP_Sys_Flt);
        fun(HSA_Sys_Flt);
        fun(HDC_Sys_Flt);
        fun(GearPos_Sys_Flt);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ESC_Sys_IDB_Flt);
        fun(EPB_Sys_Flt);
        fun(IDB_Sys_Flt);
        fun(ABS_Sys_IDB_Flt);
        fun(EBD_Sys_IDB_Flt);
        fun(TCS_Sys_IDB_Flt);
        fun(Str_A_Sys_Flt);
        fun(CDCS_Sys_Flt);
        fun(ESC_Sys_RCU7_Flt);
        fun(TCS_Sys_RCU_Flt);
        fun(HSA_Sys_RCU_Flt);
        fun(ABS_Sys_RCU_Flt);
        fun(ESC_Sys_RCU5_Flt);
        fun(BUCKLE_Sys_Flt);
        fun(EPS_ADAS_Sys_Flt);
        fun(EPS_PA_Sys_Flt);
        fun(WHEEL_rr_Sys_Flt);
        fun(WHEEL_lr_Sys_Flt);
        fun(WHEEL_rf_Sys_Flt);
        fun(WHEEL_lf_Sys_Flt);
        fun(IDB_PreFill_Sys_Flt);
        fun(IDB_PA_Sts_Sys_Flt);
        fun(IDB_Jerk_Sys_Flt);
        fun(ESC_Sys_Off_Flt);
        fun(HBA_Sys_Flt);
        fun(ROP_Sys_Flt);
        fun(HSA_Sys_Flt);
        fun(HDC_Sys_Flt);
        fun(GearPos_Sys_Flt);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ESC_Sys_IDB_Flt", ESC_Sys_IDB_Flt);
        fun("EPB_Sys_Flt", EPB_Sys_Flt);
        fun("IDB_Sys_Flt", IDB_Sys_Flt);
        fun("ABS_Sys_IDB_Flt", ABS_Sys_IDB_Flt);
        fun("EBD_Sys_IDB_Flt", EBD_Sys_IDB_Flt);
        fun("TCS_Sys_IDB_Flt", TCS_Sys_IDB_Flt);
        fun("Str_A_Sys_Flt", Str_A_Sys_Flt);
        fun("CDCS_Sys_Flt", CDCS_Sys_Flt);
        fun("ESC_Sys_RCU7_Flt", ESC_Sys_RCU7_Flt);
        fun("TCS_Sys_RCU_Flt", TCS_Sys_RCU_Flt);
        fun("HSA_Sys_RCU_Flt", HSA_Sys_RCU_Flt);
        fun("ABS_Sys_RCU_Flt", ABS_Sys_RCU_Flt);
        fun("ESC_Sys_RCU5_Flt", ESC_Sys_RCU5_Flt);
        fun("BUCKLE_Sys_Flt", BUCKLE_Sys_Flt);
        fun("EPS_ADAS_Sys_Flt", EPS_ADAS_Sys_Flt);
        fun("EPS_PA_Sys_Flt", EPS_PA_Sys_Flt);
        fun("WHEEL_rr_Sys_Flt", WHEEL_rr_Sys_Flt);
        fun("WHEEL_lr_Sys_Flt", WHEEL_lr_Sys_Flt);
        fun("WHEEL_rf_Sys_Flt", WHEEL_rf_Sys_Flt);
        fun("WHEEL_lf_Sys_Flt", WHEEL_lf_Sys_Flt);
        fun("IDB_PreFill_Sys_Flt", IDB_PreFill_Sys_Flt);
        fun("IDB_PA_Sts_Sys_Flt", IDB_PA_Sts_Sys_Flt);
        fun("IDB_Jerk_Sys_Flt", IDB_Jerk_Sys_Flt);
        fun("ESC_Sys_Off_Flt", ESC_Sys_Off_Flt);
        fun("HBA_Sys_Flt", HBA_Sys_Flt);
        fun("ROP_Sys_Flt", ROP_Sys_Flt);
        fun("HSA_Sys_Flt", HSA_Sys_Flt);
        fun("HDC_Sys_Flt", HDC_Sys_Flt);
        fun("GearPos_Sys_Flt", GearPos_Sys_Flt);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ESC_Sys_IDB_Flt", ESC_Sys_IDB_Flt);
        fun("EPB_Sys_Flt", EPB_Sys_Flt);
        fun("IDB_Sys_Flt", IDB_Sys_Flt);
        fun("ABS_Sys_IDB_Flt", ABS_Sys_IDB_Flt);
        fun("EBD_Sys_IDB_Flt", EBD_Sys_IDB_Flt);
        fun("TCS_Sys_IDB_Flt", TCS_Sys_IDB_Flt);
        fun("Str_A_Sys_Flt", Str_A_Sys_Flt);
        fun("CDCS_Sys_Flt", CDCS_Sys_Flt);
        fun("ESC_Sys_RCU7_Flt", ESC_Sys_RCU7_Flt);
        fun("TCS_Sys_RCU_Flt", TCS_Sys_RCU_Flt);
        fun("HSA_Sys_RCU_Flt", HSA_Sys_RCU_Flt);
        fun("ABS_Sys_RCU_Flt", ABS_Sys_RCU_Flt);
        fun("ESC_Sys_RCU5_Flt", ESC_Sys_RCU5_Flt);
        fun("BUCKLE_Sys_Flt", BUCKLE_Sys_Flt);
        fun("EPS_ADAS_Sys_Flt", EPS_ADAS_Sys_Flt);
        fun("EPS_PA_Sys_Flt", EPS_PA_Sys_Flt);
        fun("WHEEL_rr_Sys_Flt", WHEEL_rr_Sys_Flt);
        fun("WHEEL_lr_Sys_Flt", WHEEL_lr_Sys_Flt);
        fun("WHEEL_rf_Sys_Flt", WHEEL_rf_Sys_Flt);
        fun("WHEEL_lf_Sys_Flt", WHEEL_lf_Sys_Flt);
        fun("IDB_PreFill_Sys_Flt", IDB_PreFill_Sys_Flt);
        fun("IDB_PA_Sts_Sys_Flt", IDB_PA_Sts_Sys_Flt);
        fun("IDB_Jerk_Sys_Flt", IDB_Jerk_Sys_Flt);
        fun("ESC_Sys_Off_Flt", ESC_Sys_Off_Flt);
        fun("HBA_Sys_Flt", HBA_Sys_Flt);
        fun("ROP_Sys_Flt", ROP_Sys_Flt);
        fun("HSA_Sys_Flt", HSA_Sys_Flt);
        fun("HDC_Sys_Flt", HDC_Sys_Flt);
        fun("GearPos_Sys_Flt", GearPos_Sys_Flt);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_HafpartnrEcuSysFltInfo& t) const
    {
        return (ESC_Sys_IDB_Flt == t.ESC_Sys_IDB_Flt) && (EPB_Sys_Flt == t.EPB_Sys_Flt) && (IDB_Sys_Flt == t.IDB_Sys_Flt) && (ABS_Sys_IDB_Flt == t.ABS_Sys_IDB_Flt) && (EBD_Sys_IDB_Flt == t.EBD_Sys_IDB_Flt) && (TCS_Sys_IDB_Flt == t.TCS_Sys_IDB_Flt) && (Str_A_Sys_Flt == t.Str_A_Sys_Flt) && (CDCS_Sys_Flt == t.CDCS_Sys_Flt) && (ESC_Sys_RCU7_Flt == t.ESC_Sys_RCU7_Flt) && (TCS_Sys_RCU_Flt == t.TCS_Sys_RCU_Flt) && (HSA_Sys_RCU_Flt == t.HSA_Sys_RCU_Flt) && (ABS_Sys_RCU_Flt == t.ABS_Sys_RCU_Flt) && (ESC_Sys_RCU5_Flt == t.ESC_Sys_RCU5_Flt) && (BUCKLE_Sys_Flt == t.BUCKLE_Sys_Flt) && (EPS_ADAS_Sys_Flt == t.EPS_ADAS_Sys_Flt) && (EPS_PA_Sys_Flt == t.EPS_PA_Sys_Flt) && (WHEEL_rr_Sys_Flt == t.WHEEL_rr_Sys_Flt) && (WHEEL_lr_Sys_Flt == t.WHEEL_lr_Sys_Flt) && (WHEEL_rf_Sys_Flt == t.WHEEL_rf_Sys_Flt) && (WHEEL_lf_Sys_Flt == t.WHEEL_lf_Sys_Flt) && (IDB_PreFill_Sys_Flt == t.IDB_PreFill_Sys_Flt) && (IDB_PA_Sts_Sys_Flt == t.IDB_PA_Sts_Sys_Flt) && (IDB_Jerk_Sys_Flt == t.IDB_Jerk_Sys_Flt) && (ESC_Sys_Off_Flt == t.ESC_Sys_Off_Flt) && (HBA_Sys_Flt == t.HBA_Sys_Flt) && (ROP_Sys_Flt == t.ROP_Sys_Flt) && (HSA_Sys_Flt == t.HSA_Sys_Flt) && (HDC_Sys_Flt == t.HDC_Sys_Flt) && (GearPos_Sys_Flt == t.GearPos_Sys_Flt);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFPARTNRECUSYSFLTINFO_H
