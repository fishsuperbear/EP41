/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFCOMDATAFLTINFO_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFCOMDATAFLTINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HafcomDataFltInfo {
    ::UInt8 MCU_Fr01_0AA_Data_Error;
    ::UInt8 FMCU_Fr01_0AB_Data_Error;
    ::UInt8 EPS_Fr01_0B1_Data_Error;
    ::UInt8 EPS_Fr02_0B2_Data_Error;
    ::UInt8 IDB_Fr02_0C0_Data_Error;
    ::UInt8 ACU_Fr02_0C4_Data_Error;
    ::UInt8 IDB_Fr03_0C5_Data_Error;
    ::UInt8 IDB_Fr04_0C7_Data_Error;
    ::UInt8 IDB_Fr10_0C9_Data_Error;
    ::UInt8 VCU_Fr05_0E3_Data_Error;
    ::UInt8 IDB_Fr01_0E5_Data_Error;
    ::UInt8 RCU_Fr01_0E6_Data_Error;
    ::UInt8 BTM_Fr01_0F3_Data_Error;
    ::UInt8 VCU_Fr06_0FB_Data_Error;
    ::UInt8 ACU_Fr01_102_Data_Error;
    ::UInt8 VCU_Fr0E_108_Data_Error;
    ::UInt8 BDCS_Fr01_110_Data1_Error;
    ::UInt8 BDCS_Fr03_112_Data_Error;
    ::UInt8 BDCS_Fr16_114_Data_Error;
    ::UInt8 IDB_Fr09_121_Data_Error;
    ::UInt8 IDB_Fr05_129_Data_Error;
    ::UInt8 IDB_Fr06_12B_Data_Error;
    ::UInt8 IDB_Fr07_12D_Data_Error;
    ::UInt8 ICU_Fr01_131_Data_Error;
    ::UInt8 RCU_Fr02_137_Data_Error;
    ::UInt8 RCU_Fr03_138_Data_Error;
    ::UInt8 RCU_Fr04_139_Data_Error;
    ::UInt8 RCU_Fr06_13A_Data_Error;
    ::UInt8 RCU_Fr07_13B_Data_Error;
    ::UInt8 RCU_Fr08_13C_Data_Error;
    ::UInt8 RCU_Fr05_13D_Data_Error;
    ::UInt8 RCU_Fr09_13E_Data_Error;
    ::UInt8 GW_Fr01_15B_Data_Error;
    ::UInt8 DDCU_Fr01_1A2_Data_Error;
    ::UInt8 PDCU_Fr01_1A3_Data_Error;
    ::UInt8 TBOX_Fr02_1B6_Data_Error;
    ::UInt8 CDCS_Fr10_1BA_Data_Error;
    ::UInt8 CDCS_Fr13_1BB_Data_Error;
    ::UInt8 BDCS_Fr04_1E2_Data_Error;
    ::UInt8 BDCS_Fr05_1E4_Data_Error;
    ::UInt8 BDCS_Fr06_1E6_Data_Error;
    ::UInt8 BDCS_Fr10_29E_Data_Error;
    ::UInt8 EDU_Fr03_2C2_Data_Error;
    ::UInt8 EDU_Fr04_2C3_Data_Error;
    ::UInt8 BTM_Fr02_2F4_Data_Error;
    ::UInt8 HNS1_Fr02_2FD_Data_Error;
    ::UInt8 ICU_Fr02_325_Data_Error;
    ::UInt8 TBOX_Fr01_3E0_Data_Error;
    ::UInt8 EDU_Fr06_3E5_Data_Error;
    ::UInt8 CDCS_Fr15_203_Data_Error;
    ::UInt8 AICS_Fr2_243_Data_Error;
    ::UInt8 BMS_Fr03_258_Data_Error;
    ::UInt8 BDCS_Fr13_LIN2_285_Data_Error;
    ::UInt8 VCU_Fr04_2D4_Data_Error;
    ::UInt8 VCU_Fr03_2E3_Data_Error;
    ::UInt8 VCU_Fr07_0FC_Data_Error;
    ::UInt8 EDU_Fr05_1D5_Data_Error;
    ::UInt8 BDCS_Fr01_110_Data2_Error;
    ::UInt8 HNS2_Fr02_2FE_Data_Error;
    ::UInt8 ACU_Node_Lost;
    ::UInt8 BDCS_Node_Lost;
    ::UInt8 BTM_Node_Lost;
    ::UInt8 CDCS_Node_Lost;
    ::UInt8 DDCU_Node_Lost;
    ::UInt8 EDU_Node_Lost;
    ::UInt8 EPS_Node_Lost;
    ::UInt8 FMCU_Node_Lost;
    ::UInt8 GW_Node_Lost;
    ::UInt8 HNS_Node_Lost;
    ::UInt8 ICU_Node_Lost;
    ::UInt8 IDB_Node_Lost;
    ::UInt8 MCU_Node_Lost;
    ::UInt8 PDCU_Node_Lost;
    ::UInt8 RCU_Node_Lost;
    ::UInt8 TBOX_Node_Lost;
    ::UInt8 PDCS_Node_Lost;
    ::UInt8 BMS_Node_Lost;
    ::UInt8 FD3_Bus_Error;
    ::UInt8 FD6_Bus_Error;
    ::UInt8 FD8_Bus_Error;
    ::UInt8 Innr_Bus_Error;
    ::UInt8 Innr_MsgLost_Error;
    ::UInt8 Innr_MsgData_Error;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(MCU_Fr01_0AA_Data_Error);
        fun(FMCU_Fr01_0AB_Data_Error);
        fun(EPS_Fr01_0B1_Data_Error);
        fun(EPS_Fr02_0B2_Data_Error);
        fun(IDB_Fr02_0C0_Data_Error);
        fun(ACU_Fr02_0C4_Data_Error);
        fun(IDB_Fr03_0C5_Data_Error);
        fun(IDB_Fr04_0C7_Data_Error);
        fun(IDB_Fr10_0C9_Data_Error);
        fun(VCU_Fr05_0E3_Data_Error);
        fun(IDB_Fr01_0E5_Data_Error);
        fun(RCU_Fr01_0E6_Data_Error);
        fun(BTM_Fr01_0F3_Data_Error);
        fun(VCU_Fr06_0FB_Data_Error);
        fun(ACU_Fr01_102_Data_Error);
        fun(VCU_Fr0E_108_Data_Error);
        fun(BDCS_Fr01_110_Data1_Error);
        fun(BDCS_Fr03_112_Data_Error);
        fun(BDCS_Fr16_114_Data_Error);
        fun(IDB_Fr09_121_Data_Error);
        fun(IDB_Fr05_129_Data_Error);
        fun(IDB_Fr06_12B_Data_Error);
        fun(IDB_Fr07_12D_Data_Error);
        fun(ICU_Fr01_131_Data_Error);
        fun(RCU_Fr02_137_Data_Error);
        fun(RCU_Fr03_138_Data_Error);
        fun(RCU_Fr04_139_Data_Error);
        fun(RCU_Fr06_13A_Data_Error);
        fun(RCU_Fr07_13B_Data_Error);
        fun(RCU_Fr08_13C_Data_Error);
        fun(RCU_Fr05_13D_Data_Error);
        fun(RCU_Fr09_13E_Data_Error);
        fun(GW_Fr01_15B_Data_Error);
        fun(DDCU_Fr01_1A2_Data_Error);
        fun(PDCU_Fr01_1A3_Data_Error);
        fun(TBOX_Fr02_1B6_Data_Error);
        fun(CDCS_Fr10_1BA_Data_Error);
        fun(CDCS_Fr13_1BB_Data_Error);
        fun(BDCS_Fr04_1E2_Data_Error);
        fun(BDCS_Fr05_1E4_Data_Error);
        fun(BDCS_Fr06_1E6_Data_Error);
        fun(BDCS_Fr10_29E_Data_Error);
        fun(EDU_Fr03_2C2_Data_Error);
        fun(EDU_Fr04_2C3_Data_Error);
        fun(BTM_Fr02_2F4_Data_Error);
        fun(HNS1_Fr02_2FD_Data_Error);
        fun(ICU_Fr02_325_Data_Error);
        fun(TBOX_Fr01_3E0_Data_Error);
        fun(EDU_Fr06_3E5_Data_Error);
        fun(CDCS_Fr15_203_Data_Error);
        fun(AICS_Fr2_243_Data_Error);
        fun(BMS_Fr03_258_Data_Error);
        fun(BDCS_Fr13_LIN2_285_Data_Error);
        fun(VCU_Fr04_2D4_Data_Error);
        fun(VCU_Fr03_2E3_Data_Error);
        fun(VCU_Fr07_0FC_Data_Error);
        fun(EDU_Fr05_1D5_Data_Error);
        fun(BDCS_Fr01_110_Data2_Error);
        fun(HNS2_Fr02_2FE_Data_Error);
        fun(ACU_Node_Lost);
        fun(BDCS_Node_Lost);
        fun(BTM_Node_Lost);
        fun(CDCS_Node_Lost);
        fun(DDCU_Node_Lost);
        fun(EDU_Node_Lost);
        fun(EPS_Node_Lost);
        fun(FMCU_Node_Lost);
        fun(GW_Node_Lost);
        fun(HNS_Node_Lost);
        fun(ICU_Node_Lost);
        fun(IDB_Node_Lost);
        fun(MCU_Node_Lost);
        fun(PDCU_Node_Lost);
        fun(RCU_Node_Lost);
        fun(TBOX_Node_Lost);
        fun(PDCS_Node_Lost);
        fun(BMS_Node_Lost);
        fun(FD3_Bus_Error);
        fun(FD6_Bus_Error);
        fun(FD8_Bus_Error);
        fun(Innr_Bus_Error);
        fun(Innr_MsgLost_Error);
        fun(Innr_MsgData_Error);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(MCU_Fr01_0AA_Data_Error);
        fun(FMCU_Fr01_0AB_Data_Error);
        fun(EPS_Fr01_0B1_Data_Error);
        fun(EPS_Fr02_0B2_Data_Error);
        fun(IDB_Fr02_0C0_Data_Error);
        fun(ACU_Fr02_0C4_Data_Error);
        fun(IDB_Fr03_0C5_Data_Error);
        fun(IDB_Fr04_0C7_Data_Error);
        fun(IDB_Fr10_0C9_Data_Error);
        fun(VCU_Fr05_0E3_Data_Error);
        fun(IDB_Fr01_0E5_Data_Error);
        fun(RCU_Fr01_0E6_Data_Error);
        fun(BTM_Fr01_0F3_Data_Error);
        fun(VCU_Fr06_0FB_Data_Error);
        fun(ACU_Fr01_102_Data_Error);
        fun(VCU_Fr0E_108_Data_Error);
        fun(BDCS_Fr01_110_Data1_Error);
        fun(BDCS_Fr03_112_Data_Error);
        fun(BDCS_Fr16_114_Data_Error);
        fun(IDB_Fr09_121_Data_Error);
        fun(IDB_Fr05_129_Data_Error);
        fun(IDB_Fr06_12B_Data_Error);
        fun(IDB_Fr07_12D_Data_Error);
        fun(ICU_Fr01_131_Data_Error);
        fun(RCU_Fr02_137_Data_Error);
        fun(RCU_Fr03_138_Data_Error);
        fun(RCU_Fr04_139_Data_Error);
        fun(RCU_Fr06_13A_Data_Error);
        fun(RCU_Fr07_13B_Data_Error);
        fun(RCU_Fr08_13C_Data_Error);
        fun(RCU_Fr05_13D_Data_Error);
        fun(RCU_Fr09_13E_Data_Error);
        fun(GW_Fr01_15B_Data_Error);
        fun(DDCU_Fr01_1A2_Data_Error);
        fun(PDCU_Fr01_1A3_Data_Error);
        fun(TBOX_Fr02_1B6_Data_Error);
        fun(CDCS_Fr10_1BA_Data_Error);
        fun(CDCS_Fr13_1BB_Data_Error);
        fun(BDCS_Fr04_1E2_Data_Error);
        fun(BDCS_Fr05_1E4_Data_Error);
        fun(BDCS_Fr06_1E6_Data_Error);
        fun(BDCS_Fr10_29E_Data_Error);
        fun(EDU_Fr03_2C2_Data_Error);
        fun(EDU_Fr04_2C3_Data_Error);
        fun(BTM_Fr02_2F4_Data_Error);
        fun(HNS1_Fr02_2FD_Data_Error);
        fun(ICU_Fr02_325_Data_Error);
        fun(TBOX_Fr01_3E0_Data_Error);
        fun(EDU_Fr06_3E5_Data_Error);
        fun(CDCS_Fr15_203_Data_Error);
        fun(AICS_Fr2_243_Data_Error);
        fun(BMS_Fr03_258_Data_Error);
        fun(BDCS_Fr13_LIN2_285_Data_Error);
        fun(VCU_Fr04_2D4_Data_Error);
        fun(VCU_Fr03_2E3_Data_Error);
        fun(VCU_Fr07_0FC_Data_Error);
        fun(EDU_Fr05_1D5_Data_Error);
        fun(BDCS_Fr01_110_Data2_Error);
        fun(HNS2_Fr02_2FE_Data_Error);
        fun(ACU_Node_Lost);
        fun(BDCS_Node_Lost);
        fun(BTM_Node_Lost);
        fun(CDCS_Node_Lost);
        fun(DDCU_Node_Lost);
        fun(EDU_Node_Lost);
        fun(EPS_Node_Lost);
        fun(FMCU_Node_Lost);
        fun(GW_Node_Lost);
        fun(HNS_Node_Lost);
        fun(ICU_Node_Lost);
        fun(IDB_Node_Lost);
        fun(MCU_Node_Lost);
        fun(PDCU_Node_Lost);
        fun(RCU_Node_Lost);
        fun(TBOX_Node_Lost);
        fun(PDCS_Node_Lost);
        fun(BMS_Node_Lost);
        fun(FD3_Bus_Error);
        fun(FD6_Bus_Error);
        fun(FD8_Bus_Error);
        fun(Innr_Bus_Error);
        fun(Innr_MsgLost_Error);
        fun(Innr_MsgData_Error);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("MCU_Fr01_0AA_Data_Error", MCU_Fr01_0AA_Data_Error);
        fun("FMCU_Fr01_0AB_Data_Error", FMCU_Fr01_0AB_Data_Error);
        fun("EPS_Fr01_0B1_Data_Error", EPS_Fr01_0B1_Data_Error);
        fun("EPS_Fr02_0B2_Data_Error", EPS_Fr02_0B2_Data_Error);
        fun("IDB_Fr02_0C0_Data_Error", IDB_Fr02_0C0_Data_Error);
        fun("ACU_Fr02_0C4_Data_Error", ACU_Fr02_0C4_Data_Error);
        fun("IDB_Fr03_0C5_Data_Error", IDB_Fr03_0C5_Data_Error);
        fun("IDB_Fr04_0C7_Data_Error", IDB_Fr04_0C7_Data_Error);
        fun("IDB_Fr10_0C9_Data_Error", IDB_Fr10_0C9_Data_Error);
        fun("VCU_Fr05_0E3_Data_Error", VCU_Fr05_0E3_Data_Error);
        fun("IDB_Fr01_0E5_Data_Error", IDB_Fr01_0E5_Data_Error);
        fun("RCU_Fr01_0E6_Data_Error", RCU_Fr01_0E6_Data_Error);
        fun("BTM_Fr01_0F3_Data_Error", BTM_Fr01_0F3_Data_Error);
        fun("VCU_Fr06_0FB_Data_Error", VCU_Fr06_0FB_Data_Error);
        fun("ACU_Fr01_102_Data_Error", ACU_Fr01_102_Data_Error);
        fun("VCU_Fr0E_108_Data_Error", VCU_Fr0E_108_Data_Error);
        fun("BDCS_Fr01_110_Data1_Error", BDCS_Fr01_110_Data1_Error);
        fun("BDCS_Fr03_112_Data_Error", BDCS_Fr03_112_Data_Error);
        fun("BDCS_Fr16_114_Data_Error", BDCS_Fr16_114_Data_Error);
        fun("IDB_Fr09_121_Data_Error", IDB_Fr09_121_Data_Error);
        fun("IDB_Fr05_129_Data_Error", IDB_Fr05_129_Data_Error);
        fun("IDB_Fr06_12B_Data_Error", IDB_Fr06_12B_Data_Error);
        fun("IDB_Fr07_12D_Data_Error", IDB_Fr07_12D_Data_Error);
        fun("ICU_Fr01_131_Data_Error", ICU_Fr01_131_Data_Error);
        fun("RCU_Fr02_137_Data_Error", RCU_Fr02_137_Data_Error);
        fun("RCU_Fr03_138_Data_Error", RCU_Fr03_138_Data_Error);
        fun("RCU_Fr04_139_Data_Error", RCU_Fr04_139_Data_Error);
        fun("RCU_Fr06_13A_Data_Error", RCU_Fr06_13A_Data_Error);
        fun("RCU_Fr07_13B_Data_Error", RCU_Fr07_13B_Data_Error);
        fun("RCU_Fr08_13C_Data_Error", RCU_Fr08_13C_Data_Error);
        fun("RCU_Fr05_13D_Data_Error", RCU_Fr05_13D_Data_Error);
        fun("RCU_Fr09_13E_Data_Error", RCU_Fr09_13E_Data_Error);
        fun("GW_Fr01_15B_Data_Error", GW_Fr01_15B_Data_Error);
        fun("DDCU_Fr01_1A2_Data_Error", DDCU_Fr01_1A2_Data_Error);
        fun("PDCU_Fr01_1A3_Data_Error", PDCU_Fr01_1A3_Data_Error);
        fun("TBOX_Fr02_1B6_Data_Error", TBOX_Fr02_1B6_Data_Error);
        fun("CDCS_Fr10_1BA_Data_Error", CDCS_Fr10_1BA_Data_Error);
        fun("CDCS_Fr13_1BB_Data_Error", CDCS_Fr13_1BB_Data_Error);
        fun("BDCS_Fr04_1E2_Data_Error", BDCS_Fr04_1E2_Data_Error);
        fun("BDCS_Fr05_1E4_Data_Error", BDCS_Fr05_1E4_Data_Error);
        fun("BDCS_Fr06_1E6_Data_Error", BDCS_Fr06_1E6_Data_Error);
        fun("BDCS_Fr10_29E_Data_Error", BDCS_Fr10_29E_Data_Error);
        fun("EDU_Fr03_2C2_Data_Error", EDU_Fr03_2C2_Data_Error);
        fun("EDU_Fr04_2C3_Data_Error", EDU_Fr04_2C3_Data_Error);
        fun("BTM_Fr02_2F4_Data_Error", BTM_Fr02_2F4_Data_Error);
        fun("HNS1_Fr02_2FD_Data_Error", HNS1_Fr02_2FD_Data_Error);
        fun("ICU_Fr02_325_Data_Error", ICU_Fr02_325_Data_Error);
        fun("TBOX_Fr01_3E0_Data_Error", TBOX_Fr01_3E0_Data_Error);
        fun("EDU_Fr06_3E5_Data_Error", EDU_Fr06_3E5_Data_Error);
        fun("CDCS_Fr15_203_Data_Error", CDCS_Fr15_203_Data_Error);
        fun("AICS_Fr2_243_Data_Error", AICS_Fr2_243_Data_Error);
        fun("BMS_Fr03_258_Data_Error", BMS_Fr03_258_Data_Error);
        fun("BDCS_Fr13_LIN2_285_Data_Error", BDCS_Fr13_LIN2_285_Data_Error);
        fun("VCU_Fr04_2D4_Data_Error", VCU_Fr04_2D4_Data_Error);
        fun("VCU_Fr03_2E3_Data_Error", VCU_Fr03_2E3_Data_Error);
        fun("VCU_Fr07_0FC_Data_Error", VCU_Fr07_0FC_Data_Error);
        fun("EDU_Fr05_1D5_Data_Error", EDU_Fr05_1D5_Data_Error);
        fun("BDCS_Fr01_110_Data2_Error", BDCS_Fr01_110_Data2_Error);
        fun("HNS2_Fr02_2FE_Data_Error", HNS2_Fr02_2FE_Data_Error);
        fun("ACU_Node_Lost", ACU_Node_Lost);
        fun("BDCS_Node_Lost", BDCS_Node_Lost);
        fun("BTM_Node_Lost", BTM_Node_Lost);
        fun("CDCS_Node_Lost", CDCS_Node_Lost);
        fun("DDCU_Node_Lost", DDCU_Node_Lost);
        fun("EDU_Node_Lost", EDU_Node_Lost);
        fun("EPS_Node_Lost", EPS_Node_Lost);
        fun("FMCU_Node_Lost", FMCU_Node_Lost);
        fun("GW_Node_Lost", GW_Node_Lost);
        fun("HNS_Node_Lost", HNS_Node_Lost);
        fun("ICU_Node_Lost", ICU_Node_Lost);
        fun("IDB_Node_Lost", IDB_Node_Lost);
        fun("MCU_Node_Lost", MCU_Node_Lost);
        fun("PDCU_Node_Lost", PDCU_Node_Lost);
        fun("RCU_Node_Lost", RCU_Node_Lost);
        fun("TBOX_Node_Lost", TBOX_Node_Lost);
        fun("PDCS_Node_Lost", PDCS_Node_Lost);
        fun("BMS_Node_Lost", BMS_Node_Lost);
        fun("FD3_Bus_Error", FD3_Bus_Error);
        fun("FD6_Bus_Error", FD6_Bus_Error);
        fun("FD8_Bus_Error", FD8_Bus_Error);
        fun("Innr_Bus_Error", Innr_Bus_Error);
        fun("Innr_MsgLost_Error", Innr_MsgLost_Error);
        fun("Innr_MsgData_Error", Innr_MsgData_Error);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("MCU_Fr01_0AA_Data_Error", MCU_Fr01_0AA_Data_Error);
        fun("FMCU_Fr01_0AB_Data_Error", FMCU_Fr01_0AB_Data_Error);
        fun("EPS_Fr01_0B1_Data_Error", EPS_Fr01_0B1_Data_Error);
        fun("EPS_Fr02_0B2_Data_Error", EPS_Fr02_0B2_Data_Error);
        fun("IDB_Fr02_0C0_Data_Error", IDB_Fr02_0C0_Data_Error);
        fun("ACU_Fr02_0C4_Data_Error", ACU_Fr02_0C4_Data_Error);
        fun("IDB_Fr03_0C5_Data_Error", IDB_Fr03_0C5_Data_Error);
        fun("IDB_Fr04_0C7_Data_Error", IDB_Fr04_0C7_Data_Error);
        fun("IDB_Fr10_0C9_Data_Error", IDB_Fr10_0C9_Data_Error);
        fun("VCU_Fr05_0E3_Data_Error", VCU_Fr05_0E3_Data_Error);
        fun("IDB_Fr01_0E5_Data_Error", IDB_Fr01_0E5_Data_Error);
        fun("RCU_Fr01_0E6_Data_Error", RCU_Fr01_0E6_Data_Error);
        fun("BTM_Fr01_0F3_Data_Error", BTM_Fr01_0F3_Data_Error);
        fun("VCU_Fr06_0FB_Data_Error", VCU_Fr06_0FB_Data_Error);
        fun("ACU_Fr01_102_Data_Error", ACU_Fr01_102_Data_Error);
        fun("VCU_Fr0E_108_Data_Error", VCU_Fr0E_108_Data_Error);
        fun("BDCS_Fr01_110_Data1_Error", BDCS_Fr01_110_Data1_Error);
        fun("BDCS_Fr03_112_Data_Error", BDCS_Fr03_112_Data_Error);
        fun("BDCS_Fr16_114_Data_Error", BDCS_Fr16_114_Data_Error);
        fun("IDB_Fr09_121_Data_Error", IDB_Fr09_121_Data_Error);
        fun("IDB_Fr05_129_Data_Error", IDB_Fr05_129_Data_Error);
        fun("IDB_Fr06_12B_Data_Error", IDB_Fr06_12B_Data_Error);
        fun("IDB_Fr07_12D_Data_Error", IDB_Fr07_12D_Data_Error);
        fun("ICU_Fr01_131_Data_Error", ICU_Fr01_131_Data_Error);
        fun("RCU_Fr02_137_Data_Error", RCU_Fr02_137_Data_Error);
        fun("RCU_Fr03_138_Data_Error", RCU_Fr03_138_Data_Error);
        fun("RCU_Fr04_139_Data_Error", RCU_Fr04_139_Data_Error);
        fun("RCU_Fr06_13A_Data_Error", RCU_Fr06_13A_Data_Error);
        fun("RCU_Fr07_13B_Data_Error", RCU_Fr07_13B_Data_Error);
        fun("RCU_Fr08_13C_Data_Error", RCU_Fr08_13C_Data_Error);
        fun("RCU_Fr05_13D_Data_Error", RCU_Fr05_13D_Data_Error);
        fun("RCU_Fr09_13E_Data_Error", RCU_Fr09_13E_Data_Error);
        fun("GW_Fr01_15B_Data_Error", GW_Fr01_15B_Data_Error);
        fun("DDCU_Fr01_1A2_Data_Error", DDCU_Fr01_1A2_Data_Error);
        fun("PDCU_Fr01_1A3_Data_Error", PDCU_Fr01_1A3_Data_Error);
        fun("TBOX_Fr02_1B6_Data_Error", TBOX_Fr02_1B6_Data_Error);
        fun("CDCS_Fr10_1BA_Data_Error", CDCS_Fr10_1BA_Data_Error);
        fun("CDCS_Fr13_1BB_Data_Error", CDCS_Fr13_1BB_Data_Error);
        fun("BDCS_Fr04_1E2_Data_Error", BDCS_Fr04_1E2_Data_Error);
        fun("BDCS_Fr05_1E4_Data_Error", BDCS_Fr05_1E4_Data_Error);
        fun("BDCS_Fr06_1E6_Data_Error", BDCS_Fr06_1E6_Data_Error);
        fun("BDCS_Fr10_29E_Data_Error", BDCS_Fr10_29E_Data_Error);
        fun("EDU_Fr03_2C2_Data_Error", EDU_Fr03_2C2_Data_Error);
        fun("EDU_Fr04_2C3_Data_Error", EDU_Fr04_2C3_Data_Error);
        fun("BTM_Fr02_2F4_Data_Error", BTM_Fr02_2F4_Data_Error);
        fun("HNS1_Fr02_2FD_Data_Error", HNS1_Fr02_2FD_Data_Error);
        fun("ICU_Fr02_325_Data_Error", ICU_Fr02_325_Data_Error);
        fun("TBOX_Fr01_3E0_Data_Error", TBOX_Fr01_3E0_Data_Error);
        fun("EDU_Fr06_3E5_Data_Error", EDU_Fr06_3E5_Data_Error);
        fun("CDCS_Fr15_203_Data_Error", CDCS_Fr15_203_Data_Error);
        fun("AICS_Fr2_243_Data_Error", AICS_Fr2_243_Data_Error);
        fun("BMS_Fr03_258_Data_Error", BMS_Fr03_258_Data_Error);
        fun("BDCS_Fr13_LIN2_285_Data_Error", BDCS_Fr13_LIN2_285_Data_Error);
        fun("VCU_Fr04_2D4_Data_Error", VCU_Fr04_2D4_Data_Error);
        fun("VCU_Fr03_2E3_Data_Error", VCU_Fr03_2E3_Data_Error);
        fun("VCU_Fr07_0FC_Data_Error", VCU_Fr07_0FC_Data_Error);
        fun("EDU_Fr05_1D5_Data_Error", EDU_Fr05_1D5_Data_Error);
        fun("BDCS_Fr01_110_Data2_Error", BDCS_Fr01_110_Data2_Error);
        fun("HNS2_Fr02_2FE_Data_Error", HNS2_Fr02_2FE_Data_Error);
        fun("ACU_Node_Lost", ACU_Node_Lost);
        fun("BDCS_Node_Lost", BDCS_Node_Lost);
        fun("BTM_Node_Lost", BTM_Node_Lost);
        fun("CDCS_Node_Lost", CDCS_Node_Lost);
        fun("DDCU_Node_Lost", DDCU_Node_Lost);
        fun("EDU_Node_Lost", EDU_Node_Lost);
        fun("EPS_Node_Lost", EPS_Node_Lost);
        fun("FMCU_Node_Lost", FMCU_Node_Lost);
        fun("GW_Node_Lost", GW_Node_Lost);
        fun("HNS_Node_Lost", HNS_Node_Lost);
        fun("ICU_Node_Lost", ICU_Node_Lost);
        fun("IDB_Node_Lost", IDB_Node_Lost);
        fun("MCU_Node_Lost", MCU_Node_Lost);
        fun("PDCU_Node_Lost", PDCU_Node_Lost);
        fun("RCU_Node_Lost", RCU_Node_Lost);
        fun("TBOX_Node_Lost", TBOX_Node_Lost);
        fun("PDCS_Node_Lost", PDCS_Node_Lost);
        fun("BMS_Node_Lost", BMS_Node_Lost);
        fun("FD3_Bus_Error", FD3_Bus_Error);
        fun("FD6_Bus_Error", FD6_Bus_Error);
        fun("FD8_Bus_Error", FD8_Bus_Error);
        fun("Innr_Bus_Error", Innr_Bus_Error);
        fun("Innr_MsgLost_Error", Innr_MsgLost_Error);
        fun("Innr_MsgData_Error", Innr_MsgData_Error);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_HafcomDataFltInfo& t) const
    {
        return (MCU_Fr01_0AA_Data_Error == t.MCU_Fr01_0AA_Data_Error) && (FMCU_Fr01_0AB_Data_Error == t.FMCU_Fr01_0AB_Data_Error) && (EPS_Fr01_0B1_Data_Error == t.EPS_Fr01_0B1_Data_Error) && (EPS_Fr02_0B2_Data_Error == t.EPS_Fr02_0B2_Data_Error) && (IDB_Fr02_0C0_Data_Error == t.IDB_Fr02_0C0_Data_Error) && (ACU_Fr02_0C4_Data_Error == t.ACU_Fr02_0C4_Data_Error) && (IDB_Fr03_0C5_Data_Error == t.IDB_Fr03_0C5_Data_Error) && (IDB_Fr04_0C7_Data_Error == t.IDB_Fr04_0C7_Data_Error) && (IDB_Fr10_0C9_Data_Error == t.IDB_Fr10_0C9_Data_Error) && (VCU_Fr05_0E3_Data_Error == t.VCU_Fr05_0E3_Data_Error) && (IDB_Fr01_0E5_Data_Error == t.IDB_Fr01_0E5_Data_Error) && (RCU_Fr01_0E6_Data_Error == t.RCU_Fr01_0E6_Data_Error) && (BTM_Fr01_0F3_Data_Error == t.BTM_Fr01_0F3_Data_Error) && (VCU_Fr06_0FB_Data_Error == t.VCU_Fr06_0FB_Data_Error) && (ACU_Fr01_102_Data_Error == t.ACU_Fr01_102_Data_Error) && (VCU_Fr0E_108_Data_Error == t.VCU_Fr0E_108_Data_Error) && (BDCS_Fr01_110_Data1_Error == t.BDCS_Fr01_110_Data1_Error) && (BDCS_Fr03_112_Data_Error == t.BDCS_Fr03_112_Data_Error) && (BDCS_Fr16_114_Data_Error == t.BDCS_Fr16_114_Data_Error) && (IDB_Fr09_121_Data_Error == t.IDB_Fr09_121_Data_Error) && (IDB_Fr05_129_Data_Error == t.IDB_Fr05_129_Data_Error) && (IDB_Fr06_12B_Data_Error == t.IDB_Fr06_12B_Data_Error) && (IDB_Fr07_12D_Data_Error == t.IDB_Fr07_12D_Data_Error) && (ICU_Fr01_131_Data_Error == t.ICU_Fr01_131_Data_Error) && (RCU_Fr02_137_Data_Error == t.RCU_Fr02_137_Data_Error) && (RCU_Fr03_138_Data_Error == t.RCU_Fr03_138_Data_Error) && (RCU_Fr04_139_Data_Error == t.RCU_Fr04_139_Data_Error) && (RCU_Fr06_13A_Data_Error == t.RCU_Fr06_13A_Data_Error) && (RCU_Fr07_13B_Data_Error == t.RCU_Fr07_13B_Data_Error) && (RCU_Fr08_13C_Data_Error == t.RCU_Fr08_13C_Data_Error) && (RCU_Fr05_13D_Data_Error == t.RCU_Fr05_13D_Data_Error) && (RCU_Fr09_13E_Data_Error == t.RCU_Fr09_13E_Data_Error) && (GW_Fr01_15B_Data_Error == t.GW_Fr01_15B_Data_Error) && (DDCU_Fr01_1A2_Data_Error == t.DDCU_Fr01_1A2_Data_Error) && (PDCU_Fr01_1A3_Data_Error == t.PDCU_Fr01_1A3_Data_Error) && (TBOX_Fr02_1B6_Data_Error == t.TBOX_Fr02_1B6_Data_Error) && (CDCS_Fr10_1BA_Data_Error == t.CDCS_Fr10_1BA_Data_Error) && (CDCS_Fr13_1BB_Data_Error == t.CDCS_Fr13_1BB_Data_Error) && (BDCS_Fr04_1E2_Data_Error == t.BDCS_Fr04_1E2_Data_Error) && (BDCS_Fr05_1E4_Data_Error == t.BDCS_Fr05_1E4_Data_Error) && (BDCS_Fr06_1E6_Data_Error == t.BDCS_Fr06_1E6_Data_Error) && (BDCS_Fr10_29E_Data_Error == t.BDCS_Fr10_29E_Data_Error) && (EDU_Fr03_2C2_Data_Error == t.EDU_Fr03_2C2_Data_Error) && (EDU_Fr04_2C3_Data_Error == t.EDU_Fr04_2C3_Data_Error) && (BTM_Fr02_2F4_Data_Error == t.BTM_Fr02_2F4_Data_Error) && (HNS1_Fr02_2FD_Data_Error == t.HNS1_Fr02_2FD_Data_Error) && (ICU_Fr02_325_Data_Error == t.ICU_Fr02_325_Data_Error) && (TBOX_Fr01_3E0_Data_Error == t.TBOX_Fr01_3E0_Data_Error) && (EDU_Fr06_3E5_Data_Error == t.EDU_Fr06_3E5_Data_Error) && (CDCS_Fr15_203_Data_Error == t.CDCS_Fr15_203_Data_Error) && (AICS_Fr2_243_Data_Error == t.AICS_Fr2_243_Data_Error) && (BMS_Fr03_258_Data_Error == t.BMS_Fr03_258_Data_Error) && (BDCS_Fr13_LIN2_285_Data_Error == t.BDCS_Fr13_LIN2_285_Data_Error) && (VCU_Fr04_2D4_Data_Error == t.VCU_Fr04_2D4_Data_Error) && (VCU_Fr03_2E3_Data_Error == t.VCU_Fr03_2E3_Data_Error) && (VCU_Fr07_0FC_Data_Error == t.VCU_Fr07_0FC_Data_Error) && (EDU_Fr05_1D5_Data_Error == t.EDU_Fr05_1D5_Data_Error) && (BDCS_Fr01_110_Data2_Error == t.BDCS_Fr01_110_Data2_Error) && (HNS2_Fr02_2FE_Data_Error == t.HNS2_Fr02_2FE_Data_Error) && (ACU_Node_Lost == t.ACU_Node_Lost) && (BDCS_Node_Lost == t.BDCS_Node_Lost) && (BTM_Node_Lost == t.BTM_Node_Lost) && (CDCS_Node_Lost == t.CDCS_Node_Lost) && (DDCU_Node_Lost == t.DDCU_Node_Lost) && (EDU_Node_Lost == t.EDU_Node_Lost) && (EPS_Node_Lost == t.EPS_Node_Lost) && (FMCU_Node_Lost == t.FMCU_Node_Lost) && (GW_Node_Lost == t.GW_Node_Lost) && (HNS_Node_Lost == t.HNS_Node_Lost) && (ICU_Node_Lost == t.ICU_Node_Lost) && (IDB_Node_Lost == t.IDB_Node_Lost) && (MCU_Node_Lost == t.MCU_Node_Lost) && (PDCU_Node_Lost == t.PDCU_Node_Lost) && (RCU_Node_Lost == t.RCU_Node_Lost) && (TBOX_Node_Lost == t.TBOX_Node_Lost) && (PDCS_Node_Lost == t.PDCS_Node_Lost) && (BMS_Node_Lost == t.BMS_Node_Lost) && (FD3_Bus_Error == t.FD3_Bus_Error) && (FD6_Bus_Error == t.FD6_Bus_Error) && (FD8_Bus_Error == t.FD8_Bus_Error) && (Innr_Bus_Error == t.Innr_Bus_Error) && (Innr_MsgLost_Error == t.Innr_MsgLost_Error) && (Innr_MsgData_Error == t.Innr_MsgData_Error);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFCOMDATAFLTINFO_H
