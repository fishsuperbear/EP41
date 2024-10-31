/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_NVM_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_NVM_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtcloud_hafcan2nvm.h"
#include "hozon/soc_mcu/impl_type_dtcloud_ads2nvm_funmode.h"
#include "hozon/soc_mcu/impl_type_dtcloud_ctrl_nvm.h"
#include "hozon/soc_mcu/impl_type_dtcloud_hafaeb2nvm.h"
#include "impl_type_uint8.h"
#include "hozon/soc_mcu/impl_type_dtcloud_nvm2ads_mcuinputs.h"
#include "hozon/soc_mcu/impl_type_dtcloud_hafswc2aeb.h"
#include "hozon/soc_mcu/impl_type_dtcloud_egomemmsg.h"
#include "hozon/soc_mcu/impl_type_dtcloud_nvmblockstate.h"
#include "hozon/soc_mcu/impl_type_dtcloud_rb_nvm_asw_remmberstate.h"
#include "impl_type_uint32.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_NVM {
    ::hozon::soc_mcu::DtCloud_HafCAN2NVM CAN_NVM_SOCData;
    ::hozon::soc_mcu::DtCloud_ADS2NVM_FunMode ADS_NVM_Data;
    ::hozon::soc_mcu::DtCloud_Ctrl_NVM Ctrl_NVM_Data;
    ::hozon::soc_mcu::DtCloud_HafAEB2NVM AEB_NVM_Data;
    ::UInt8 VehicleCfgD44;
    ::UInt8 VehicleCfgD45;
    ::UInt8 VehicleCfgD47;
    ::UInt8 VehicleCfgD48;
    ::hozon::soc_mcu::DtCloud_NVM2ADS_MCUInputs NVM_ADS_Data;
    ::hozon::soc_mcu::DtCloud_Ctrl_NVM NVM_Ctrl_Data;
    ::hozon::soc_mcu::DtCloud_HafSwc2AEB NVM_AEB_Data;
    ::hozon::soc_mcu::DtCloud_EgoMemMsg NVM_ETH_SOCData;
    ::hozon::soc_mcu::DtCloud_NVMBlockState Nvm_Blcok_StateEth;
    ::hozon::soc_mcu::DtCloud_rb_NvM_ASW_RemmberState NvM_ASW_RemmberState_Data;
    ::hozon::soc_mcu::DtCloud_rb_NvM_ASW_RemmberState App_ASW_RemmberState_Data;
    ::UInt32 TaskCntNVMread_10ms;
    ::UInt32 TaskCntNVMwrite_10ms;
    ::UInt32 TaskCntNVMmainfunction_10ms;
    ::UInt8 EthCloud_WR_NVM_Count;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(CAN_NVM_SOCData);
        fun(ADS_NVM_Data);
        fun(Ctrl_NVM_Data);
        fun(AEB_NVM_Data);
        fun(VehicleCfgD44);
        fun(VehicleCfgD45);
        fun(VehicleCfgD47);
        fun(VehicleCfgD48);
        fun(NVM_ADS_Data);
        fun(NVM_Ctrl_Data);
        fun(NVM_AEB_Data);
        fun(NVM_ETH_SOCData);
        fun(Nvm_Blcok_StateEth);
        fun(NvM_ASW_RemmberState_Data);
        fun(App_ASW_RemmberState_Data);
        fun(TaskCntNVMread_10ms);
        fun(TaskCntNVMwrite_10ms);
        fun(TaskCntNVMmainfunction_10ms);
        fun(EthCloud_WR_NVM_Count);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(CAN_NVM_SOCData);
        fun(ADS_NVM_Data);
        fun(Ctrl_NVM_Data);
        fun(AEB_NVM_Data);
        fun(VehicleCfgD44);
        fun(VehicleCfgD45);
        fun(VehicleCfgD47);
        fun(VehicleCfgD48);
        fun(NVM_ADS_Data);
        fun(NVM_Ctrl_Data);
        fun(NVM_AEB_Data);
        fun(NVM_ETH_SOCData);
        fun(Nvm_Blcok_StateEth);
        fun(NvM_ASW_RemmberState_Data);
        fun(App_ASW_RemmberState_Data);
        fun(TaskCntNVMread_10ms);
        fun(TaskCntNVMwrite_10ms);
        fun(TaskCntNVMmainfunction_10ms);
        fun(EthCloud_WR_NVM_Count);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("CAN_NVM_SOCData", CAN_NVM_SOCData);
        fun("ADS_NVM_Data", ADS_NVM_Data);
        fun("Ctrl_NVM_Data", Ctrl_NVM_Data);
        fun("AEB_NVM_Data", AEB_NVM_Data);
        fun("VehicleCfgD44", VehicleCfgD44);
        fun("VehicleCfgD45", VehicleCfgD45);
        fun("VehicleCfgD47", VehicleCfgD47);
        fun("VehicleCfgD48", VehicleCfgD48);
        fun("NVM_ADS_Data", NVM_ADS_Data);
        fun("NVM_Ctrl_Data", NVM_Ctrl_Data);
        fun("NVM_AEB_Data", NVM_AEB_Data);
        fun("NVM_ETH_SOCData", NVM_ETH_SOCData);
        fun("Nvm_Blcok_StateEth", Nvm_Blcok_StateEth);
        fun("NvM_ASW_RemmberState_Data", NvM_ASW_RemmberState_Data);
        fun("App_ASW_RemmberState_Data", App_ASW_RemmberState_Data);
        fun("TaskCntNVMread_10ms", TaskCntNVMread_10ms);
        fun("TaskCntNVMwrite_10ms", TaskCntNVMwrite_10ms);
        fun("TaskCntNVMmainfunction_10ms", TaskCntNVMmainfunction_10ms);
        fun("EthCloud_WR_NVM_Count", EthCloud_WR_NVM_Count);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("CAN_NVM_SOCData", CAN_NVM_SOCData);
        fun("ADS_NVM_Data", ADS_NVM_Data);
        fun("Ctrl_NVM_Data", Ctrl_NVM_Data);
        fun("AEB_NVM_Data", AEB_NVM_Data);
        fun("VehicleCfgD44", VehicleCfgD44);
        fun("VehicleCfgD45", VehicleCfgD45);
        fun("VehicleCfgD47", VehicleCfgD47);
        fun("VehicleCfgD48", VehicleCfgD48);
        fun("NVM_ADS_Data", NVM_ADS_Data);
        fun("NVM_Ctrl_Data", NVM_Ctrl_Data);
        fun("NVM_AEB_Data", NVM_AEB_Data);
        fun("NVM_ETH_SOCData", NVM_ETH_SOCData);
        fun("Nvm_Blcok_StateEth", Nvm_Blcok_StateEth);
        fun("NvM_ASW_RemmberState_Data", NvM_ASW_RemmberState_Data);
        fun("App_ASW_RemmberState_Data", App_ASW_RemmberState_Data);
        fun("TaskCntNVMread_10ms", TaskCntNVMread_10ms);
        fun("TaskCntNVMwrite_10ms", TaskCntNVMwrite_10ms);
        fun("TaskCntNVMmainfunction_10ms", TaskCntNVMmainfunction_10ms);
        fun("EthCloud_WR_NVM_Count", EthCloud_WR_NVM_Count);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_NVM& t) const
    {
        return (CAN_NVM_SOCData == t.CAN_NVM_SOCData) && (ADS_NVM_Data == t.ADS_NVM_Data) && (Ctrl_NVM_Data == t.Ctrl_NVM_Data) && (AEB_NVM_Data == t.AEB_NVM_Data) && (VehicleCfgD44 == t.VehicleCfgD44) && (VehicleCfgD45 == t.VehicleCfgD45) && (VehicleCfgD47 == t.VehicleCfgD47) && (VehicleCfgD48 == t.VehicleCfgD48) && (NVM_ADS_Data == t.NVM_ADS_Data) && (NVM_Ctrl_Data == t.NVM_Ctrl_Data) && (NVM_AEB_Data == t.NVM_AEB_Data) && (NVM_ETH_SOCData == t.NVM_ETH_SOCData) && (Nvm_Blcok_StateEth == t.Nvm_Blcok_StateEth) && (NvM_ASW_RemmberState_Data == t.NvM_ASW_RemmberState_Data) && (App_ASW_RemmberState_Data == t.App_ASW_RemmberState_Data) && (TaskCntNVMread_10ms == t.TaskCntNVMread_10ms) && (TaskCntNVMwrite_10ms == t.TaskCntNVMwrite_10ms) && (TaskCntNVMmainfunction_10ms == t.TaskCntNVMmainfunction_10ms) && (EthCloud_WR_NVM_Count == t.EthCloud_WR_NVM_Count);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_NVM_H
