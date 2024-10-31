/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_NVMBLOCKSTATE_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_NVMBLOCKSTATE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_NVMBlockState {
    ::UInt8 FBL_BootData_State;
    ::UInt8 SecOc_State;
    ::UInt8 BSWConfig_State;
    ::UInt8 System_ResetWakeup_State;
    ::UInt8 System_Awake_State;
    ::UInt8 ASW_RemmberState_State;
    ::UInt8 System_Cfg0_State;
    ::UInt8 System_Cfg1_State;
    ::UInt8 DTC_Information0_State;
    ::UInt8 DTC_Information1_State;
    ::UInt8 DTC_Information2_State;
    ::UInt8 DTC_Information3_State;
    ::UInt8 Reserved0_State;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(FBL_BootData_State);
        fun(SecOc_State);
        fun(BSWConfig_State);
        fun(System_ResetWakeup_State);
        fun(System_Awake_State);
        fun(ASW_RemmberState_State);
        fun(System_Cfg0_State);
        fun(System_Cfg1_State);
        fun(DTC_Information0_State);
        fun(DTC_Information1_State);
        fun(DTC_Information2_State);
        fun(DTC_Information3_State);
        fun(Reserved0_State);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(FBL_BootData_State);
        fun(SecOc_State);
        fun(BSWConfig_State);
        fun(System_ResetWakeup_State);
        fun(System_Awake_State);
        fun(ASW_RemmberState_State);
        fun(System_Cfg0_State);
        fun(System_Cfg1_State);
        fun(DTC_Information0_State);
        fun(DTC_Information1_State);
        fun(DTC_Information2_State);
        fun(DTC_Information3_State);
        fun(Reserved0_State);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("FBL_BootData_State", FBL_BootData_State);
        fun("SecOc_State", SecOc_State);
        fun("BSWConfig_State", BSWConfig_State);
        fun("System_ResetWakeup_State", System_ResetWakeup_State);
        fun("System_Awake_State", System_Awake_State);
        fun("ASW_RemmberState_State", ASW_RemmberState_State);
        fun("System_Cfg0_State", System_Cfg0_State);
        fun("System_Cfg1_State", System_Cfg1_State);
        fun("DTC_Information0_State", DTC_Information0_State);
        fun("DTC_Information1_State", DTC_Information1_State);
        fun("DTC_Information2_State", DTC_Information2_State);
        fun("DTC_Information3_State", DTC_Information3_State);
        fun("Reserved0_State", Reserved0_State);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("FBL_BootData_State", FBL_BootData_State);
        fun("SecOc_State", SecOc_State);
        fun("BSWConfig_State", BSWConfig_State);
        fun("System_ResetWakeup_State", System_ResetWakeup_State);
        fun("System_Awake_State", System_Awake_State);
        fun("ASW_RemmberState_State", ASW_RemmberState_State);
        fun("System_Cfg0_State", System_Cfg0_State);
        fun("System_Cfg1_State", System_Cfg1_State);
        fun("DTC_Information0_State", DTC_Information0_State);
        fun("DTC_Information1_State", DTC_Information1_State);
        fun("DTC_Information2_State", DTC_Information2_State);
        fun("DTC_Information3_State", DTC_Information3_State);
        fun("Reserved0_State", Reserved0_State);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_NVMBlockState& t) const
    {
        return (FBL_BootData_State == t.FBL_BootData_State) && (SecOc_State == t.SecOc_State) && (BSWConfig_State == t.BSWConfig_State) && (System_ResetWakeup_State == t.System_ResetWakeup_State) && (System_Awake_State == t.System_Awake_State) && (ASW_RemmberState_State == t.ASW_RemmberState_State) && (System_Cfg0_State == t.System_Cfg0_State) && (System_Cfg1_State == t.System_Cfg1_State) && (DTC_Information0_State == t.DTC_Information0_State) && (DTC_Information1_State == t.DTC_Information1_State) && (DTC_Information2_State == t.DTC_Information2_State) && (DTC_Information3_State == t.DTC_Information3_State) && (Reserved0_State == t.Reserved0_State);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_NVMBLOCKSTATE_H
