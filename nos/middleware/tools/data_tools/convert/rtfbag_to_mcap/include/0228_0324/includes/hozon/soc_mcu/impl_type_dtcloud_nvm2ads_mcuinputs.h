/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_NVM2ADS_MCUINPUTS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_NVM2ADS_MCUINPUTS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_NVM2ADS_MCUInputs {
    ::UInt8 LDWMode;
    ::UInt8 LKSSetspeed;
    ::UInt8 LDPMode;
    ::UInt8 VoiceMode;
    ::UInt8 ACCOnOffState;
    ::UInt8 DCLCSysState;
    ::UInt8 NNP_State;
    ::UInt8 AutoOnOffSet;
    ::UInt8 ALC_mode;
    ::UInt8 ADSDriving_mode;
    ::UInt8 TSR_SLFStatefeedback;
    ::UInt32 F170FuncConf;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(LDWMode);
        fun(LKSSetspeed);
        fun(LDPMode);
        fun(VoiceMode);
        fun(ACCOnOffState);
        fun(DCLCSysState);
        fun(NNP_State);
        fun(AutoOnOffSet);
        fun(ALC_mode);
        fun(ADSDriving_mode);
        fun(TSR_SLFStatefeedback);
        fun(F170FuncConf);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(LDWMode);
        fun(LKSSetspeed);
        fun(LDPMode);
        fun(VoiceMode);
        fun(ACCOnOffState);
        fun(DCLCSysState);
        fun(NNP_State);
        fun(AutoOnOffSet);
        fun(ALC_mode);
        fun(ADSDriving_mode);
        fun(TSR_SLFStatefeedback);
        fun(F170FuncConf);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("LDWMode", LDWMode);
        fun("LKSSetspeed", LKSSetspeed);
        fun("LDPMode", LDPMode);
        fun("VoiceMode", VoiceMode);
        fun("ACCOnOffState", ACCOnOffState);
        fun("DCLCSysState", DCLCSysState);
        fun("NNP_State", NNP_State);
        fun("AutoOnOffSet", AutoOnOffSet);
        fun("ALC_mode", ALC_mode);
        fun("ADSDriving_mode", ADSDriving_mode);
        fun("TSR_SLFStatefeedback", TSR_SLFStatefeedback);
        fun("F170FuncConf", F170FuncConf);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("LDWMode", LDWMode);
        fun("LKSSetspeed", LKSSetspeed);
        fun("LDPMode", LDPMode);
        fun("VoiceMode", VoiceMode);
        fun("ACCOnOffState", ACCOnOffState);
        fun("DCLCSysState", DCLCSysState);
        fun("NNP_State", NNP_State);
        fun("AutoOnOffSet", AutoOnOffSet);
        fun("ALC_mode", ALC_mode);
        fun("ADSDriving_mode", ADSDriving_mode);
        fun("TSR_SLFStatefeedback", TSR_SLFStatefeedback);
        fun("F170FuncConf", F170FuncConf);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_NVM2ADS_MCUInputs& t) const
    {
        return (LDWMode == t.LDWMode) && (LKSSetspeed == t.LKSSetspeed) && (LDPMode == t.LDPMode) && (VoiceMode == t.VoiceMode) && (ACCOnOffState == t.ACCOnOffState) && (DCLCSysState == t.DCLCSysState) && (NNP_State == t.NNP_State) && (AutoOnOffSet == t.AutoOnOffSet) && (ALC_mode == t.ALC_mode) && (ADSDriving_mode == t.ADSDriving_mode) && (TSR_SLFStatefeedback == t.TSR_SLFStatefeedback) && (F170FuncConf == t.F170FuncConf);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_NVM2ADS_MCUINPUTS_H
