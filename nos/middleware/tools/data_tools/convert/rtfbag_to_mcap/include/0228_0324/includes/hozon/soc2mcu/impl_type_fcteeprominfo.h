/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_FCTEEPROMINFO_H
#define HOZON_SOC2MCU_IMPL_TYPE_FCTEEPROMINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc2mcu {
struct FctEEPromInfo {
    ::UInt8 MCU_stat_LDWMode_u8;
    ::UInt8 MCU_V_LKSSetspeed_u8;
    ::UInt8 MCU_stat_LDPMode_u8;
    ::UInt8 MCU_stat_VoiceMode_u8;
    ::UInt8 MCU_stat_ACCOnOffState_u8;
    ::UInt8 MCU_stat_DCLCSysState_u8;
    ::UInt8 MCU_stat_NNPState_u8;
    ::UInt8 MCU_stat_AutoOnOffSet_u8;
    ::UInt8 MCU_stat_ALCMode_u8;
    ::UInt8 MCU_stat_ADSDrivingMode_u8;
    ::UInt8 MCU_stat_SLFStatefeedback_u8;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(MCU_stat_LDWMode_u8);
        fun(MCU_V_LKSSetspeed_u8);
        fun(MCU_stat_LDPMode_u8);
        fun(MCU_stat_VoiceMode_u8);
        fun(MCU_stat_ACCOnOffState_u8);
        fun(MCU_stat_DCLCSysState_u8);
        fun(MCU_stat_NNPState_u8);
        fun(MCU_stat_AutoOnOffSet_u8);
        fun(MCU_stat_ALCMode_u8);
        fun(MCU_stat_ADSDrivingMode_u8);
        fun(MCU_stat_SLFStatefeedback_u8);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(MCU_stat_LDWMode_u8);
        fun(MCU_V_LKSSetspeed_u8);
        fun(MCU_stat_LDPMode_u8);
        fun(MCU_stat_VoiceMode_u8);
        fun(MCU_stat_ACCOnOffState_u8);
        fun(MCU_stat_DCLCSysState_u8);
        fun(MCU_stat_NNPState_u8);
        fun(MCU_stat_AutoOnOffSet_u8);
        fun(MCU_stat_ALCMode_u8);
        fun(MCU_stat_ADSDrivingMode_u8);
        fun(MCU_stat_SLFStatefeedback_u8);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("MCU_stat_LDWMode_u8", MCU_stat_LDWMode_u8);
        fun("MCU_V_LKSSetspeed_u8", MCU_V_LKSSetspeed_u8);
        fun("MCU_stat_LDPMode_u8", MCU_stat_LDPMode_u8);
        fun("MCU_stat_VoiceMode_u8", MCU_stat_VoiceMode_u8);
        fun("MCU_stat_ACCOnOffState_u8", MCU_stat_ACCOnOffState_u8);
        fun("MCU_stat_DCLCSysState_u8", MCU_stat_DCLCSysState_u8);
        fun("MCU_stat_NNPState_u8", MCU_stat_NNPState_u8);
        fun("MCU_stat_AutoOnOffSet_u8", MCU_stat_AutoOnOffSet_u8);
        fun("MCU_stat_ALCMode_u8", MCU_stat_ALCMode_u8);
        fun("MCU_stat_ADSDrivingMode_u8", MCU_stat_ADSDrivingMode_u8);
        fun("MCU_stat_SLFStatefeedback_u8", MCU_stat_SLFStatefeedback_u8);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("MCU_stat_LDWMode_u8", MCU_stat_LDWMode_u8);
        fun("MCU_V_LKSSetspeed_u8", MCU_V_LKSSetspeed_u8);
        fun("MCU_stat_LDPMode_u8", MCU_stat_LDPMode_u8);
        fun("MCU_stat_VoiceMode_u8", MCU_stat_VoiceMode_u8);
        fun("MCU_stat_ACCOnOffState_u8", MCU_stat_ACCOnOffState_u8);
        fun("MCU_stat_DCLCSysState_u8", MCU_stat_DCLCSysState_u8);
        fun("MCU_stat_NNPState_u8", MCU_stat_NNPState_u8);
        fun("MCU_stat_AutoOnOffSet_u8", MCU_stat_AutoOnOffSet_u8);
        fun("MCU_stat_ALCMode_u8", MCU_stat_ALCMode_u8);
        fun("MCU_stat_ADSDrivingMode_u8", MCU_stat_ADSDrivingMode_u8);
        fun("MCU_stat_SLFStatefeedback_u8", MCU_stat_SLFStatefeedback_u8);
    }

    bool operator==(const ::hozon::soc2mcu::FctEEPromInfo& t) const
    {
        return (MCU_stat_LDWMode_u8 == t.MCU_stat_LDWMode_u8) && (MCU_V_LKSSetspeed_u8 == t.MCU_V_LKSSetspeed_u8) && (MCU_stat_LDPMode_u8 == t.MCU_stat_LDPMode_u8) && (MCU_stat_VoiceMode_u8 == t.MCU_stat_VoiceMode_u8) && (MCU_stat_ACCOnOffState_u8 == t.MCU_stat_ACCOnOffState_u8) && (MCU_stat_DCLCSysState_u8 == t.MCU_stat_DCLCSysState_u8) && (MCU_stat_NNPState_u8 == t.MCU_stat_NNPState_u8) && (MCU_stat_AutoOnOffSet_u8 == t.MCU_stat_AutoOnOffSet_u8) && (MCU_stat_ALCMode_u8 == t.MCU_stat_ALCMode_u8) && (MCU_stat_ADSDrivingMode_u8 == t.MCU_stat_ADSDrivingMode_u8) && (MCU_stat_SLFStatefeedback_u8 == t.MCU_stat_SLFStatefeedback_u8);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_FCTEEPROMINFO_H
