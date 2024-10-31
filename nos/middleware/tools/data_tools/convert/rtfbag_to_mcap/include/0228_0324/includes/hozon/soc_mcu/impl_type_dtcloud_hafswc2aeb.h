/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFSWC2AEB_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFSWC2AEB_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HafSwc2AEB {
    ::UInt8 ADCS8_VoiceMode;
    ::UInt8 FCW_WarnTiming_Restore;
    ::UInt8 AEB_OnOff_State;
    ::UInt8 FCW_OnOff_State;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADCS8_VoiceMode);
        fun(FCW_WarnTiming_Restore);
        fun(AEB_OnOff_State);
        fun(FCW_OnOff_State);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADCS8_VoiceMode);
        fun(FCW_WarnTiming_Restore);
        fun(AEB_OnOff_State);
        fun(FCW_OnOff_State);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADCS8_VoiceMode", ADCS8_VoiceMode);
        fun("FCW_WarnTiming_Restore", FCW_WarnTiming_Restore);
        fun("AEB_OnOff_State", AEB_OnOff_State);
        fun("FCW_OnOff_State", FCW_OnOff_State);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADCS8_VoiceMode", ADCS8_VoiceMode);
        fun("FCW_WarnTiming_Restore", FCW_WarnTiming_Restore);
        fun("AEB_OnOff_State", AEB_OnOff_State);
        fun("FCW_OnOff_State", FCW_OnOff_State);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_HafSwc2AEB& t) const
    {
        return (ADCS8_VoiceMode == t.ADCS8_VoiceMode) && (FCW_WarnTiming_Restore == t.FCW_WarnTiming_Restore) && (AEB_OnOff_State == t.AEB_OnOff_State) && (FCW_OnOff_State == t.FCW_OnOff_State);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFSWC2AEB_H
