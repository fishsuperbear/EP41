/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFCAN2NVM_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFCAN2NVM_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HafCAN2NVM {
    ::UInt8 ADCS8_RCTA_State;
    ::UInt8 ADCS8_FCTA_State;
    ::UInt8 ADCS8_DOWState;
    ::UInt8 ADCS8_RCW_State;
    ::UInt8 ADCS8_LCAState;
    ::UInt8 ADCS8_TSRState;
    ::UInt8 ADCS8_TSR_OverspeedOnOffSet;
    ::UInt8 ADCS8_ADAS_IHBCStat;
    ::UInt8 CDCS5_ResetAllSetup;
    ::UInt8 CDCS5_FactoryReset;
    ::UInt8 ADCS8_VoiceMode;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADCS8_RCTA_State);
        fun(ADCS8_FCTA_State);
        fun(ADCS8_DOWState);
        fun(ADCS8_RCW_State);
        fun(ADCS8_LCAState);
        fun(ADCS8_TSRState);
        fun(ADCS8_TSR_OverspeedOnOffSet);
        fun(ADCS8_ADAS_IHBCStat);
        fun(CDCS5_ResetAllSetup);
        fun(CDCS5_FactoryReset);
        fun(ADCS8_VoiceMode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADCS8_RCTA_State);
        fun(ADCS8_FCTA_State);
        fun(ADCS8_DOWState);
        fun(ADCS8_RCW_State);
        fun(ADCS8_LCAState);
        fun(ADCS8_TSRState);
        fun(ADCS8_TSR_OverspeedOnOffSet);
        fun(ADCS8_ADAS_IHBCStat);
        fun(CDCS5_ResetAllSetup);
        fun(CDCS5_FactoryReset);
        fun(ADCS8_VoiceMode);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADCS8_RCTA_State", ADCS8_RCTA_State);
        fun("ADCS8_FCTA_State", ADCS8_FCTA_State);
        fun("ADCS8_DOWState", ADCS8_DOWState);
        fun("ADCS8_RCW_State", ADCS8_RCW_State);
        fun("ADCS8_LCAState", ADCS8_LCAState);
        fun("ADCS8_TSRState", ADCS8_TSRState);
        fun("ADCS8_TSR_OverspeedOnOffSet", ADCS8_TSR_OverspeedOnOffSet);
        fun("ADCS8_ADAS_IHBCStat", ADCS8_ADAS_IHBCStat);
        fun("CDCS5_ResetAllSetup", CDCS5_ResetAllSetup);
        fun("CDCS5_FactoryReset", CDCS5_FactoryReset);
        fun("ADCS8_VoiceMode", ADCS8_VoiceMode);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADCS8_RCTA_State", ADCS8_RCTA_State);
        fun("ADCS8_FCTA_State", ADCS8_FCTA_State);
        fun("ADCS8_DOWState", ADCS8_DOWState);
        fun("ADCS8_RCW_State", ADCS8_RCW_State);
        fun("ADCS8_LCAState", ADCS8_LCAState);
        fun("ADCS8_TSRState", ADCS8_TSRState);
        fun("ADCS8_TSR_OverspeedOnOffSet", ADCS8_TSR_OverspeedOnOffSet);
        fun("ADCS8_ADAS_IHBCStat", ADCS8_ADAS_IHBCStat);
        fun("CDCS5_ResetAllSetup", CDCS5_ResetAllSetup);
        fun("CDCS5_FactoryReset", CDCS5_FactoryReset);
        fun("ADCS8_VoiceMode", ADCS8_VoiceMode);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_HafCAN2NVM& t) const
    {
        return (ADCS8_RCTA_State == t.ADCS8_RCTA_State) && (ADCS8_FCTA_State == t.ADCS8_FCTA_State) && (ADCS8_DOWState == t.ADCS8_DOWState) && (ADCS8_RCW_State == t.ADCS8_RCW_State) && (ADCS8_LCAState == t.ADCS8_LCAState) && (ADCS8_TSRState == t.ADCS8_TSRState) && (ADCS8_TSR_OverspeedOnOffSet == t.ADCS8_TSR_OverspeedOnOffSet) && (ADCS8_ADAS_IHBCStat == t.ADCS8_ADAS_IHBCStat) && (CDCS5_ResetAllSetup == t.CDCS5_ResetAllSetup) && (CDCS5_FactoryReset == t.CDCS5_FactoryReset) && (ADCS8_VoiceMode == t.ADCS8_VoiceMode);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFCAN2NVM_H
