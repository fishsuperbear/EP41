/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWREVENT_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWREVENT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_PwrEvent {
    ::UInt8 NMMDCDefaultIndication;
    ::UInt8 NMSocWakeUpIndication;
    ::UInt8 NmSocShutDownIndication;
    ::UInt8 NmBusSleepIndication;
    ::UInt8 FmSocForceShutDownIndication;
    ::UInt8 FmSocForceResetIndication;
    ::UInt8 FmMdcForceShutDownIndication;
    ::UInt8 FmMdcForceResetIndication;
    ::UInt8 g_PwrOnSOCFromPwrOff;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(NMMDCDefaultIndication);
        fun(NMSocWakeUpIndication);
        fun(NmSocShutDownIndication);
        fun(NmBusSleepIndication);
        fun(FmSocForceShutDownIndication);
        fun(FmSocForceResetIndication);
        fun(FmMdcForceShutDownIndication);
        fun(FmMdcForceResetIndication);
        fun(g_PwrOnSOCFromPwrOff);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(NMMDCDefaultIndication);
        fun(NMSocWakeUpIndication);
        fun(NmSocShutDownIndication);
        fun(NmBusSleepIndication);
        fun(FmSocForceShutDownIndication);
        fun(FmSocForceResetIndication);
        fun(FmMdcForceShutDownIndication);
        fun(FmMdcForceResetIndication);
        fun(g_PwrOnSOCFromPwrOff);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("NMMDCDefaultIndication", NMMDCDefaultIndication);
        fun("NMSocWakeUpIndication", NMSocWakeUpIndication);
        fun("NmSocShutDownIndication", NmSocShutDownIndication);
        fun("NmBusSleepIndication", NmBusSleepIndication);
        fun("FmSocForceShutDownIndication", FmSocForceShutDownIndication);
        fun("FmSocForceResetIndication", FmSocForceResetIndication);
        fun("FmMdcForceShutDownIndication", FmMdcForceShutDownIndication);
        fun("FmMdcForceResetIndication", FmMdcForceResetIndication);
        fun("g_PwrOnSOCFromPwrOff", g_PwrOnSOCFromPwrOff);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("NMMDCDefaultIndication", NMMDCDefaultIndication);
        fun("NMSocWakeUpIndication", NMSocWakeUpIndication);
        fun("NmSocShutDownIndication", NmSocShutDownIndication);
        fun("NmBusSleepIndication", NmBusSleepIndication);
        fun("FmSocForceShutDownIndication", FmSocForceShutDownIndication);
        fun("FmSocForceResetIndication", FmSocForceResetIndication);
        fun("FmMdcForceShutDownIndication", FmMdcForceShutDownIndication);
        fun("FmMdcForceResetIndication", FmMdcForceResetIndication);
        fun("g_PwrOnSOCFromPwrOff", g_PwrOnSOCFromPwrOff);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_PwrEvent& t) const
    {
        return (NMMDCDefaultIndication == t.NMMDCDefaultIndication) && (NMSocWakeUpIndication == t.NMSocWakeUpIndication) && (NmSocShutDownIndication == t.NmSocShutDownIndication) && (NmBusSleepIndication == t.NmBusSleepIndication) && (FmSocForceShutDownIndication == t.FmSocForceShutDownIndication) && (FmSocForceResetIndication == t.FmSocForceResetIndication) && (FmMdcForceShutDownIndication == t.FmMdcForceShutDownIndication) && (FmMdcForceResetIndication == t.FmMdcForceResetIndication) && (g_PwrOnSOCFromPwrOff == t.g_PwrOnSOCFromPwrOff);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWREVENT_H
