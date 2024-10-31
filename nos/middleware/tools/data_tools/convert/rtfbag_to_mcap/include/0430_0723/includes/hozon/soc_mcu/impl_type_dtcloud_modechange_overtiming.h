/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_MODECHANGE_OVERTIMING_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_MODECHANGE_OVERTIMING_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_ModeChange_OverTiming {
    ::UInt32 Timing_ToStandByAndShutDown;
    ::UInt32 Timing_ToStandAlone;
    ::UInt32 Timing_ToWorking;
    ::UInt32 Timing_SocForceShutDown;
    ::UInt32 Timing_SocForceReset;
    ::UInt32 Timing_MdcForceShutDown;
    ::UInt32 Timing_MdcForceReset;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Timing_ToStandByAndShutDown);
        fun(Timing_ToStandAlone);
        fun(Timing_ToWorking);
        fun(Timing_SocForceShutDown);
        fun(Timing_SocForceReset);
        fun(Timing_MdcForceShutDown);
        fun(Timing_MdcForceReset);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Timing_ToStandByAndShutDown);
        fun(Timing_ToStandAlone);
        fun(Timing_ToWorking);
        fun(Timing_SocForceShutDown);
        fun(Timing_SocForceReset);
        fun(Timing_MdcForceShutDown);
        fun(Timing_MdcForceReset);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Timing_ToStandByAndShutDown", Timing_ToStandByAndShutDown);
        fun("Timing_ToStandAlone", Timing_ToStandAlone);
        fun("Timing_ToWorking", Timing_ToWorking);
        fun("Timing_SocForceShutDown", Timing_SocForceShutDown);
        fun("Timing_SocForceReset", Timing_SocForceReset);
        fun("Timing_MdcForceShutDown", Timing_MdcForceShutDown);
        fun("Timing_MdcForceReset", Timing_MdcForceReset);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Timing_ToStandByAndShutDown", Timing_ToStandByAndShutDown);
        fun("Timing_ToStandAlone", Timing_ToStandAlone);
        fun("Timing_ToWorking", Timing_ToWorking);
        fun("Timing_SocForceShutDown", Timing_SocForceShutDown);
        fun("Timing_SocForceReset", Timing_SocForceReset);
        fun("Timing_MdcForceShutDown", Timing_MdcForceShutDown);
        fun("Timing_MdcForceReset", Timing_MdcForceReset);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_ModeChange_OverTiming& t) const
    {
        return (Timing_ToStandByAndShutDown == t.Timing_ToStandByAndShutDown) && (Timing_ToStandAlone == t.Timing_ToStandAlone) && (Timing_ToWorking == t.Timing_ToWorking) && (Timing_SocForceShutDown == t.Timing_SocForceShutDown) && (Timing_SocForceReset == t.Timing_SocForceReset) && (Timing_MdcForceShutDown == t.Timing_MdcForceShutDown) && (Timing_MdcForceReset == t.Timing_MdcForceReset);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_MODECHANGE_OVERTIMING_H
