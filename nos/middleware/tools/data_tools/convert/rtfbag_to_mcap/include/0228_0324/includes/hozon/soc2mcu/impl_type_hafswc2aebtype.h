/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_HAFSWC2AEBTYPE_H
#define HOZON_SOC2MCU_IMPL_TYPE_HAFSWC2AEBTYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace soc2mcu {
struct HafSwc2AEBType {
    ::Boolean VoiceMode;
    ::UInt8 FCW_WarnTiming_Restore;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(VoiceMode);
        fun(FCW_WarnTiming_Restore);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(VoiceMode);
        fun(FCW_WarnTiming_Restore);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("VoiceMode", VoiceMode);
        fun("FCW_WarnTiming_Restore", FCW_WarnTiming_Restore);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("VoiceMode", VoiceMode);
        fun("FCW_WarnTiming_Restore", FCW_WarnTiming_Restore);
    }

    bool operator==(const ::hozon::soc2mcu::HafSwc2AEBType& t) const
    {
        return (VoiceMode == t.VoiceMode) && (FCW_WarnTiming_Restore == t.FCW_WarnTiming_Restore);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_HAFSWC2AEBTYPE_H
