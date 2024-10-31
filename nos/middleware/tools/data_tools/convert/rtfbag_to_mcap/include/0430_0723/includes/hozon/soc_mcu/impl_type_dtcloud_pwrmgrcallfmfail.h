/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWRMGRCALLFMFAIL_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWRMGRCALLFMFAIL_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_PwrMgrCallFMFail {
    ::UInt8 SOCPowerOffCallFMFail;
    ::UInt8 SOCPowerOffCallFMFailCount;
    ::UInt8 SOCPowerOnCallFMFail;
    ::UInt8 SOCPowerOnCallFMFailCount;
    ::UInt8 MCUPowerOffCallFMFail;
    ::UInt8 MCUPowerOffCallFMFailCount;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SOCPowerOffCallFMFail);
        fun(SOCPowerOffCallFMFailCount);
        fun(SOCPowerOnCallFMFail);
        fun(SOCPowerOnCallFMFailCount);
        fun(MCUPowerOffCallFMFail);
        fun(MCUPowerOffCallFMFailCount);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SOCPowerOffCallFMFail);
        fun(SOCPowerOffCallFMFailCount);
        fun(SOCPowerOnCallFMFail);
        fun(SOCPowerOnCallFMFailCount);
        fun(MCUPowerOffCallFMFail);
        fun(MCUPowerOffCallFMFailCount);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("SOCPowerOffCallFMFail", SOCPowerOffCallFMFail);
        fun("SOCPowerOffCallFMFailCount", SOCPowerOffCallFMFailCount);
        fun("SOCPowerOnCallFMFail", SOCPowerOnCallFMFail);
        fun("SOCPowerOnCallFMFailCount", SOCPowerOnCallFMFailCount);
        fun("MCUPowerOffCallFMFail", MCUPowerOffCallFMFail);
        fun("MCUPowerOffCallFMFailCount", MCUPowerOffCallFMFailCount);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("SOCPowerOffCallFMFail", SOCPowerOffCallFMFail);
        fun("SOCPowerOffCallFMFailCount", SOCPowerOffCallFMFailCount);
        fun("SOCPowerOnCallFMFail", SOCPowerOnCallFMFail);
        fun("SOCPowerOnCallFMFailCount", SOCPowerOnCallFMFailCount);
        fun("MCUPowerOffCallFMFail", MCUPowerOffCallFMFail);
        fun("MCUPowerOffCallFMFailCount", MCUPowerOffCallFMFailCount);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_PwrMgrCallFMFail& t) const
    {
        return (SOCPowerOffCallFMFail == t.SOCPowerOffCallFMFail) && (SOCPowerOffCallFMFailCount == t.SOCPowerOffCallFMFailCount) && (SOCPowerOnCallFMFail == t.SOCPowerOnCallFMFail) && (SOCPowerOnCallFMFailCount == t.SOCPowerOnCallFMFailCount) && (MCUPowerOffCallFMFail == t.MCUPowerOffCallFMFail) && (MCUPowerOffCallFMFailCount == t.MCUPowerOffCallFMFailCount);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWRMGRCALLFMFAIL_H
