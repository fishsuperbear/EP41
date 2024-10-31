/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_WSTHAFGLOBALTIME_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_WSTHAFGLOBALTIME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_wstHafGlobalTime {
    ::UInt32 TimeStampSec;
    ::UInt32 TimeStampNSec;
    ::UInt8 GetTimeEnable;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(TimeStampSec);
        fun(TimeStampNSec);
        fun(GetTimeEnable);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(TimeStampSec);
        fun(TimeStampNSec);
        fun(GetTimeEnable);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("TimeStampSec", TimeStampSec);
        fun("TimeStampNSec", TimeStampNSec);
        fun("GetTimeEnable", GetTimeEnable);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("TimeStampSec", TimeStampSec);
        fun("TimeStampNSec", TimeStampNSec);
        fun("GetTimeEnable", GetTimeEnable);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_wstHafGlobalTime& t) const
    {
        return (TimeStampSec == t.TimeStampSec) && (TimeStampNSec == t.TimeStampNSec) && (GetTimeEnable == t.GetTimeEnable);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_WSTHAFGLOBALTIME_H
