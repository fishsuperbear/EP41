/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_HAFGLOBALTIME_H
#define HOZON_SOC2MCU_IMPL_TYPE_HAFGLOBALTIME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace soc2mcu {
struct HafGlobalTime {
    ::UInt32 TimeStampSec;
    ::UInt32 TimeStampNsec;
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
        fun(TimeStampNsec);
        fun(GetTimeEnable);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(TimeStampSec);
        fun(TimeStampNsec);
        fun(GetTimeEnable);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("TimeStampSec", TimeStampSec);
        fun("TimeStampNsec", TimeStampNsec);
        fun("GetTimeEnable", GetTimeEnable);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("TimeStampSec", TimeStampSec);
        fun("TimeStampNsec", TimeStampNsec);
        fun("GetTimeEnable", GetTimeEnable);
    }

    bool operator==(const ::hozon::soc2mcu::HafGlobalTime& t) const
    {
        return (TimeStampSec == t.TimeStampSec) && (TimeStampNsec == t.TimeStampNsec) && (GetTimeEnable == t.GetTimeEnable);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_HAFGLOBALTIME_H
