/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_ADAS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_ADAS_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_uint8array_7000.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_ADAS {
    ::hozon::soc_mcu::uint8Array_7000 ADAStateData;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADAStateData);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADAStateData);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADAStateData", ADAStateData);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADAStateData", ADAStateData);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_ADAS& t) const
    {
        return (ADAStateData == t.ADAStateData);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_ADAS_H
