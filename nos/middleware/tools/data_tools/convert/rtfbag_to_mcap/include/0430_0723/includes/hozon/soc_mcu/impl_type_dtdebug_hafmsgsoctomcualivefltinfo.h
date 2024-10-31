/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFMSGSOCTOMCUALIVEFLTINFO_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFMSGSOCTOMCUALIVEFLTINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_HafmsgSocToMcuAliveFltInfo {
    ::UInt8 MsgAliveFlt_0x100;
    ::UInt8 MsgAliveFlt_0x112;
    ::UInt8 MsgAliveFlt_0x1A2;
    ::UInt8 MsgAliveFlt_0x0A2;
    ::UInt8 MsgAliveFlt_0x0E3;
    ::UInt8 MsgAliveFlt_0x0E5;
    ::UInt8 MsgAliveFlt_0x200;
    ::UInt8 MsgAliveFlt_0x201;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(MsgAliveFlt_0x100);
        fun(MsgAliveFlt_0x112);
        fun(MsgAliveFlt_0x1A2);
        fun(MsgAliveFlt_0x0A2);
        fun(MsgAliveFlt_0x0E3);
        fun(MsgAliveFlt_0x0E5);
        fun(MsgAliveFlt_0x200);
        fun(MsgAliveFlt_0x201);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(MsgAliveFlt_0x100);
        fun(MsgAliveFlt_0x112);
        fun(MsgAliveFlt_0x1A2);
        fun(MsgAliveFlt_0x0A2);
        fun(MsgAliveFlt_0x0E3);
        fun(MsgAliveFlt_0x0E5);
        fun(MsgAliveFlt_0x200);
        fun(MsgAliveFlt_0x201);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("MsgAliveFlt_0x100", MsgAliveFlt_0x100);
        fun("MsgAliveFlt_0x112", MsgAliveFlt_0x112);
        fun("MsgAliveFlt_0x1A2", MsgAliveFlt_0x1A2);
        fun("MsgAliveFlt_0x0A2", MsgAliveFlt_0x0A2);
        fun("MsgAliveFlt_0x0E3", MsgAliveFlt_0x0E3);
        fun("MsgAliveFlt_0x0E5", MsgAliveFlt_0x0E5);
        fun("MsgAliveFlt_0x200", MsgAliveFlt_0x200);
        fun("MsgAliveFlt_0x201", MsgAliveFlt_0x201);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("MsgAliveFlt_0x100", MsgAliveFlt_0x100);
        fun("MsgAliveFlt_0x112", MsgAliveFlt_0x112);
        fun("MsgAliveFlt_0x1A2", MsgAliveFlt_0x1A2);
        fun("MsgAliveFlt_0x0A2", MsgAliveFlt_0x0A2);
        fun("MsgAliveFlt_0x0E3", MsgAliveFlt_0x0E3);
        fun("MsgAliveFlt_0x0E5", MsgAliveFlt_0x0E5);
        fun("MsgAliveFlt_0x200", MsgAliveFlt_0x200);
        fun("MsgAliveFlt_0x201", MsgAliveFlt_0x201);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_HafmsgSocToMcuAliveFltInfo& t) const
    {
        return (MsgAliveFlt_0x100 == t.MsgAliveFlt_0x100) && (MsgAliveFlt_0x112 == t.MsgAliveFlt_0x112) && (MsgAliveFlt_0x1A2 == t.MsgAliveFlt_0x1A2) && (MsgAliveFlt_0x0A2 == t.MsgAliveFlt_0x0A2) && (MsgAliveFlt_0x0E3 == t.MsgAliveFlt_0x0E3) && (MsgAliveFlt_0x0E5 == t.MsgAliveFlt_0x0E5) && (MsgAliveFlt_0x200 == t.MsgAliveFlt_0x200) && (MsgAliveFlt_0x201 == t.MsgAliveFlt_0x201);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFMSGSOCTOMCUALIVEFLTINFO_H
