/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_SIGNALSENDSTATUS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_SIGNALSENDSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_INTF_SignalSendStatus {
    ::UInt16 INTF_Send_0x3D8_Cnt;
    ::UInt16 INTF_Send_0x136_Cnt;
    ::UInt16 INTF_Send_0x265_Cnt;
    ::UInt16 INTF_Send_0x8E_Cnt;
    ::UInt16 INTF_Send_0xFE_Cnt;
    ::UInt16 INTF_Send_0x190_Cnt;
    ::UInt16 INTF_Send_0x191_Cnt;
    ::UInt16 INTF_Send_0x192_Cnt;
    ::UInt16 INTF_Send_0x193_Cnt;
    ::UInt16 INTF_Send_0x210_Cnt;
    ::UInt16 INTF_Send_0x194_Cnt;
    ::UInt16 INTF_Send_0x8F_Cnt;
    ::UInt16 INTF_Send_0x255_Cnt;
    ::UInt16 INTF_Send_0x301_Cnt;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(INTF_Send_0x3D8_Cnt);
        fun(INTF_Send_0x136_Cnt);
        fun(INTF_Send_0x265_Cnt);
        fun(INTF_Send_0x8E_Cnt);
        fun(INTF_Send_0xFE_Cnt);
        fun(INTF_Send_0x190_Cnt);
        fun(INTF_Send_0x191_Cnt);
        fun(INTF_Send_0x192_Cnt);
        fun(INTF_Send_0x193_Cnt);
        fun(INTF_Send_0x210_Cnt);
        fun(INTF_Send_0x194_Cnt);
        fun(INTF_Send_0x8F_Cnt);
        fun(INTF_Send_0x255_Cnt);
        fun(INTF_Send_0x301_Cnt);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(INTF_Send_0x3D8_Cnt);
        fun(INTF_Send_0x136_Cnt);
        fun(INTF_Send_0x265_Cnt);
        fun(INTF_Send_0x8E_Cnt);
        fun(INTF_Send_0xFE_Cnt);
        fun(INTF_Send_0x190_Cnt);
        fun(INTF_Send_0x191_Cnt);
        fun(INTF_Send_0x192_Cnt);
        fun(INTF_Send_0x193_Cnt);
        fun(INTF_Send_0x210_Cnt);
        fun(INTF_Send_0x194_Cnt);
        fun(INTF_Send_0x8F_Cnt);
        fun(INTF_Send_0x255_Cnt);
        fun(INTF_Send_0x301_Cnt);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("INTF_Send_0x3D8_Cnt", INTF_Send_0x3D8_Cnt);
        fun("INTF_Send_0x136_Cnt", INTF_Send_0x136_Cnt);
        fun("INTF_Send_0x265_Cnt", INTF_Send_0x265_Cnt);
        fun("INTF_Send_0x8E_Cnt", INTF_Send_0x8E_Cnt);
        fun("INTF_Send_0xFE_Cnt", INTF_Send_0xFE_Cnt);
        fun("INTF_Send_0x190_Cnt", INTF_Send_0x190_Cnt);
        fun("INTF_Send_0x191_Cnt", INTF_Send_0x191_Cnt);
        fun("INTF_Send_0x192_Cnt", INTF_Send_0x192_Cnt);
        fun("INTF_Send_0x193_Cnt", INTF_Send_0x193_Cnt);
        fun("INTF_Send_0x210_Cnt", INTF_Send_0x210_Cnt);
        fun("INTF_Send_0x194_Cnt", INTF_Send_0x194_Cnt);
        fun("INTF_Send_0x8F_Cnt", INTF_Send_0x8F_Cnt);
        fun("INTF_Send_0x255_Cnt", INTF_Send_0x255_Cnt);
        fun("INTF_Send_0x301_Cnt", INTF_Send_0x301_Cnt);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("INTF_Send_0x3D8_Cnt", INTF_Send_0x3D8_Cnt);
        fun("INTF_Send_0x136_Cnt", INTF_Send_0x136_Cnt);
        fun("INTF_Send_0x265_Cnt", INTF_Send_0x265_Cnt);
        fun("INTF_Send_0x8E_Cnt", INTF_Send_0x8E_Cnt);
        fun("INTF_Send_0xFE_Cnt", INTF_Send_0xFE_Cnt);
        fun("INTF_Send_0x190_Cnt", INTF_Send_0x190_Cnt);
        fun("INTF_Send_0x191_Cnt", INTF_Send_0x191_Cnt);
        fun("INTF_Send_0x192_Cnt", INTF_Send_0x192_Cnt);
        fun("INTF_Send_0x193_Cnt", INTF_Send_0x193_Cnt);
        fun("INTF_Send_0x210_Cnt", INTF_Send_0x210_Cnt);
        fun("INTF_Send_0x194_Cnt", INTF_Send_0x194_Cnt);
        fun("INTF_Send_0x8F_Cnt", INTF_Send_0x8F_Cnt);
        fun("INTF_Send_0x255_Cnt", INTF_Send_0x255_Cnt);
        fun("INTF_Send_0x301_Cnt", INTF_Send_0x301_Cnt);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_INTF_SignalSendStatus& t) const
    {
        return (INTF_Send_0x3D8_Cnt == t.INTF_Send_0x3D8_Cnt) && (INTF_Send_0x136_Cnt == t.INTF_Send_0x136_Cnt) && (INTF_Send_0x265_Cnt == t.INTF_Send_0x265_Cnt) && (INTF_Send_0x8E_Cnt == t.INTF_Send_0x8E_Cnt) && (INTF_Send_0xFE_Cnt == t.INTF_Send_0xFE_Cnt) && (INTF_Send_0x190_Cnt == t.INTF_Send_0x190_Cnt) && (INTF_Send_0x191_Cnt == t.INTF_Send_0x191_Cnt) && (INTF_Send_0x192_Cnt == t.INTF_Send_0x192_Cnt) && (INTF_Send_0x193_Cnt == t.INTF_Send_0x193_Cnt) && (INTF_Send_0x210_Cnt == t.INTF_Send_0x210_Cnt) && (INTF_Send_0x194_Cnt == t.INTF_Send_0x194_Cnt) && (INTF_Send_0x8F_Cnt == t.INTF_Send_0x8F_Cnt) && (INTF_Send_0x255_Cnt == t.INTF_Send_0x255_Cnt) && (INTF_Send_0x301_Cnt == t.INTF_Send_0x301_Cnt);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_SIGNALSENDSTATUS_H
