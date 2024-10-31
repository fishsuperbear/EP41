/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_BUSOFFSTATUS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_BUSOFFSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_INTF_BusoffStatus {
    ::UInt8 INTF_Busoff_FD3;
    ::UInt8 INTF_Busoff_FD6;
    ::UInt8 INTF_Busoff_FD8;
    ::UInt8 INTF_Busoff_CAN6;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(INTF_Busoff_FD3);
        fun(INTF_Busoff_FD6);
        fun(INTF_Busoff_FD8);
        fun(INTF_Busoff_CAN6);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(INTF_Busoff_FD3);
        fun(INTF_Busoff_FD6);
        fun(INTF_Busoff_FD8);
        fun(INTF_Busoff_CAN6);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("INTF_Busoff_FD3", INTF_Busoff_FD3);
        fun("INTF_Busoff_FD6", INTF_Busoff_FD6);
        fun("INTF_Busoff_FD8", INTF_Busoff_FD8);
        fun("INTF_Busoff_CAN6", INTF_Busoff_CAN6);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("INTF_Busoff_FD3", INTF_Busoff_FD3);
        fun("INTF_Busoff_FD6", INTF_Busoff_FD6);
        fun("INTF_Busoff_FD8", INTF_Busoff_FD8);
        fun("INTF_Busoff_CAN6", INTF_Busoff_CAN6);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_INTF_BusoffStatus& t) const
    {
        return (INTF_Busoff_FD3 == t.INTF_Busoff_FD3) && (INTF_Busoff_FD6 == t.INTF_Busoff_FD6) && (INTF_Busoff_FD8 == t.INTF_Busoff_FD8) && (INTF_Busoff_CAN6 == t.INTF_Busoff_CAN6);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_BUSOFFSTATUS_H
