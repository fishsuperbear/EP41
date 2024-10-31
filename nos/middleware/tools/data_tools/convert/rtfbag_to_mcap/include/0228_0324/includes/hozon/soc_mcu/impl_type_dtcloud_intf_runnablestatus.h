/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_RUNNABLESTATUS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_RUNNABLESTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_INTF_RunnableStatus {
    ::UInt32 INTF_Core0_MainFunction_Cnt;
    ::UInt32 INTF_Core0_MainFunction_FM_Cnt;
    ::UInt32 INTF_Core0_MainFunctionTx_Cnt;
    ::UInt32 INTF_Core1_MainFunctionRx_Cnt;
    ::UInt32 INTF_Core1_MainFunctionTx_Cnt;
    ::UInt32 INTF_Core3_MainFunctionRx_Cnt;
    ::UInt32 INTF_Core3_MainFunctionTx_Cnt;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(INTF_Core0_MainFunction_Cnt);
        fun(INTF_Core0_MainFunction_FM_Cnt);
        fun(INTF_Core0_MainFunctionTx_Cnt);
        fun(INTF_Core1_MainFunctionRx_Cnt);
        fun(INTF_Core1_MainFunctionTx_Cnt);
        fun(INTF_Core3_MainFunctionRx_Cnt);
        fun(INTF_Core3_MainFunctionTx_Cnt);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(INTF_Core0_MainFunction_Cnt);
        fun(INTF_Core0_MainFunction_FM_Cnt);
        fun(INTF_Core0_MainFunctionTx_Cnt);
        fun(INTF_Core1_MainFunctionRx_Cnt);
        fun(INTF_Core1_MainFunctionTx_Cnt);
        fun(INTF_Core3_MainFunctionRx_Cnt);
        fun(INTF_Core3_MainFunctionTx_Cnt);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("INTF_Core0_MainFunction_Cnt", INTF_Core0_MainFunction_Cnt);
        fun("INTF_Core0_MainFunction_FM_Cnt", INTF_Core0_MainFunction_FM_Cnt);
        fun("INTF_Core0_MainFunctionTx_Cnt", INTF_Core0_MainFunctionTx_Cnt);
        fun("INTF_Core1_MainFunctionRx_Cnt", INTF_Core1_MainFunctionRx_Cnt);
        fun("INTF_Core1_MainFunctionTx_Cnt", INTF_Core1_MainFunctionTx_Cnt);
        fun("INTF_Core3_MainFunctionRx_Cnt", INTF_Core3_MainFunctionRx_Cnt);
        fun("INTF_Core3_MainFunctionTx_Cnt", INTF_Core3_MainFunctionTx_Cnt);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("INTF_Core0_MainFunction_Cnt", INTF_Core0_MainFunction_Cnt);
        fun("INTF_Core0_MainFunction_FM_Cnt", INTF_Core0_MainFunction_FM_Cnt);
        fun("INTF_Core0_MainFunctionTx_Cnt", INTF_Core0_MainFunctionTx_Cnt);
        fun("INTF_Core1_MainFunctionRx_Cnt", INTF_Core1_MainFunctionRx_Cnt);
        fun("INTF_Core1_MainFunctionTx_Cnt", INTF_Core1_MainFunctionTx_Cnt);
        fun("INTF_Core3_MainFunctionRx_Cnt", INTF_Core3_MainFunctionRx_Cnt);
        fun("INTF_Core3_MainFunctionTx_Cnt", INTF_Core3_MainFunctionTx_Cnt);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_INTF_RunnableStatus& t) const
    {
        return (INTF_Core0_MainFunction_Cnt == t.INTF_Core0_MainFunction_Cnt) && (INTF_Core0_MainFunction_FM_Cnt == t.INTF_Core0_MainFunction_FM_Cnt) && (INTF_Core0_MainFunctionTx_Cnt == t.INTF_Core0_MainFunctionTx_Cnt) && (INTF_Core1_MainFunctionRx_Cnt == t.INTF_Core1_MainFunctionRx_Cnt) && (INTF_Core1_MainFunctionTx_Cnt == t.INTF_Core1_MainFunctionTx_Cnt) && (INTF_Core3_MainFunctionRx_Cnt == t.INTF_Core3_MainFunctionRx_Cnt) && (INTF_Core3_MainFunctionTx_Cnt == t.INTF_Core3_MainFunctionTx_Cnt);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_RUNNABLESTATUS_H
