/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MCUDEBUGDATATYPE_H
#define HOZON_SOC_MCU_IMPL_TYPE_MCUDEBUGDATATYPE_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtdebug_intf_fm.h"
#include "hozon/soc_mcu/impl_type_dtdebug_fm.h"
#include "hozon/soc_mcu/impl_type_dtdebug_eth.h"
#include "hozon/soc_mcu/impl_type_dtdebug_servcallfail.h"

namespace hozon {
namespace soc_mcu {
struct MCUDebugDataType {
    ::hozon::soc_mcu::DtDebug_INTF_FM DebugData_INTF;
    ::hozon::soc_mcu::DtDebug_FM DebugData_FM;
    ::hozon::soc_mcu::DtDebug_ETH DebugData_ETH;
    ::hozon::soc_mcu::DtDebug_ServCallFail DebugData_WDGMC0;
    ::hozon::soc_mcu::DtDebug_ServCallFail DebugData_WDGMC1;
    ::hozon::soc_mcu::DtDebug_ServCallFail DebugData_WDGMC2;
    ::hozon::soc_mcu::DtDebug_ServCallFail DebugData_WDGMC3;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(DebugData_INTF);
        fun(DebugData_FM);
        fun(DebugData_ETH);
        fun(DebugData_WDGMC0);
        fun(DebugData_WDGMC1);
        fun(DebugData_WDGMC2);
        fun(DebugData_WDGMC3);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(DebugData_INTF);
        fun(DebugData_FM);
        fun(DebugData_ETH);
        fun(DebugData_WDGMC0);
        fun(DebugData_WDGMC1);
        fun(DebugData_WDGMC2);
        fun(DebugData_WDGMC3);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("DebugData_INTF", DebugData_INTF);
        fun("DebugData_FM", DebugData_FM);
        fun("DebugData_ETH", DebugData_ETH);
        fun("DebugData_WDGMC0", DebugData_WDGMC0);
        fun("DebugData_WDGMC1", DebugData_WDGMC1);
        fun("DebugData_WDGMC2", DebugData_WDGMC2);
        fun("DebugData_WDGMC3", DebugData_WDGMC3);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("DebugData_INTF", DebugData_INTF);
        fun("DebugData_FM", DebugData_FM);
        fun("DebugData_ETH", DebugData_ETH);
        fun("DebugData_WDGMC0", DebugData_WDGMC0);
        fun("DebugData_WDGMC1", DebugData_WDGMC1);
        fun("DebugData_WDGMC2", DebugData_WDGMC2);
        fun("DebugData_WDGMC3", DebugData_WDGMC3);
    }

    bool operator==(const ::hozon::soc_mcu::MCUDebugDataType& t) const
    {
        return (DebugData_INTF == t.DebugData_INTF) && (DebugData_FM == t.DebugData_FM) && (DebugData_ETH == t.DebugData_ETH) && (DebugData_WDGMC0 == t.DebugData_WDGMC0) && (DebugData_WDGMC1 == t.DebugData_WDGMC1) && (DebugData_WDGMC2 == t.DebugData_WDGMC2) && (DebugData_WDGMC3 == t.DebugData_WDGMC3);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MCUDEBUGDATATYPE_H
