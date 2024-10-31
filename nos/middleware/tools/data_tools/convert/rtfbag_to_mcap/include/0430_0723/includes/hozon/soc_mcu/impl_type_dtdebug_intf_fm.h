/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_FM_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_FM_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtdebug_intf_eculoststatus.h"
#include "hozon/soc_mcu/impl_type_dtdebug_intf_busoffstatus.h"
#include "hozon/soc_mcu/impl_type_dtdebug_intf_e2estatus.h"
#include "hozon/soc_mcu/impl_type_dtdebug_intf_csfailstatus.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_INTF_FM {
    ::hozon::soc_mcu::DtDebug_INTF_EcuLostStatus INTF_EcuLostStatus;
    ::hozon::soc_mcu::DtDebug_INTF_BusoffStatus INTF_BusoffStatus;
    ::hozon::soc_mcu::DtDebug_INTF_E2EStatus INTF_E2EStatus;
    ::hozon::soc_mcu::DtDebug_INTF_CSFailStatus INTF_CSFailStatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(INTF_EcuLostStatus);
        fun(INTF_BusoffStatus);
        fun(INTF_E2EStatus);
        fun(INTF_CSFailStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(INTF_EcuLostStatus);
        fun(INTF_BusoffStatus);
        fun(INTF_E2EStatus);
        fun(INTF_CSFailStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("INTF_EcuLostStatus", INTF_EcuLostStatus);
        fun("INTF_BusoffStatus", INTF_BusoffStatus);
        fun("INTF_E2EStatus", INTF_E2EStatus);
        fun("INTF_CSFailStatus", INTF_CSFailStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("INTF_EcuLostStatus", INTF_EcuLostStatus);
        fun("INTF_BusoffStatus", INTF_BusoffStatus);
        fun("INTF_E2EStatus", INTF_E2EStatus);
        fun("INTF_CSFailStatus", INTF_CSFailStatus);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_INTF_FM& t) const
    {
        return (INTF_EcuLostStatus == t.INTF_EcuLostStatus) && (INTF_BusoffStatus == t.INTF_BusoffStatus) && (INTF_E2EStatus == t.INTF_E2EStatus) && (INTF_CSFailStatus == t.INTF_CSFailStatus);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_FM_H
