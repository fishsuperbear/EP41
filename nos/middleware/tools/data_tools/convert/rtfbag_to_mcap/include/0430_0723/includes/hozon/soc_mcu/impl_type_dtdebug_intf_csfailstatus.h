/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_CSFAILSTATUS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_CSFAILSTATUS_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtdebug_servcallfail.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_INTF_CSFailStatus {
    ::hozon::soc_mcu::DtDebug_ServCallFail FM_INTF_E2E_ServCallFail;
    ::hozon::soc_mcu::DtDebug_ServCallFail FM_INTF_EcuLost_ServCallFail;
    ::hozon::soc_mcu::DtDebug_ServCallFail FM_INTF_BusOff_ServCallFail;
    ::hozon::soc_mcu::DtDebug_ServCallFail FM_INTF_EduMissing_ServCallFail;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(FM_INTF_E2E_ServCallFail);
        fun(FM_INTF_EcuLost_ServCallFail);
        fun(FM_INTF_BusOff_ServCallFail);
        fun(FM_INTF_EduMissing_ServCallFail);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(FM_INTF_E2E_ServCallFail);
        fun(FM_INTF_EcuLost_ServCallFail);
        fun(FM_INTF_BusOff_ServCallFail);
        fun(FM_INTF_EduMissing_ServCallFail);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("FM_INTF_E2E_ServCallFail", FM_INTF_E2E_ServCallFail);
        fun("FM_INTF_EcuLost_ServCallFail", FM_INTF_EcuLost_ServCallFail);
        fun("FM_INTF_BusOff_ServCallFail", FM_INTF_BusOff_ServCallFail);
        fun("FM_INTF_EduMissing_ServCallFail", FM_INTF_EduMissing_ServCallFail);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("FM_INTF_E2E_ServCallFail", FM_INTF_E2E_ServCallFail);
        fun("FM_INTF_EcuLost_ServCallFail", FM_INTF_EcuLost_ServCallFail);
        fun("FM_INTF_BusOff_ServCallFail", FM_INTF_BusOff_ServCallFail);
        fun("FM_INTF_EduMissing_ServCallFail", FM_INTF_EduMissing_ServCallFail);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_INTF_CSFailStatus& t) const
    {
        return (FM_INTF_E2E_ServCallFail == t.FM_INTF_E2E_ServCallFail) && (FM_INTF_EcuLost_ServCallFail == t.FM_INTF_EcuLost_ServCallFail) && (FM_INTF_BusOff_ServCallFail == t.FM_INTF_BusOff_ServCallFail) && (FM_INTF_EduMissing_ServCallFail == t.FM_INTF_EduMissing_ServCallFail);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_CSFAILSTATUS_H
