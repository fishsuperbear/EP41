/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWRMGRERRORSTATUS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWRMGRERRORSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_PwrMgrErrorstatus {
    ::UInt8 SocFaultFlag;
    ::UInt8 MDCFaultFlag;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SocFaultFlag);
        fun(MDCFaultFlag);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SocFaultFlag);
        fun(MDCFaultFlag);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("SocFaultFlag", SocFaultFlag);
        fun("MDCFaultFlag", MDCFaultFlag);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("SocFaultFlag", SocFaultFlag);
        fun("MDCFaultFlag", MDCFaultFlag);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_PwrMgrErrorstatus& t) const
    {
        return (SocFaultFlag == t.SocFaultFlag) && (MDCFaultFlag == t.MDCFaultFlag);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWRMGRERRORSTATUS_H
