/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_OSEE_CCB_VAR_CORE0_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_OSEE_CCB_VAR_CORE0_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/soc_mcu/impl_type_dtcloud_p_curr.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_osEE_ccb_var_core0 {
    ::UInt32 os_status;
    ::hozon::soc_mcu::DtCloud_p_curr p_curr;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(os_status);
        fun(p_curr);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(os_status);
        fun(p_curr);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("os_status", os_status);
        fun("p_curr", p_curr);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("os_status", os_status);
        fun("p_curr", p_curr);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_osEE_ccb_var_core0& t) const
    {
        return (os_status == t.os_status) && (p_curr == t.p_curr);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_OSEE_CCB_VAR_CORE0_H
