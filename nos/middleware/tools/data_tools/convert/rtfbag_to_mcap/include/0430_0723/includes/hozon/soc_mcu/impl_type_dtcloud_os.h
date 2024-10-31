/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_OS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_OS_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtcloud_osee_ccb_var_core0.h"
#include "hozon/soc_mcu/impl_type_dtcloud_osee_ccb_var_core1.h"
#include "hozon/soc_mcu/impl_type_dtcloud_osee_ccb_var_core2.h"
#include "hozon/soc_mcu/impl_type_dtcloud_osee_ccb_var_core3.h"
#include "hozon/soc_mcu/impl_type_dtcloud_osee_ccb_var_core4.h"
#include "hozon/soc_mcu/impl_type_dtcloud_osee_ccb_var_core5.h"
#include "hozon/soc_mcu/impl_type_dtcloud_osee_tcb_12.h"
#include "hozon/soc_mcu/impl_type_dtcloud_osee_tcb_3.h"
#include "hozon/soc_mcu/impl_type_dtcloud_osee_tcb.h"
#include "hozon/soc_mcu/impl_type_dtcloud_vosm_cpu_6.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_OS {
    ::hozon::soc_mcu::DtCloud_osEE_ccb_var_core0 osEE_ccb_var_core0;
    ::hozon::soc_mcu::DtCloud_osEE_ccb_var_core1 osEE_ccb_var_core1;
    ::hozon::soc_mcu::DtCloud_osEE_ccb_var_core2 osEE_ccb_var_core2;
    ::hozon::soc_mcu::DtCloud_osEE_ccb_var_core3 osEE_ccb_var_core3;
    ::hozon::soc_mcu::DtCloud_osEE_ccb_var_core4 osEE_ccb_var_core4;
    ::hozon::soc_mcu::DtCloud_osEE_ccb_var_core5 osEE_ccb_var_core5;
    ::hozon::soc_mcu::DtCloud_OsEE_TCB_12 osEE_tcb_array_core0;
    ::hozon::soc_mcu::DtCloud_OsEE_TCB_3 osEE_tcb_array_core1;
    ::hozon::soc_mcu::DtCloud_OsEE_TCB osEE_tcb_array_core2;
    ::hozon::soc_mcu::DtCloud_OsEE_TCB_3 osEE_tcb_array_core3;
    ::hozon::soc_mcu::DtCloud_OsEE_TCB_3 osEE_tcb_array_core4;
    ::hozon::soc_mcu::DtCloud_OsEE_TCB osEE_tcb_array_core5;
    ::hozon::soc_mcu::DtCloud_Vosm_cpu_6 Vosm_cpu;
    ::UInt8 tin_flag;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(osEE_ccb_var_core0);
        fun(osEE_ccb_var_core1);
        fun(osEE_ccb_var_core2);
        fun(osEE_ccb_var_core3);
        fun(osEE_ccb_var_core4);
        fun(osEE_ccb_var_core5);
        fun(osEE_tcb_array_core0);
        fun(osEE_tcb_array_core1);
        fun(osEE_tcb_array_core2);
        fun(osEE_tcb_array_core3);
        fun(osEE_tcb_array_core4);
        fun(osEE_tcb_array_core5);
        fun(Vosm_cpu);
        fun(tin_flag);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(osEE_ccb_var_core0);
        fun(osEE_ccb_var_core1);
        fun(osEE_ccb_var_core2);
        fun(osEE_ccb_var_core3);
        fun(osEE_ccb_var_core4);
        fun(osEE_ccb_var_core5);
        fun(osEE_tcb_array_core0);
        fun(osEE_tcb_array_core1);
        fun(osEE_tcb_array_core2);
        fun(osEE_tcb_array_core3);
        fun(osEE_tcb_array_core4);
        fun(osEE_tcb_array_core5);
        fun(Vosm_cpu);
        fun(tin_flag);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("osEE_ccb_var_core0", osEE_ccb_var_core0);
        fun("osEE_ccb_var_core1", osEE_ccb_var_core1);
        fun("osEE_ccb_var_core2", osEE_ccb_var_core2);
        fun("osEE_ccb_var_core3", osEE_ccb_var_core3);
        fun("osEE_ccb_var_core4", osEE_ccb_var_core4);
        fun("osEE_ccb_var_core5", osEE_ccb_var_core5);
        fun("osEE_tcb_array_core0", osEE_tcb_array_core0);
        fun("osEE_tcb_array_core1", osEE_tcb_array_core1);
        fun("osEE_tcb_array_core2", osEE_tcb_array_core2);
        fun("osEE_tcb_array_core3", osEE_tcb_array_core3);
        fun("osEE_tcb_array_core4", osEE_tcb_array_core4);
        fun("osEE_tcb_array_core5", osEE_tcb_array_core5);
        fun("Vosm_cpu", Vosm_cpu);
        fun("tin_flag", tin_flag);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("osEE_ccb_var_core0", osEE_ccb_var_core0);
        fun("osEE_ccb_var_core1", osEE_ccb_var_core1);
        fun("osEE_ccb_var_core2", osEE_ccb_var_core2);
        fun("osEE_ccb_var_core3", osEE_ccb_var_core3);
        fun("osEE_ccb_var_core4", osEE_ccb_var_core4);
        fun("osEE_ccb_var_core5", osEE_ccb_var_core5);
        fun("osEE_tcb_array_core0", osEE_tcb_array_core0);
        fun("osEE_tcb_array_core1", osEE_tcb_array_core1);
        fun("osEE_tcb_array_core2", osEE_tcb_array_core2);
        fun("osEE_tcb_array_core3", osEE_tcb_array_core3);
        fun("osEE_tcb_array_core4", osEE_tcb_array_core4);
        fun("osEE_tcb_array_core5", osEE_tcb_array_core5);
        fun("Vosm_cpu", Vosm_cpu);
        fun("tin_flag", tin_flag);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_OS& t) const
    {
        return (osEE_ccb_var_core0 == t.osEE_ccb_var_core0) && (osEE_ccb_var_core1 == t.osEE_ccb_var_core1) && (osEE_ccb_var_core2 == t.osEE_ccb_var_core2) && (osEE_ccb_var_core3 == t.osEE_ccb_var_core3) && (osEE_ccb_var_core4 == t.osEE_ccb_var_core4) && (osEE_ccb_var_core5 == t.osEE_ccb_var_core5) && (osEE_tcb_array_core0 == t.osEE_tcb_array_core0) && (osEE_tcb_array_core1 == t.osEE_tcb_array_core1) && (osEE_tcb_array_core2 == t.osEE_tcb_array_core2) && (osEE_tcb_array_core3 == t.osEE_tcb_array_core3) && (osEE_tcb_array_core4 == t.osEE_tcb_array_core4) && (osEE_tcb_array_core5 == t.osEE_tcb_array_core5) && (Vosm_cpu == t.Vosm_cpu) && (tin_flag == t.tin_flag);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_OS_H
