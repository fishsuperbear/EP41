/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_VOSM_CPU_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_VOSM_CPU_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_Vosm_cpu {
    ::UInt8 cpu_usage;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(cpu_usage);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(cpu_usage);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("cpu_usage", cpu_usage);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("cpu_usage", cpu_usage);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_Vosm_cpu& t) const
    {
        return (cpu_usage == t.cpu_usage);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_VOSM_CPU_H
