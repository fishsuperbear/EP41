/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPALGOINTRFAILED_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPALGOINTRFAILED_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HafLidarPercpAlgoIntrFailed {
    ::UInt8 algo_intr_configload_error;
    ::UInt8 algo_intr_module_error;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(algo_intr_configload_error);
        fun(algo_intr_module_error);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(algo_intr_configload_error);
        fun(algo_intr_module_error);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("algo_intr_configload_error", algo_intr_configload_error);
        fun("algo_intr_module_error", algo_intr_module_error);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("algo_intr_configload_error", algo_intr_configload_error);
        fun("algo_intr_module_error", algo_intr_module_error);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_HafLidarPercpAlgoIntrFailed& t) const
    {
        return (algo_intr_configload_error == t.algo_intr_configload_error) && (algo_intr_module_error == t.algo_intr_module_error);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPALGOINTRFAILED_H
