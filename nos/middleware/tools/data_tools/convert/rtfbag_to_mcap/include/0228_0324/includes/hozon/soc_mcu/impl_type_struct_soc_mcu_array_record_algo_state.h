/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_STRUCT_SOC_MCU_ARRAY_RECORD_ALGO_STATE_H
#define HOZON_SOC_MCU_IMPL_TYPE_STRUCT_SOC_MCU_ARRAY_RECORD_ALGO_STATE_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_uint8array_7000.h"

namespace hozon {
namespace soc_mcu {
struct struct_soc_mcu_array_record_algo_state {
    ::hozon::soc_mcu::uint8Array_7000 soc_mcu_array;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(soc_mcu_array);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(soc_mcu_array);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("soc_mcu_array", soc_mcu_array);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("soc_mcu_array", soc_mcu_array);
    }

    bool operator==(const ::hozon::soc_mcu::struct_soc_mcu_array_record_algo_state& t) const
    {
        return (soc_mcu_array == t.soc_mcu_array);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_STRUCT_SOC_MCU_ARRAY_RECORD_ALGO_STATE_H