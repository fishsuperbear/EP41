/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_ENGAGEADVICE_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_ENGAGEADVICE_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "hozon/planning/impl_type_advise.h"
#include "hozon/soc_mcu/impl_type_uint8array_20.h"

namespace hozon {
namespace soc_mcu {
struct EngageAdvice_soc_mcu {
    ::hozon::planning::Advise advise;
    ::hozon::soc_mcu::uint8Array_20 reason;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(advise);
        fun(reason);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(advise);
        fun(reason);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("advise", advise);
        fun("reason", reason);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("advise", advise);
        fun("reason", reason);
    }

    bool operator==(const ::hozon::soc_mcu::EngageAdvice_soc_mcu& t) const
    {
        return (advise == t.advise) && (reason == t.reason);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_ENGAGEADVICE_SOC_MCU_H
