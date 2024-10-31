/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_COUNTERSEQ_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_COUNTERSEQ_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_counterseq {
    ::UInt32 counterseq;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(counterseq);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(counterseq);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("counterseq", counterseq);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("counterseq", counterseq);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_counterseq& t) const
    {
        return (counterseq == t.counterseq);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_COUNTERSEQ_H
