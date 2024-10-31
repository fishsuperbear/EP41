/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_TMSTOHM_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_TMSTOHM_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_TmsToHm {
    ::Float EDUThroughput;
    ::UInt8 EDUFlag;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(EDUThroughput);
        fun(EDUFlag);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(EDUThroughput);
        fun(EDUFlag);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("EDUThroughput", EDUThroughput);
        fun("EDUFlag", EDUFlag);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("EDUThroughput", EDUThroughput);
        fun("EDUFlag", EDUFlag);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_TmsToHm& t) const
    {
        return (fabs(static_cast<double>(EDUThroughput - t.EDUThroughput)) < DBL_EPSILON) && (EDUFlag == t.EDUFlag);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_TMSTOHM_H
