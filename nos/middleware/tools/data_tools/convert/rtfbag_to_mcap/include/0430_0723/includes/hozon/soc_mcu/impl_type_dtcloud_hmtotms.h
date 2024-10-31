/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HMTOTMS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HMTOTMS_H
#include <cfloat>
#include <cmath>
#include "impl_type_int16.h"
#include "impl_type_uint8.h"
#include "impl_type_int8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HmToTms {
    ::Int16 EDUTmp;
    ::UInt8 EDUState;
    ::Int8 EnvironmentTmp;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(EDUTmp);
        fun(EDUState);
        fun(EnvironmentTmp);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(EDUTmp);
        fun(EDUState);
        fun(EnvironmentTmp);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("EDUTmp", EDUTmp);
        fun("EDUState", EDUState);
        fun("EnvironmentTmp", EnvironmentTmp);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("EDUTmp", EDUTmp);
        fun("EDUState", EDUState);
        fun("EnvironmentTmp", EnvironmentTmp);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_HmToTms& t) const
    {
        return (EDUTmp == t.EDUTmp) && (EDUState == t.EDUState) && (EnvironmentTmp == t.EnvironmentTmp);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HMTOTMS_H
