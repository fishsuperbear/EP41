/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_ADAS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_ADAS_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_uint8array_1024.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_ADAS {
    ::hozon::soc_mcu::uint8Array_1024 ADASCloudData;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADASCloudData);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADASCloudData);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADASCloudData", ADASCloudData);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADASCloudData", ADASCloudData);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_ADAS& t) const
    {
        return (ADASCloudData == t.ADASCloudData);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_ADAS_H
