/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_ACCELWITHCOVARIANCE_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_ACCELWITHCOVARIANCE_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "impl_type_accel.h"

namespace hozon {
namespace soc_mcu {
struct AccelWithCovariance_soc_mcu {
    ::Accel accel;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(accel);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(accel);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("accel", accel);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("accel", accel);
    }

    bool operator==(const ::hozon::soc_mcu::AccelWithCovariance_soc_mcu& t) const
    {
        return (accel == t.accel);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_ACCELWITHCOVARIANCE_SOC_MCU_H
