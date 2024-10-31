/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_TWISTWITHCOVARIANCE_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_TWISTWITHCOVARIANCE_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "impl_type_twist.h"

namespace hozon {
namespace soc_mcu {
struct TwistWithCovariance_soc_mcu {
    ::Twist twist;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(twist);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(twist);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("twist", twist);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("twist", twist);
    }

    bool operator==(const ::hozon::soc_mcu::TwistWithCovariance_soc_mcu& t) const
    {
        return (twist == t.twist);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_TWISTWITHCOVARIANCE_SOC_MCU_H
