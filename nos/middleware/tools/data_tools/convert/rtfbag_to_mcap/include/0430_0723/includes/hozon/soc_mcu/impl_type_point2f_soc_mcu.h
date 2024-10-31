/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_POINT2F_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_POINT2F_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"

namespace hozon {
namespace soc_mcu {
struct Point2f_soc_mcu {
    ::Float x;
    ::Float y;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(x);
        fun(y);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
    }

    bool operator==(const ::hozon::soc_mcu::Point2f_soc_mcu& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_POINT2F_SOC_MCU_H
