/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_BDCSFR01INFO_H
#define HOZON_CHASSIS_IMPL_TYPE_BDCSFR01INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace chassis {
struct BdcsFr01Info {
    ::uint8_t PowerManageMode;
    ::uint8_t PowerMode;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(PowerManageMode);
        fun(PowerMode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(PowerManageMode);
        fun(PowerMode);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("PowerManageMode", PowerManageMode);
        fun("PowerMode", PowerMode);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("PowerManageMode", PowerManageMode);
        fun("PowerMode", PowerMode);
    }

    bool operator==(const ::hozon::chassis::BdcsFr01Info& t) const
    {
        return (PowerManageMode == t.PowerManageMode) && (PowerMode == t.PowerMode);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_BDCSFR01INFO_H
