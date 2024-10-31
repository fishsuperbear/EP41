/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ACTCOMPENSATION_IMPL_TYPE_GEARINFO_H
#define ARA_ACTCOMPENSATION_IMPL_TYPE_GEARINFO_H
#include <cfloat>
#include <cmath>
#include "ara/actcompensation/impl_type_uint8withvalid.h"

namespace ara {
namespace actcompensation {
struct GearInfo {
    ::ara::actcompensation::Uint8WithValid gear;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(gear);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(gear);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("gear", gear);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("gear", gear);
    }

    bool operator==(const ::ara::actcompensation::GearInfo& t) const
    {
        return (gear == t.gear);
    }
};
} // namespace actcompensation
} // namespace ara


#endif // ARA_ACTCOMPENSATION_IMPL_TYPE_GEARINFO_H
