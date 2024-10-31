/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_GEARCMD_H
#define ARA_CHASSIS_IMPL_TYPE_GEARCMD_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace chassis {
struct GearCmd {
    ::UInt8 value;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(value);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(value);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("value", value);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("value", value);
    }

    bool operator==(const ::ara::chassis::GearCmd& t) const
    {
        return (value == t.value);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_GEARCMD_H
