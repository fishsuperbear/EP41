/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_VCUFR06INFO_H
#define ARA_VEHICLE_IMPL_TYPE_VCUFR06INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct VCUFr06Info {
    ::UInt8 vcu6_gas_pedal_position;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vcu6_gas_pedal_position);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vcu6_gas_pedal_position);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vcu6_gas_pedal_position", vcu6_gas_pedal_position);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vcu6_gas_pedal_position", vcu6_gas_pedal_position);
    }

    bool operator==(const ::ara::vehicle::VCUFr06Info& t) const
    {
        return (vcu6_gas_pedal_position == t.vcu6_gas_pedal_position);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_VCUFR06INFO_H
