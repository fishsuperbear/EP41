/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_VCUFR05INFO_H
#define ARA_VEHICLE_IMPL_TYPE_VCUFR05INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct VCUFr05Info {
    ::UInt8 vcu5_gear_box_position_display;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vcu5_gear_box_position_display);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vcu5_gear_box_position_display);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vcu5_gear_box_position_display", vcu5_gear_box_position_display);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vcu5_gear_box_position_display", vcu5_gear_box_position_display);
    }

    bool operator==(const ::ara::vehicle::VCUFr05Info& t) const
    {
        return (vcu5_gear_box_position_display == t.vcu5_gear_box_position_display);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_VCUFR05INFO_H
