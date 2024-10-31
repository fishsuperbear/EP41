/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_VCUINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_VCUINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace chassis {
struct VcuInfo {
    ::UInt8 VCU_ActGearPosition;
    ::Boolean VCU_ActGearPosition_Valid;
    ::UInt8 VCU_Real_ThrottlePosition;
    ::Boolean VCU_Real_ThrottlePos_Valid;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(VCU_ActGearPosition);
        fun(VCU_ActGearPosition_Valid);
        fun(VCU_Real_ThrottlePosition);
        fun(VCU_Real_ThrottlePos_Valid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(VCU_ActGearPosition);
        fun(VCU_ActGearPosition_Valid);
        fun(VCU_Real_ThrottlePosition);
        fun(VCU_Real_ThrottlePos_Valid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("VCU_ActGearPosition", VCU_ActGearPosition);
        fun("VCU_ActGearPosition_Valid", VCU_ActGearPosition_Valid);
        fun("VCU_Real_ThrottlePosition", VCU_Real_ThrottlePosition);
        fun("VCU_Real_ThrottlePos_Valid", VCU_Real_ThrottlePos_Valid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("VCU_ActGearPosition", VCU_ActGearPosition);
        fun("VCU_ActGearPosition_Valid", VCU_ActGearPosition_Valid);
        fun("VCU_Real_ThrottlePosition", VCU_Real_ThrottlePosition);
        fun("VCU_Real_ThrottlePos_Valid", VCU_Real_ThrottlePos_Valid);
    }

    bool operator==(const ::hozon::chassis::VcuInfo& t) const
    {
        return (VCU_ActGearPosition == t.VCU_ActGearPosition) && (VCU_ActGearPosition_Valid == t.VCU_ActGearPosition_Valid) && (VCU_Real_ThrottlePosition == t.VCU_Real_ThrottlePosition) && (VCU_Real_ThrottlePos_Valid == t.VCU_Real_ThrottlePos_Valid);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_VCUINFO_H
