/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_GEARINFO_H
#define ARA_CHASSIS_IMPL_TYPE_GEARINFO_H
#include <cfloat>
#include <cmath>
#include "ara/chassis/impl_type_gear.h"
#include "impl_type_uint8.h"
#include "ara/chassis/impl_type_uint8withvalid.h"
#include "impl_type_int32.h"

namespace ara {
namespace chassis {
struct GearInfo {
    ::ara::chassis::Gear gear;
    ::ara::chassis::Gear gearLever;
    ::UInt8 gearShiftStatus;
    ::ara::chassis::Uint8WithValid driverOverride;
    ::Int32 faultCode;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(gear);
        fun(gearLever);
        fun(gearShiftStatus);
        fun(driverOverride);
        fun(faultCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(gear);
        fun(gearLever);
        fun(gearShiftStatus);
        fun(driverOverride);
        fun(faultCode);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("gear", gear);
        fun("gearLever", gearLever);
        fun("gearShiftStatus", gearShiftStatus);
        fun("driverOverride", driverOverride);
        fun("faultCode", faultCode);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("gear", gear);
        fun("gearLever", gearLever);
        fun("gearShiftStatus", gearShiftStatus);
        fun("driverOverride", driverOverride);
        fun("faultCode", faultCode);
    }

    bool operator==(const ::ara::chassis::GearInfo& t) const
    {
        return (gear == t.gear) && (gearLever == t.gearLever) && (gearShiftStatus == t.gearShiftStatus) && (driverOverride == t.driverOverride) && (faultCode == t.faultCode);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_GEARINFO_H
