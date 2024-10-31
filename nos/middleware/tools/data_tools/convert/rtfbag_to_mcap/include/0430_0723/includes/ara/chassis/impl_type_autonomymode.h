/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_AUTONOMYMODE_H
#define ARA_CHASSIS_IMPL_TYPE_AUTONOMYMODE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_boolean.h"

namespace ara {
namespace chassis {
struct AutonomyMode {
    ::UInt8 autonomyLevel;
    ::Boolean gearAutonomous;
    ::Boolean speedAutonomous;
    ::Boolean steeringAutonomous;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(autonomyLevel);
        fun(gearAutonomous);
        fun(speedAutonomous);
        fun(steeringAutonomous);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(autonomyLevel);
        fun(gearAutonomous);
        fun(speedAutonomous);
        fun(steeringAutonomous);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("autonomyLevel", autonomyLevel);
        fun("gearAutonomous", gearAutonomous);
        fun("speedAutonomous", speedAutonomous);
        fun("steeringAutonomous", steeringAutonomous);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("autonomyLevel", autonomyLevel);
        fun("gearAutonomous", gearAutonomous);
        fun("speedAutonomous", speedAutonomous);
        fun("steeringAutonomous", steeringAutonomous);
    }

    bool operator==(const ::ara::chassis::AutonomyMode& t) const
    {
        return (autonomyLevel == t.autonomyLevel) && (gearAutonomous == t.gearAutonomous) && (speedAutonomous == t.speedAutonomous) && (steeringAutonomous == t.steeringAutonomous);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_AUTONOMYMODE_H
