/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ACTCOMPENSATION_IMPL_TYPE_BRAKEINFO_H
#define ARA_ACTCOMPENSATION_IMPL_TYPE_BRAKEINFO_H
#include <cfloat>
#include <cmath>
#include "ara/actcompensation/impl_type_wheelspeedinfo.h"
#include "ara/actcompensation/impl_type_float32withvalid.h"
#include "ara/actcompensation/impl_type_uint32withvalid.h"

namespace ara {
namespace actcompensation {
struct BrakeInfo {
    ::ara::actcompensation::WheelSpeedInfo wheelSpeedFl;
    ::ara::actcompensation::WheelSpeedInfo wheelSpeedFr;
    ::ara::actcompensation::WheelSpeedInfo wheelSpeedRl;
    ::ara::actcompensation::WheelSpeedInfo wheelSpeedRr;
    ::ara::actcompensation::Float32WithValid brakePedal;
    ::ara::actcompensation::Float32WithValid velocity;
    ::ara::actcompensation::Uint32WithValid masterMotorSpeed;
    ::ara::actcompensation::Uint32WithValid slaveMotorSpeed;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(wheelSpeedFl);
        fun(wheelSpeedFr);
        fun(wheelSpeedRl);
        fun(wheelSpeedRr);
        fun(brakePedal);
        fun(velocity);
        fun(masterMotorSpeed);
        fun(slaveMotorSpeed);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(wheelSpeedFl);
        fun(wheelSpeedFr);
        fun(wheelSpeedRl);
        fun(wheelSpeedRr);
        fun(brakePedal);
        fun(velocity);
        fun(masterMotorSpeed);
        fun(slaveMotorSpeed);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("wheelSpeedFl", wheelSpeedFl);
        fun("wheelSpeedFr", wheelSpeedFr);
        fun("wheelSpeedRl", wheelSpeedRl);
        fun("wheelSpeedRr", wheelSpeedRr);
        fun("brakePedal", brakePedal);
        fun("velocity", velocity);
        fun("masterMotorSpeed", masterMotorSpeed);
        fun("slaveMotorSpeed", slaveMotorSpeed);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("wheelSpeedFl", wheelSpeedFl);
        fun("wheelSpeedFr", wheelSpeedFr);
        fun("wheelSpeedRl", wheelSpeedRl);
        fun("wheelSpeedRr", wheelSpeedRr);
        fun("brakePedal", brakePedal);
        fun("velocity", velocity);
        fun("masterMotorSpeed", masterMotorSpeed);
        fun("slaveMotorSpeed", slaveMotorSpeed);
    }

    bool operator==(const ::ara::actcompensation::BrakeInfo& t) const
    {
        return (wheelSpeedFl == t.wheelSpeedFl) && (wheelSpeedFr == t.wheelSpeedFr) && (wheelSpeedRl == t.wheelSpeedRl) && (wheelSpeedRr == t.wheelSpeedRr) && (brakePedal == t.brakePedal) && (velocity == t.velocity) && (masterMotorSpeed == t.masterMotorSpeed) && (slaveMotorSpeed == t.slaveMotorSpeed);
    }
};
} // namespace actcompensation
} // namespace ara


#endif // ARA_ACTCOMPENSATION_IMPL_TYPE_BRAKEINFO_H
