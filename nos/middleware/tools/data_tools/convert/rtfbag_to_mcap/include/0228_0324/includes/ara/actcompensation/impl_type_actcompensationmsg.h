/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ACTCOMPENSATION_IMPL_TYPE_ACTCOMPENSATIONMSG_H
#define ARA_ACTCOMPENSATION_IMPL_TYPE_ACTCOMPENSATIONMSG_H
#include <cfloat>
#include <cmath>
#include "ara/actcompensation/impl_type_statusinfo.h"
#include "ara/actcompensation/impl_type_sensorinfo.h"
#include "ara/actcompensation/impl_type_gearinfo.h"
#include "ara/actcompensation/impl_type_brakeinfo.h"
#include "ara/actcompensation/impl_type_steerinfo.h"
#include "ara/actcompensation/impl_type_vehiclemotion.h"
#include "ara/actcompensation/impl_type_offsetinfo.h"
#include "ara/actcompensation/impl_type_actcommonheader.h"

namespace ara {
namespace actcompensation {
struct ActCompensationMsg {
    ::ara::actcompensation::StatusInfo status;
    ::ara::actcompensation::SensorInfo sensor;
    ::ara::actcompensation::GearInfo gear;
    ::ara::actcompensation::BrakeInfo brake;
    ::ara::actcompensation::SteerInfo steer;
    ::ara::actcompensation::VehicleMotion egoMotion;
    ::ara::actcompensation::OffsetInfo offset;
    ::ara::actcompensation::ActCommonHeader header;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(status);
        fun(sensor);
        fun(gear);
        fun(brake);
        fun(steer);
        fun(egoMotion);
        fun(offset);
        fun(header);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(status);
        fun(sensor);
        fun(gear);
        fun(brake);
        fun(steer);
        fun(egoMotion);
        fun(offset);
        fun(header);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("status", status);
        fun("sensor", sensor);
        fun("gear", gear);
        fun("brake", brake);
        fun("steer", steer);
        fun("egoMotion", egoMotion);
        fun("offset", offset);
        fun("header", header);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("status", status);
        fun("sensor", sensor);
        fun("gear", gear);
        fun("brake", brake);
        fun("steer", steer);
        fun("egoMotion", egoMotion);
        fun("offset", offset);
        fun("header", header);
    }

    bool operator==(const ::ara::actcompensation::ActCompensationMsg& t) const
    {
        return (status == t.status) && (sensor == t.sensor) && (gear == t.gear) && (brake == t.brake) && (steer == t.steer) && (egoMotion == t.egoMotion) && (offset == t.offset) && (header == t.header);
    }
};
} // namespace actcompensation
} // namespace ara


#endif // ARA_ACTCOMPENSATION_IMPL_TYPE_ACTCOMPENSATIONMSG_H
