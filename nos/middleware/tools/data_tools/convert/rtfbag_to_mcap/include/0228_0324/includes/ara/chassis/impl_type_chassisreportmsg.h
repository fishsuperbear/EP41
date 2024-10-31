/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_CHASSISREPORTMSG_H
#define ARA_CHASSIS_IMPL_TYPE_CHASSISREPORTMSG_H
#include <cfloat>
#include <cmath>
#include "ara/chassis/impl_type_autonomymode.h"
#include "impl_type_float.h"
#include "ara/chassis/impl_type_gear.h"
#include "ara/chassis/impl_type_actuatorstatus.h"
#include "ara/chassis/impl_type_steerinfo.h"
#include "ara/chassis/impl_type_brakeinfo.h"
#include "ara/chassis/impl_type_throttleinfo.h"
#include "ara/chassis/impl_type_gearinfo.h"
#include "ara/chassis/impl_type_vehiclemotion.h"
#include "ara/chassis/impl_type_chassisreportheader.h"

namespace ara {
namespace chassis {
struct ChassisReportMsg {
    ::ara::chassis::AutonomyMode autoMode;
    ::Float velocity;
    ::Float steerAngle;
    ::Float frontWheelAngle;
    ::ara::chassis::Gear actualGear;
    ::ara::chassis::ActuatorStatus master;
    ::ara::chassis::ActuatorStatus slave;
    ::ara::chassis::SteerInfo steer;
    ::ara::chassis::BrakeInfo brake;
    ::ara::chassis::ThrottleInfo vcu;
    ::ara::chassis::GearInfo gear;
    ::ara::chassis::VehicleMotion egoMotion;
    ::ara::chassis::ChassisReportHeader header;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(autoMode);
        fun(velocity);
        fun(steerAngle);
        fun(frontWheelAngle);
        fun(actualGear);
        fun(master);
        fun(slave);
        fun(steer);
        fun(brake);
        fun(vcu);
        fun(gear);
        fun(egoMotion);
        fun(header);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(autoMode);
        fun(velocity);
        fun(steerAngle);
        fun(frontWheelAngle);
        fun(actualGear);
        fun(master);
        fun(slave);
        fun(steer);
        fun(brake);
        fun(vcu);
        fun(gear);
        fun(egoMotion);
        fun(header);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("autoMode", autoMode);
        fun("velocity", velocity);
        fun("steerAngle", steerAngle);
        fun("frontWheelAngle", frontWheelAngle);
        fun("actualGear", actualGear);
        fun("master", master);
        fun("slave", slave);
        fun("steer", steer);
        fun("brake", brake);
        fun("vcu", vcu);
        fun("gear", gear);
        fun("egoMotion", egoMotion);
        fun("header", header);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("autoMode", autoMode);
        fun("velocity", velocity);
        fun("steerAngle", steerAngle);
        fun("frontWheelAngle", frontWheelAngle);
        fun("actualGear", actualGear);
        fun("master", master);
        fun("slave", slave);
        fun("steer", steer);
        fun("brake", brake);
        fun("vcu", vcu);
        fun("gear", gear);
        fun("egoMotion", egoMotion);
        fun("header", header);
    }

    bool operator==(const ::ara::chassis::ChassisReportMsg& t) const
    {
        return (autoMode == t.autoMode) && (fabs(static_cast<double>(velocity - t.velocity)) < DBL_EPSILON) && (fabs(static_cast<double>(steerAngle - t.steerAngle)) < DBL_EPSILON) && (fabs(static_cast<double>(frontWheelAngle - t.frontWheelAngle)) < DBL_EPSILON) && (actualGear == t.actualGear) && (master == t.master) && (slave == t.slave) && (steer == t.steer) && (brake == t.brake) && (vcu == t.vcu) && (gear == t.gear) && (egoMotion == t.egoMotion) && (header == t.header);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_CHASSISREPORTMSG_H
