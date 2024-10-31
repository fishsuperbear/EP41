/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CONTROL_IMPL_TYPE_CONTROLFRAME_H
#define HOZON_CONTROL_IMPL_TYPE_CONTROLFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_double.h"
#include "impl_type_boolean.h"
#include "impl_type_uint8.h"
#include "hozon/control/impl_type_vehiclesignal.h"
#include "hozon/planning/impl_type_engageadvice.h"

namespace hozon {
namespace control {
struct ControlFrame {
    ::hozon::common::CommonHeader header;
    ::Double throttle;
    ::Double brake;
    ::Double torque;
    ::Double steeringAngleRate;
    ::Double steeringAngle;
    ::Double steeringTorque;
    ::Boolean parkingBrake;
    ::Double speed;
    ::Double acceleration;
    ::Boolean resetModel;
    ::Boolean engineOnOff;
    ::UInt8 drivingMode;
    ::UInt8 gearPosition;
    ::hozon::control::VehicleSignal vehicleSignal;
    ::hozon::planning::EngageAdvice engageAdvice;
    ::Boolean isInSafeMode;
    ::UInt8 ctrlRequestMode;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(throttle);
        fun(brake);
        fun(torque);
        fun(steeringAngleRate);
        fun(steeringAngle);
        fun(steeringTorque);
        fun(parkingBrake);
        fun(speed);
        fun(acceleration);
        fun(resetModel);
        fun(engineOnOff);
        fun(drivingMode);
        fun(gearPosition);
        fun(vehicleSignal);
        fun(engageAdvice);
        fun(isInSafeMode);
        fun(ctrlRequestMode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(throttle);
        fun(brake);
        fun(torque);
        fun(steeringAngleRate);
        fun(steeringAngle);
        fun(steeringTorque);
        fun(parkingBrake);
        fun(speed);
        fun(acceleration);
        fun(resetModel);
        fun(engineOnOff);
        fun(drivingMode);
        fun(gearPosition);
        fun(vehicleSignal);
        fun(engageAdvice);
        fun(isInSafeMode);
        fun(ctrlRequestMode);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("throttle", throttle);
        fun("brake", brake);
        fun("torque", torque);
        fun("steeringAngleRate", steeringAngleRate);
        fun("steeringAngle", steeringAngle);
        fun("steeringTorque", steeringTorque);
        fun("parkingBrake", parkingBrake);
        fun("speed", speed);
        fun("acceleration", acceleration);
        fun("resetModel", resetModel);
        fun("engineOnOff", engineOnOff);
        fun("drivingMode", drivingMode);
        fun("gearPosition", gearPosition);
        fun("vehicleSignal", vehicleSignal);
        fun("engageAdvice", engageAdvice);
        fun("isInSafeMode", isInSafeMode);
        fun("ctrlRequestMode", ctrlRequestMode);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("throttle", throttle);
        fun("brake", brake);
        fun("torque", torque);
        fun("steeringAngleRate", steeringAngleRate);
        fun("steeringAngle", steeringAngle);
        fun("steeringTorque", steeringTorque);
        fun("parkingBrake", parkingBrake);
        fun("speed", speed);
        fun("acceleration", acceleration);
        fun("resetModel", resetModel);
        fun("engineOnOff", engineOnOff);
        fun("drivingMode", drivingMode);
        fun("gearPosition", gearPosition);
        fun("vehicleSignal", vehicleSignal);
        fun("engageAdvice", engageAdvice);
        fun("isInSafeMode", isInSafeMode);
        fun("ctrlRequestMode", ctrlRequestMode);
    }

    bool operator==(const ::hozon::control::ControlFrame& t) const
    {
        return (header == t.header) && (fabs(static_cast<double>(throttle - t.throttle)) < DBL_EPSILON) && (fabs(static_cast<double>(brake - t.brake)) < DBL_EPSILON) && (fabs(static_cast<double>(torque - t.torque)) < DBL_EPSILON) && (fabs(static_cast<double>(steeringAngleRate - t.steeringAngleRate)) < DBL_EPSILON) && (fabs(static_cast<double>(steeringAngle - t.steeringAngle)) < DBL_EPSILON) && (fabs(static_cast<double>(steeringTorque - t.steeringTorque)) < DBL_EPSILON) && (parkingBrake == t.parkingBrake) && (fabs(static_cast<double>(speed - t.speed)) < DBL_EPSILON) && (fabs(static_cast<double>(acceleration - t.acceleration)) < DBL_EPSILON) && (resetModel == t.resetModel) && (engineOnOff == t.engineOnOff) && (drivingMode == t.drivingMode) && (gearPosition == t.gearPosition) && (vehicleSignal == t.vehicleSignal) && (engageAdvice == t.engageAdvice) && (isInSafeMode == t.isInSafeMode) && (ctrlRequestMode == t.ctrlRequestMode);
    }
};
} // namespace control
} // namespace hozon


#endif // HOZON_CONTROL_IMPL_TYPE_CONTROLFRAME_H
