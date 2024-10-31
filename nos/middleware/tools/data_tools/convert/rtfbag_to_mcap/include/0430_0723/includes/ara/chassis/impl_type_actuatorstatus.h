/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_ACTUATORSTATUS_H
#define ARA_CHASSIS_IMPL_TYPE_ACTUATORSTATUS_H
#include <cfloat>
#include <cmath>
#include "ara/chassis/impl_type_actuatorworkstatus.h"

namespace ara {
namespace chassis {
struct ActuatorStatus {
    ::ara::chassis::ActuatorWorkStatus angleSteer;
    ::ara::chassis::ActuatorWorkStatus torqueSteer;
    ::ara::chassis::ActuatorWorkStatus emergencyAngleSteer;
    ::ara::chassis::ActuatorWorkStatus emergencyTorqueSteer;
    ::ara::chassis::ActuatorWorkStatus steerVibrate;
    ::ara::chassis::ActuatorWorkStatus accelerate;
    ::ara::chassis::ActuatorWorkStatus decelerate;
    ::ara::chassis::ActuatorWorkStatus emergencyDecelerate;
    ::ara::chassis::ActuatorWorkStatus drive;
    ::ara::chassis::ActuatorWorkStatus brake;
    ::ara::chassis::ActuatorWorkStatus vlc;
    ::ara::chassis::ActuatorWorkStatus emergencyStop;
    ::ara::chassis::ActuatorWorkStatus stop;
    ::ara::chassis::ActuatorWorkStatus park;
    ::ara::chassis::ActuatorWorkStatus gear;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(angleSteer);
        fun(torqueSteer);
        fun(emergencyAngleSteer);
        fun(emergencyTorqueSteer);
        fun(steerVibrate);
        fun(accelerate);
        fun(decelerate);
        fun(emergencyDecelerate);
        fun(drive);
        fun(brake);
        fun(vlc);
        fun(emergencyStop);
        fun(stop);
        fun(park);
        fun(gear);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(angleSteer);
        fun(torqueSteer);
        fun(emergencyAngleSteer);
        fun(emergencyTorqueSteer);
        fun(steerVibrate);
        fun(accelerate);
        fun(decelerate);
        fun(emergencyDecelerate);
        fun(drive);
        fun(brake);
        fun(vlc);
        fun(emergencyStop);
        fun(stop);
        fun(park);
        fun(gear);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("angleSteer", angleSteer);
        fun("torqueSteer", torqueSteer);
        fun("emergencyAngleSteer", emergencyAngleSteer);
        fun("emergencyTorqueSteer", emergencyTorqueSteer);
        fun("steerVibrate", steerVibrate);
        fun("accelerate", accelerate);
        fun("decelerate", decelerate);
        fun("emergencyDecelerate", emergencyDecelerate);
        fun("drive", drive);
        fun("brake", brake);
        fun("vlc", vlc);
        fun("emergencyStop", emergencyStop);
        fun("stop", stop);
        fun("park", park);
        fun("gear", gear);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("angleSteer", angleSteer);
        fun("torqueSteer", torqueSteer);
        fun("emergencyAngleSteer", emergencyAngleSteer);
        fun("emergencyTorqueSteer", emergencyTorqueSteer);
        fun("steerVibrate", steerVibrate);
        fun("accelerate", accelerate);
        fun("decelerate", decelerate);
        fun("emergencyDecelerate", emergencyDecelerate);
        fun("drive", drive);
        fun("brake", brake);
        fun("vlc", vlc);
        fun("emergencyStop", emergencyStop);
        fun("stop", stop);
        fun("park", park);
        fun("gear", gear);
    }

    bool operator==(const ::ara::chassis::ActuatorStatus& t) const
    {
        return (angleSteer == t.angleSteer) && (torqueSteer == t.torqueSteer) && (emergencyAngleSteer == t.emergencyAngleSteer) && (emergencyTorqueSteer == t.emergencyTorqueSteer) && (steerVibrate == t.steerVibrate) && (accelerate == t.accelerate) && (decelerate == t.decelerate) && (emergencyDecelerate == t.emergencyDecelerate) && (drive == t.drive) && (brake == t.brake) && (vlc == t.vlc) && (emergencyStop == t.emergencyStop) && (stop == t.stop) && (park == t.park) && (gear == t.gear);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_ACTUATORSTATUS_H
