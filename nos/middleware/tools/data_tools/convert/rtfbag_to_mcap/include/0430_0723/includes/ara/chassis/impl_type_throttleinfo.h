/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_THROTTLEINFO_H
#define ARA_CHASSIS_IMPL_TYPE_THROTTLEINFO_H
#include <cfloat>
#include <cmath>
#include "ara/chassis/impl_type_float32withvalid.h"
#include "ara/chassis/impl_type_uint8withvalid.h"
#include "impl_type_uint8.h"
#include "impl_type_int32.h"

namespace ara {
namespace chassis {
struct ThrottleInfo {
    ::ara::chassis::Float32WithValid throttlePedal;
    ::ara::chassis::Float32WithValid throttlePedalRate;
    ::ara::chassis::Uint8WithValid driverOverride;
    ::ara::chassis::Float32WithValid driveTorque;
    ::ara::chassis::Float32WithValid driveTorqueMax;
    ::ara::chassis::Float32WithValid driveTorqueMin;
    ::ara::chassis::Float32WithValid driverDesiredTorque;
    ::ara::chassis::Float32WithValid engineSpeed;
    ::ara::chassis::Float32WithValid motorSpeed;
    ::ara::chassis::Uint8WithValid powerTrainReady;
    ::UInt8 driveTorqueCtrlAvl;
    ::UInt8 driveTorqueActv;
    ::UInt8 vcuAbortFbk;
    ::ara::chassis::Float32WithValid fuelRange;
    ::ara::chassis::Float32WithValid socHighVoltBattery;
    ::ara::chassis::Float32WithValid sohHighVoltBattery;
    ::ara::chassis::Float32WithValid socLowVoltBattery;
    ::ara::chassis::Float32WithValid sohLowVoltBattery;
    ::UInt8 vcuStatus;
    ::UInt8 personalMode;
    ::UInt8 commandFault;
    ::Int32 faultCode;
    ::ara::chassis::Float32WithValid frontMotorSpeed;
    ::ara::chassis::Float32WithValid frontAxleActualTorque;
    ::ara::chassis::Float32WithValid rearMotorSpeed;
    ::ara::chassis::Float32WithValid rearAxleActualTorque;
    ::UInt8 trailerStatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(throttlePedal);
        fun(throttlePedalRate);
        fun(driverOverride);
        fun(driveTorque);
        fun(driveTorqueMax);
        fun(driveTorqueMin);
        fun(driverDesiredTorque);
        fun(engineSpeed);
        fun(motorSpeed);
        fun(powerTrainReady);
        fun(driveTorqueCtrlAvl);
        fun(driveTorqueActv);
        fun(vcuAbortFbk);
        fun(fuelRange);
        fun(socHighVoltBattery);
        fun(sohHighVoltBattery);
        fun(socLowVoltBattery);
        fun(sohLowVoltBattery);
        fun(vcuStatus);
        fun(personalMode);
        fun(commandFault);
        fun(faultCode);
        fun(frontMotorSpeed);
        fun(frontAxleActualTorque);
        fun(rearMotorSpeed);
        fun(rearAxleActualTorque);
        fun(trailerStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(throttlePedal);
        fun(throttlePedalRate);
        fun(driverOverride);
        fun(driveTorque);
        fun(driveTorqueMax);
        fun(driveTorqueMin);
        fun(driverDesiredTorque);
        fun(engineSpeed);
        fun(motorSpeed);
        fun(powerTrainReady);
        fun(driveTorqueCtrlAvl);
        fun(driveTorqueActv);
        fun(vcuAbortFbk);
        fun(fuelRange);
        fun(socHighVoltBattery);
        fun(sohHighVoltBattery);
        fun(socLowVoltBattery);
        fun(sohLowVoltBattery);
        fun(vcuStatus);
        fun(personalMode);
        fun(commandFault);
        fun(faultCode);
        fun(frontMotorSpeed);
        fun(frontAxleActualTorque);
        fun(rearMotorSpeed);
        fun(rearAxleActualTorque);
        fun(trailerStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("throttlePedal", throttlePedal);
        fun("throttlePedalRate", throttlePedalRate);
        fun("driverOverride", driverOverride);
        fun("driveTorque", driveTorque);
        fun("driveTorqueMax", driveTorqueMax);
        fun("driveTorqueMin", driveTorqueMin);
        fun("driverDesiredTorque", driverDesiredTorque);
        fun("engineSpeed", engineSpeed);
        fun("motorSpeed", motorSpeed);
        fun("powerTrainReady", powerTrainReady);
        fun("driveTorqueCtrlAvl", driveTorqueCtrlAvl);
        fun("driveTorqueActv", driveTorqueActv);
        fun("vcuAbortFbk", vcuAbortFbk);
        fun("fuelRange", fuelRange);
        fun("socHighVoltBattery", socHighVoltBattery);
        fun("sohHighVoltBattery", sohHighVoltBattery);
        fun("socLowVoltBattery", socLowVoltBattery);
        fun("sohLowVoltBattery", sohLowVoltBattery);
        fun("vcuStatus", vcuStatus);
        fun("personalMode", personalMode);
        fun("commandFault", commandFault);
        fun("faultCode", faultCode);
        fun("frontMotorSpeed", frontMotorSpeed);
        fun("frontAxleActualTorque", frontAxleActualTorque);
        fun("rearMotorSpeed", rearMotorSpeed);
        fun("rearAxleActualTorque", rearAxleActualTorque);
        fun("trailerStatus", trailerStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("throttlePedal", throttlePedal);
        fun("throttlePedalRate", throttlePedalRate);
        fun("driverOverride", driverOverride);
        fun("driveTorque", driveTorque);
        fun("driveTorqueMax", driveTorqueMax);
        fun("driveTorqueMin", driveTorqueMin);
        fun("driverDesiredTorque", driverDesiredTorque);
        fun("engineSpeed", engineSpeed);
        fun("motorSpeed", motorSpeed);
        fun("powerTrainReady", powerTrainReady);
        fun("driveTorqueCtrlAvl", driveTorqueCtrlAvl);
        fun("driveTorqueActv", driveTorqueActv);
        fun("vcuAbortFbk", vcuAbortFbk);
        fun("fuelRange", fuelRange);
        fun("socHighVoltBattery", socHighVoltBattery);
        fun("sohHighVoltBattery", sohHighVoltBattery);
        fun("socLowVoltBattery", socLowVoltBattery);
        fun("sohLowVoltBattery", sohLowVoltBattery);
        fun("vcuStatus", vcuStatus);
        fun("personalMode", personalMode);
        fun("commandFault", commandFault);
        fun("faultCode", faultCode);
        fun("frontMotorSpeed", frontMotorSpeed);
        fun("frontAxleActualTorque", frontAxleActualTorque);
        fun("rearMotorSpeed", rearMotorSpeed);
        fun("rearAxleActualTorque", rearAxleActualTorque);
        fun("trailerStatus", trailerStatus);
    }

    bool operator==(const ::ara::chassis::ThrottleInfo& t) const
    {
        return (throttlePedal == t.throttlePedal) && (throttlePedalRate == t.throttlePedalRate) && (driverOverride == t.driverOverride) && (driveTorque == t.driveTorque) && (driveTorqueMax == t.driveTorqueMax) && (driveTorqueMin == t.driveTorqueMin) && (driverDesiredTorque == t.driverDesiredTorque) && (engineSpeed == t.engineSpeed) && (motorSpeed == t.motorSpeed) && (powerTrainReady == t.powerTrainReady) && (driveTorqueCtrlAvl == t.driveTorqueCtrlAvl) && (driveTorqueActv == t.driveTorqueActv) && (vcuAbortFbk == t.vcuAbortFbk) && (fuelRange == t.fuelRange) && (socHighVoltBattery == t.socHighVoltBattery) && (sohHighVoltBattery == t.sohHighVoltBattery) && (socLowVoltBattery == t.socLowVoltBattery) && (sohLowVoltBattery == t.sohLowVoltBattery) && (vcuStatus == t.vcuStatus) && (personalMode == t.personalMode) && (commandFault == t.commandFault) && (faultCode == t.faultCode) && (frontMotorSpeed == t.frontMotorSpeed) && (frontAxleActualTorque == t.frontAxleActualTorque) && (rearMotorSpeed == t.rearMotorSpeed) && (rearAxleActualTorque == t.rearAxleActualTorque) && (trailerStatus == t.trailerStatus);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_THROTTLEINFO_H
