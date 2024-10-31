/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_STEERINFO_H
#define ARA_CHASSIS_IMPL_TYPE_STEERINFO_H
#include <cfloat>
#include <cmath>
#include "ara/chassis/impl_type_float32withvalid.h"
#include "ara/chassis/impl_type_uint8withvalid.h"
#include "impl_type_uint8.h"
#include "impl_type_int32.h"
#include "impl_type_boolean.h"

namespace ara {
namespace chassis {
struct SteerInfo {
    ::ara::chassis::Float32WithValid steerAngle;
    ::ara::chassis::Float32WithValid steerAngleRate;
    ::ara::chassis::Float32WithValid steerPinionAngle;
    ::ara::chassis::Float32WithValid steerPinionAngleRate;
    ::ara::chassis::Float32WithValid frontSteerAngle;
    ::ara::chassis::Float32WithValid frontSteerAngleRate;
    ::ara::chassis::Uint8WithValid driverHandOn;
    ::ara::chassis::Float32WithValid driverHandTorque;
    ::ara::chassis::Float32WithValid steerTorque;
    ::ara::chassis::Float32WithValid motorCurrent;
    ::ara::chassis::Uint8WithValid driverOverride;
    ::UInt8 personalMode;
    ::UInt8 commandFault;
    ::UInt8 epsStatusMaster;
    ::UInt8 epsStatusSlave;
    ::Int32 faultCode;
    ::Boolean drvrSteerMonrEnaSts;
    ::ara::chassis::Uint8WithValid handsOffConf;
    ::ara::chassis::Float32WithValid epsMotorTq;
    ::UInt8 epsMotorTemp;
    ::UInt8 ldwWarnSts;
    ::UInt8 epsOperMod;
    ::UInt8 epsAbortFb;
    ::UInt8 epsTqSensSts;
    ::UInt8 epsSteerAgSensFilr;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(steerAngle);
        fun(steerAngleRate);
        fun(steerPinionAngle);
        fun(steerPinionAngleRate);
        fun(frontSteerAngle);
        fun(frontSteerAngleRate);
        fun(driverHandOn);
        fun(driverHandTorque);
        fun(steerTorque);
        fun(motorCurrent);
        fun(driverOverride);
        fun(personalMode);
        fun(commandFault);
        fun(epsStatusMaster);
        fun(epsStatusSlave);
        fun(faultCode);
        fun(drvrSteerMonrEnaSts);
        fun(handsOffConf);
        fun(epsMotorTq);
        fun(epsMotorTemp);
        fun(ldwWarnSts);
        fun(epsOperMod);
        fun(epsAbortFb);
        fun(epsTqSensSts);
        fun(epsSteerAgSensFilr);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(steerAngle);
        fun(steerAngleRate);
        fun(steerPinionAngle);
        fun(steerPinionAngleRate);
        fun(frontSteerAngle);
        fun(frontSteerAngleRate);
        fun(driverHandOn);
        fun(driverHandTorque);
        fun(steerTorque);
        fun(motorCurrent);
        fun(driverOverride);
        fun(personalMode);
        fun(commandFault);
        fun(epsStatusMaster);
        fun(epsStatusSlave);
        fun(faultCode);
        fun(drvrSteerMonrEnaSts);
        fun(handsOffConf);
        fun(epsMotorTq);
        fun(epsMotorTemp);
        fun(ldwWarnSts);
        fun(epsOperMod);
        fun(epsAbortFb);
        fun(epsTqSensSts);
        fun(epsSteerAgSensFilr);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("steerAngle", steerAngle);
        fun("steerAngleRate", steerAngleRate);
        fun("steerPinionAngle", steerPinionAngle);
        fun("steerPinionAngleRate", steerPinionAngleRate);
        fun("frontSteerAngle", frontSteerAngle);
        fun("frontSteerAngleRate", frontSteerAngleRate);
        fun("driverHandOn", driverHandOn);
        fun("driverHandTorque", driverHandTorque);
        fun("steerTorque", steerTorque);
        fun("motorCurrent", motorCurrent);
        fun("driverOverride", driverOverride);
        fun("personalMode", personalMode);
        fun("commandFault", commandFault);
        fun("epsStatusMaster", epsStatusMaster);
        fun("epsStatusSlave", epsStatusSlave);
        fun("faultCode", faultCode);
        fun("drvrSteerMonrEnaSts", drvrSteerMonrEnaSts);
        fun("handsOffConf", handsOffConf);
        fun("epsMotorTq", epsMotorTq);
        fun("epsMotorTemp", epsMotorTemp);
        fun("ldwWarnSts", ldwWarnSts);
        fun("epsOperMod", epsOperMod);
        fun("epsAbortFb", epsAbortFb);
        fun("epsTqSensSts", epsTqSensSts);
        fun("epsSteerAgSensFilr", epsSteerAgSensFilr);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("steerAngle", steerAngle);
        fun("steerAngleRate", steerAngleRate);
        fun("steerPinionAngle", steerPinionAngle);
        fun("steerPinionAngleRate", steerPinionAngleRate);
        fun("frontSteerAngle", frontSteerAngle);
        fun("frontSteerAngleRate", frontSteerAngleRate);
        fun("driverHandOn", driverHandOn);
        fun("driverHandTorque", driverHandTorque);
        fun("steerTorque", steerTorque);
        fun("motorCurrent", motorCurrent);
        fun("driverOverride", driverOverride);
        fun("personalMode", personalMode);
        fun("commandFault", commandFault);
        fun("epsStatusMaster", epsStatusMaster);
        fun("epsStatusSlave", epsStatusSlave);
        fun("faultCode", faultCode);
        fun("drvrSteerMonrEnaSts", drvrSteerMonrEnaSts);
        fun("handsOffConf", handsOffConf);
        fun("epsMotorTq", epsMotorTq);
        fun("epsMotorTemp", epsMotorTemp);
        fun("ldwWarnSts", ldwWarnSts);
        fun("epsOperMod", epsOperMod);
        fun("epsAbortFb", epsAbortFb);
        fun("epsTqSensSts", epsTqSensSts);
        fun("epsSteerAgSensFilr", epsSteerAgSensFilr);
    }

    bool operator==(const ::ara::chassis::SteerInfo& t) const
    {
        return (steerAngle == t.steerAngle) && (steerAngleRate == t.steerAngleRate) && (steerPinionAngle == t.steerPinionAngle) && (steerPinionAngleRate == t.steerPinionAngleRate) && (frontSteerAngle == t.frontSteerAngle) && (frontSteerAngleRate == t.frontSteerAngleRate) && (driverHandOn == t.driverHandOn) && (driverHandTorque == t.driverHandTorque) && (steerTorque == t.steerTorque) && (motorCurrent == t.motorCurrent) && (driverOverride == t.driverOverride) && (personalMode == t.personalMode) && (commandFault == t.commandFault) && (epsStatusMaster == t.epsStatusMaster) && (epsStatusSlave == t.epsStatusSlave) && (faultCode == t.faultCode) && (drvrSteerMonrEnaSts == t.drvrSteerMonrEnaSts) && (handsOffConf == t.handsOffConf) && (epsMotorTq == t.epsMotorTq) && (epsMotorTemp == t.epsMotorTemp) && (ldwWarnSts == t.ldwWarnSts) && (epsOperMod == t.epsOperMod) && (epsAbortFb == t.epsAbortFb) && (epsTqSensSts == t.epsTqSensSts) && (epsSteerAgSensFilr == t.epsSteerAgSensFilr);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_STEERINFO_H
