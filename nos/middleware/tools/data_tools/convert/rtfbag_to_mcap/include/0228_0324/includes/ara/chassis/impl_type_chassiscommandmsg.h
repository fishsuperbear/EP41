/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_CHASSISCOMMANDMSG_H
#define ARA_CHASSIS_IMPL_TYPE_CHASSISCOMMANDMSG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"
#include "impl_type_float.h"
#include "impl_type_boolean.h"
#include "ara/chassis/impl_type_gearcmd.h"
#include "ara/chassis/impl_type_chassiscmdheader.h"

namespace ara {
namespace chassis {
struct ChassisCommandMsg {
    ::UInt8 source;
    ::UInt16 moduleId;
    ::UInt16 adsFunction;
    ::UInt16 adsWorkStatus;
    ::Float steerAngleCmd;
    ::Boolean steerAngleValid;
    ::UInt8 steerAngleState;
    ::UInt8 steerAngleClean;
    ::Float steerAngleMaxLimit;
    ::Float steerAngleRateMaxLimit;
    ::UInt8 steerAngleMode;
    ::Float steerTorqueSmoothFactor;
    ::Float steerTorqueCmd;
    ::Boolean steerTorqueValid;
    ::UInt8 steerTorqueState;
    ::Float steerTorqueMaxLimit;
    ::UInt8 steerTorqueMode;
    ::UInt8 steerVibrationCmd;
    ::UInt8 steerVibrationState;
    ::Float steerCrvtCmd;
    ::Boolean steerCrvtValid;
    ::UInt8 steerCrvtState;
    ::Float steerCrvtGrdCmd;
    ::Float steerFactorSstyCmd;
    ::UInt8 steerForceLmtReqSts;
    ::Float longAccelerationCmd;
    ::Boolean longAccelerationValid;
    ::UInt8 longAccelerationState;
    ::UInt8 longControlMode;
    ::Float jerkMax;
    ::Float jerkMin;
    ::Float comfortBoundaryUp;
    ::Float comfortBoundaryLow;
    ::Float longAccelerationPredictCmd;
    ::Float aebAccelerationCmd;
    ::Boolean aebAccelerationValid;
    ::UInt8 aebAccelerationState;
    ::UInt8 prefillEnable;
    ::UInt8 jerkBrakeEnable;
    ::Boolean ebaCtrlEnable;
    ::UInt8 ebaLevel;
    ::Float driveTorqueCmd;
    ::Boolean driveTorqueValid;
    ::UInt8 driveTorqueState;
    ::Float driveTorqueLimit;
    ::UInt8 rapidRespEnable;
    ::UInt8 brakePreferEnable;
    ::Float brakeTorqueCmd;
    ::Boolean brakeTorqueValid;
    ::UInt8 brakeTorqueState;
    ::UInt8 brakeTorqueMode;
    ::Float brakeTorqueLimit;
    ::UInt8 brakeLightOnReq;
    ::Float aebBrakeTorqueCmd;
    ::Boolean aebBrakeTorqueValid;
    ::UInt8 aebBrakeTorqueState;
    ::Float aebBrakeTorqueLimit;
    ::UInt8 stopEnable;
    ::UInt8 driveOffEnable;
    ::UInt8 brakeHoldReq;
    ::UInt8 emergencyStopEnable;
    ::Boolean emergencyStopEnableValid;
    ::UInt8 epbCmd;
    ::UInt8 epbCmdState;
    ::UInt8 apaDistSpdEnable;
    ::Float speedLimit;
    ::Float distanceRemain;
    ::UInt8 immediateStopEnable;
    ::UInt8 rpaEnable;
    ::ara::chassis::GearCmd gearCmd;
    ::Boolean gearCmdValid;
    ::UInt8 gearCmdState;
    ::UInt8 gearCmdClean;
    ::Float brakePedalCmd;
    ::Boolean brakePedalValid;
    ::UInt8 brakePedalState;
    ::UInt8 brakePedalClean;
    ::Float throttlePedalCmd;
    ::Boolean throttlePedalValid;
    ::UInt8 throttlePedalState;
    ::UInt8 throttlePedalClean;
    ::ara::chassis::ChassisCmdHeader header;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(source);
        fun(moduleId);
        fun(adsFunction);
        fun(adsWorkStatus);
        fun(steerAngleCmd);
        fun(steerAngleValid);
        fun(steerAngleState);
        fun(steerAngleClean);
        fun(steerAngleMaxLimit);
        fun(steerAngleRateMaxLimit);
        fun(steerAngleMode);
        fun(steerTorqueSmoothFactor);
        fun(steerTorqueCmd);
        fun(steerTorqueValid);
        fun(steerTorqueState);
        fun(steerTorqueMaxLimit);
        fun(steerTorqueMode);
        fun(steerVibrationCmd);
        fun(steerVibrationState);
        fun(steerCrvtCmd);
        fun(steerCrvtValid);
        fun(steerCrvtState);
        fun(steerCrvtGrdCmd);
        fun(steerFactorSstyCmd);
        fun(steerForceLmtReqSts);
        fun(longAccelerationCmd);
        fun(longAccelerationValid);
        fun(longAccelerationState);
        fun(longControlMode);
        fun(jerkMax);
        fun(jerkMin);
        fun(comfortBoundaryUp);
        fun(comfortBoundaryLow);
        fun(longAccelerationPredictCmd);
        fun(aebAccelerationCmd);
        fun(aebAccelerationValid);
        fun(aebAccelerationState);
        fun(prefillEnable);
        fun(jerkBrakeEnable);
        fun(ebaCtrlEnable);
        fun(ebaLevel);
        fun(driveTorqueCmd);
        fun(driveTorqueValid);
        fun(driveTorqueState);
        fun(driveTorqueLimit);
        fun(rapidRespEnable);
        fun(brakePreferEnable);
        fun(brakeTorqueCmd);
        fun(brakeTorqueValid);
        fun(brakeTorqueState);
        fun(brakeTorqueMode);
        fun(brakeTorqueLimit);
        fun(brakeLightOnReq);
        fun(aebBrakeTorqueCmd);
        fun(aebBrakeTorqueValid);
        fun(aebBrakeTorqueState);
        fun(aebBrakeTorqueLimit);
        fun(stopEnable);
        fun(driveOffEnable);
        fun(brakeHoldReq);
        fun(emergencyStopEnable);
        fun(emergencyStopEnableValid);
        fun(epbCmd);
        fun(epbCmdState);
        fun(apaDistSpdEnable);
        fun(speedLimit);
        fun(distanceRemain);
        fun(immediateStopEnable);
        fun(rpaEnable);
        fun(gearCmd);
        fun(gearCmdValid);
        fun(gearCmdState);
        fun(gearCmdClean);
        fun(brakePedalCmd);
        fun(brakePedalValid);
        fun(brakePedalState);
        fun(brakePedalClean);
        fun(throttlePedalCmd);
        fun(throttlePedalValid);
        fun(throttlePedalState);
        fun(throttlePedalClean);
        fun(header);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(source);
        fun(moduleId);
        fun(adsFunction);
        fun(adsWorkStatus);
        fun(steerAngleCmd);
        fun(steerAngleValid);
        fun(steerAngleState);
        fun(steerAngleClean);
        fun(steerAngleMaxLimit);
        fun(steerAngleRateMaxLimit);
        fun(steerAngleMode);
        fun(steerTorqueSmoothFactor);
        fun(steerTorqueCmd);
        fun(steerTorqueValid);
        fun(steerTorqueState);
        fun(steerTorqueMaxLimit);
        fun(steerTorqueMode);
        fun(steerVibrationCmd);
        fun(steerVibrationState);
        fun(steerCrvtCmd);
        fun(steerCrvtValid);
        fun(steerCrvtState);
        fun(steerCrvtGrdCmd);
        fun(steerFactorSstyCmd);
        fun(steerForceLmtReqSts);
        fun(longAccelerationCmd);
        fun(longAccelerationValid);
        fun(longAccelerationState);
        fun(longControlMode);
        fun(jerkMax);
        fun(jerkMin);
        fun(comfortBoundaryUp);
        fun(comfortBoundaryLow);
        fun(longAccelerationPredictCmd);
        fun(aebAccelerationCmd);
        fun(aebAccelerationValid);
        fun(aebAccelerationState);
        fun(prefillEnable);
        fun(jerkBrakeEnable);
        fun(ebaCtrlEnable);
        fun(ebaLevel);
        fun(driveTorqueCmd);
        fun(driveTorqueValid);
        fun(driveTorqueState);
        fun(driveTorqueLimit);
        fun(rapidRespEnable);
        fun(brakePreferEnable);
        fun(brakeTorqueCmd);
        fun(brakeTorqueValid);
        fun(brakeTorqueState);
        fun(brakeTorqueMode);
        fun(brakeTorqueLimit);
        fun(brakeLightOnReq);
        fun(aebBrakeTorqueCmd);
        fun(aebBrakeTorqueValid);
        fun(aebBrakeTorqueState);
        fun(aebBrakeTorqueLimit);
        fun(stopEnable);
        fun(driveOffEnable);
        fun(brakeHoldReq);
        fun(emergencyStopEnable);
        fun(emergencyStopEnableValid);
        fun(epbCmd);
        fun(epbCmdState);
        fun(apaDistSpdEnable);
        fun(speedLimit);
        fun(distanceRemain);
        fun(immediateStopEnable);
        fun(rpaEnable);
        fun(gearCmd);
        fun(gearCmdValid);
        fun(gearCmdState);
        fun(gearCmdClean);
        fun(brakePedalCmd);
        fun(brakePedalValid);
        fun(brakePedalState);
        fun(brakePedalClean);
        fun(throttlePedalCmd);
        fun(throttlePedalValid);
        fun(throttlePedalState);
        fun(throttlePedalClean);
        fun(header);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("source", source);
        fun("moduleId", moduleId);
        fun("adsFunction", adsFunction);
        fun("adsWorkStatus", adsWorkStatus);
        fun("steerAngleCmd", steerAngleCmd);
        fun("steerAngleValid", steerAngleValid);
        fun("steerAngleState", steerAngleState);
        fun("steerAngleClean", steerAngleClean);
        fun("steerAngleMaxLimit", steerAngleMaxLimit);
        fun("steerAngleRateMaxLimit", steerAngleRateMaxLimit);
        fun("steerAngleMode", steerAngleMode);
        fun("steerTorqueSmoothFactor", steerTorqueSmoothFactor);
        fun("steerTorqueCmd", steerTorqueCmd);
        fun("steerTorqueValid", steerTorqueValid);
        fun("steerTorqueState", steerTorqueState);
        fun("steerTorqueMaxLimit", steerTorqueMaxLimit);
        fun("steerTorqueMode", steerTorqueMode);
        fun("steerVibrationCmd", steerVibrationCmd);
        fun("steerVibrationState", steerVibrationState);
        fun("steerCrvtCmd", steerCrvtCmd);
        fun("steerCrvtValid", steerCrvtValid);
        fun("steerCrvtState", steerCrvtState);
        fun("steerCrvtGrdCmd", steerCrvtGrdCmd);
        fun("steerFactorSstyCmd", steerFactorSstyCmd);
        fun("steerForceLmtReqSts", steerForceLmtReqSts);
        fun("longAccelerationCmd", longAccelerationCmd);
        fun("longAccelerationValid", longAccelerationValid);
        fun("longAccelerationState", longAccelerationState);
        fun("longControlMode", longControlMode);
        fun("jerkMax", jerkMax);
        fun("jerkMin", jerkMin);
        fun("comfortBoundaryUp", comfortBoundaryUp);
        fun("comfortBoundaryLow", comfortBoundaryLow);
        fun("longAccelerationPredictCmd", longAccelerationPredictCmd);
        fun("aebAccelerationCmd", aebAccelerationCmd);
        fun("aebAccelerationValid", aebAccelerationValid);
        fun("aebAccelerationState", aebAccelerationState);
        fun("prefillEnable", prefillEnable);
        fun("jerkBrakeEnable", jerkBrakeEnable);
        fun("ebaCtrlEnable", ebaCtrlEnable);
        fun("ebaLevel", ebaLevel);
        fun("driveTorqueCmd", driveTorqueCmd);
        fun("driveTorqueValid", driveTorqueValid);
        fun("driveTorqueState", driveTorqueState);
        fun("driveTorqueLimit", driveTorqueLimit);
        fun("rapidRespEnable", rapidRespEnable);
        fun("brakePreferEnable", brakePreferEnable);
        fun("brakeTorqueCmd", brakeTorqueCmd);
        fun("brakeTorqueValid", brakeTorqueValid);
        fun("brakeTorqueState", brakeTorqueState);
        fun("brakeTorqueMode", brakeTorqueMode);
        fun("brakeTorqueLimit", brakeTorqueLimit);
        fun("brakeLightOnReq", brakeLightOnReq);
        fun("aebBrakeTorqueCmd", aebBrakeTorqueCmd);
        fun("aebBrakeTorqueValid", aebBrakeTorqueValid);
        fun("aebBrakeTorqueState", aebBrakeTorqueState);
        fun("aebBrakeTorqueLimit", aebBrakeTorqueLimit);
        fun("stopEnable", stopEnable);
        fun("driveOffEnable", driveOffEnable);
        fun("brakeHoldReq", brakeHoldReq);
        fun("emergencyStopEnable", emergencyStopEnable);
        fun("emergencyStopEnableValid", emergencyStopEnableValid);
        fun("epbCmd", epbCmd);
        fun("epbCmdState", epbCmdState);
        fun("apaDistSpdEnable", apaDistSpdEnable);
        fun("speedLimit", speedLimit);
        fun("distanceRemain", distanceRemain);
        fun("immediateStopEnable", immediateStopEnable);
        fun("rpaEnable", rpaEnable);
        fun("gearCmd", gearCmd);
        fun("gearCmdValid", gearCmdValid);
        fun("gearCmdState", gearCmdState);
        fun("gearCmdClean", gearCmdClean);
        fun("brakePedalCmd", brakePedalCmd);
        fun("brakePedalValid", brakePedalValid);
        fun("brakePedalState", brakePedalState);
        fun("brakePedalClean", brakePedalClean);
        fun("throttlePedalCmd", throttlePedalCmd);
        fun("throttlePedalValid", throttlePedalValid);
        fun("throttlePedalState", throttlePedalState);
        fun("throttlePedalClean", throttlePedalClean);
        fun("header", header);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("source", source);
        fun("moduleId", moduleId);
        fun("adsFunction", adsFunction);
        fun("adsWorkStatus", adsWorkStatus);
        fun("steerAngleCmd", steerAngleCmd);
        fun("steerAngleValid", steerAngleValid);
        fun("steerAngleState", steerAngleState);
        fun("steerAngleClean", steerAngleClean);
        fun("steerAngleMaxLimit", steerAngleMaxLimit);
        fun("steerAngleRateMaxLimit", steerAngleRateMaxLimit);
        fun("steerAngleMode", steerAngleMode);
        fun("steerTorqueSmoothFactor", steerTorqueSmoothFactor);
        fun("steerTorqueCmd", steerTorqueCmd);
        fun("steerTorqueValid", steerTorqueValid);
        fun("steerTorqueState", steerTorqueState);
        fun("steerTorqueMaxLimit", steerTorqueMaxLimit);
        fun("steerTorqueMode", steerTorqueMode);
        fun("steerVibrationCmd", steerVibrationCmd);
        fun("steerVibrationState", steerVibrationState);
        fun("steerCrvtCmd", steerCrvtCmd);
        fun("steerCrvtValid", steerCrvtValid);
        fun("steerCrvtState", steerCrvtState);
        fun("steerCrvtGrdCmd", steerCrvtGrdCmd);
        fun("steerFactorSstyCmd", steerFactorSstyCmd);
        fun("steerForceLmtReqSts", steerForceLmtReqSts);
        fun("longAccelerationCmd", longAccelerationCmd);
        fun("longAccelerationValid", longAccelerationValid);
        fun("longAccelerationState", longAccelerationState);
        fun("longControlMode", longControlMode);
        fun("jerkMax", jerkMax);
        fun("jerkMin", jerkMin);
        fun("comfortBoundaryUp", comfortBoundaryUp);
        fun("comfortBoundaryLow", comfortBoundaryLow);
        fun("longAccelerationPredictCmd", longAccelerationPredictCmd);
        fun("aebAccelerationCmd", aebAccelerationCmd);
        fun("aebAccelerationValid", aebAccelerationValid);
        fun("aebAccelerationState", aebAccelerationState);
        fun("prefillEnable", prefillEnable);
        fun("jerkBrakeEnable", jerkBrakeEnable);
        fun("ebaCtrlEnable", ebaCtrlEnable);
        fun("ebaLevel", ebaLevel);
        fun("driveTorqueCmd", driveTorqueCmd);
        fun("driveTorqueValid", driveTorqueValid);
        fun("driveTorqueState", driveTorqueState);
        fun("driveTorqueLimit", driveTorqueLimit);
        fun("rapidRespEnable", rapidRespEnable);
        fun("brakePreferEnable", brakePreferEnable);
        fun("brakeTorqueCmd", brakeTorqueCmd);
        fun("brakeTorqueValid", brakeTorqueValid);
        fun("brakeTorqueState", brakeTorqueState);
        fun("brakeTorqueMode", brakeTorqueMode);
        fun("brakeTorqueLimit", brakeTorqueLimit);
        fun("brakeLightOnReq", brakeLightOnReq);
        fun("aebBrakeTorqueCmd", aebBrakeTorqueCmd);
        fun("aebBrakeTorqueValid", aebBrakeTorqueValid);
        fun("aebBrakeTorqueState", aebBrakeTorqueState);
        fun("aebBrakeTorqueLimit", aebBrakeTorqueLimit);
        fun("stopEnable", stopEnable);
        fun("driveOffEnable", driveOffEnable);
        fun("brakeHoldReq", brakeHoldReq);
        fun("emergencyStopEnable", emergencyStopEnable);
        fun("emergencyStopEnableValid", emergencyStopEnableValid);
        fun("epbCmd", epbCmd);
        fun("epbCmdState", epbCmdState);
        fun("apaDistSpdEnable", apaDistSpdEnable);
        fun("speedLimit", speedLimit);
        fun("distanceRemain", distanceRemain);
        fun("immediateStopEnable", immediateStopEnable);
        fun("rpaEnable", rpaEnable);
        fun("gearCmd", gearCmd);
        fun("gearCmdValid", gearCmdValid);
        fun("gearCmdState", gearCmdState);
        fun("gearCmdClean", gearCmdClean);
        fun("brakePedalCmd", brakePedalCmd);
        fun("brakePedalValid", brakePedalValid);
        fun("brakePedalState", brakePedalState);
        fun("brakePedalClean", brakePedalClean);
        fun("throttlePedalCmd", throttlePedalCmd);
        fun("throttlePedalValid", throttlePedalValid);
        fun("throttlePedalState", throttlePedalState);
        fun("throttlePedalClean", throttlePedalClean);
        fun("header", header);
    }

    bool operator==(const ::ara::chassis::ChassisCommandMsg& t) const
    {
        return (source == t.source) && (moduleId == t.moduleId) && (adsFunction == t.adsFunction) && (adsWorkStatus == t.adsWorkStatus) && (fabs(static_cast<double>(steerAngleCmd - t.steerAngleCmd)) < DBL_EPSILON) && (steerAngleValid == t.steerAngleValid) && (steerAngleState == t.steerAngleState) && (steerAngleClean == t.steerAngleClean) && (fabs(static_cast<double>(steerAngleMaxLimit - t.steerAngleMaxLimit)) < DBL_EPSILON) && (fabs(static_cast<double>(steerAngleRateMaxLimit - t.steerAngleRateMaxLimit)) < DBL_EPSILON) && (steerAngleMode == t.steerAngleMode) && (fabs(static_cast<double>(steerTorqueSmoothFactor - t.steerTorqueSmoothFactor)) < DBL_EPSILON) && (fabs(static_cast<double>(steerTorqueCmd - t.steerTorqueCmd)) < DBL_EPSILON) && (steerTorqueValid == t.steerTorqueValid) && (steerTorqueState == t.steerTorqueState) && (fabs(static_cast<double>(steerTorqueMaxLimit - t.steerTorqueMaxLimit)) < DBL_EPSILON) && (steerTorqueMode == t.steerTorqueMode) && (steerVibrationCmd == t.steerVibrationCmd) && (steerVibrationState == t.steerVibrationState) && (fabs(static_cast<double>(steerCrvtCmd - t.steerCrvtCmd)) < DBL_EPSILON) && (steerCrvtValid == t.steerCrvtValid) && (steerCrvtState == t.steerCrvtState) && (fabs(static_cast<double>(steerCrvtGrdCmd - t.steerCrvtGrdCmd)) < DBL_EPSILON) && (fabs(static_cast<double>(steerFactorSstyCmd - t.steerFactorSstyCmd)) < DBL_EPSILON) && (steerForceLmtReqSts == t.steerForceLmtReqSts) && (fabs(static_cast<double>(longAccelerationCmd - t.longAccelerationCmd)) < DBL_EPSILON) && (longAccelerationValid == t.longAccelerationValid) && (longAccelerationState == t.longAccelerationState) && (longControlMode == t.longControlMode) && (fabs(static_cast<double>(jerkMax - t.jerkMax)) < DBL_EPSILON) && (fabs(static_cast<double>(jerkMin - t.jerkMin)) < DBL_EPSILON) && (fabs(static_cast<double>(comfortBoundaryUp - t.comfortBoundaryUp)) < DBL_EPSILON) && (fabs(static_cast<double>(comfortBoundaryLow - t.comfortBoundaryLow)) < DBL_EPSILON) && (fabs(static_cast<double>(longAccelerationPredictCmd - t.longAccelerationPredictCmd)) < DBL_EPSILON) && (fabs(static_cast<double>(aebAccelerationCmd - t.aebAccelerationCmd)) < DBL_EPSILON) && (aebAccelerationValid == t.aebAccelerationValid) && (aebAccelerationState == t.aebAccelerationState) && (prefillEnable == t.prefillEnable) && (jerkBrakeEnable == t.jerkBrakeEnable) && (ebaCtrlEnable == t.ebaCtrlEnable) && (ebaLevel == t.ebaLevel) && (fabs(static_cast<double>(driveTorqueCmd - t.driveTorqueCmd)) < DBL_EPSILON) && (driveTorqueValid == t.driveTorqueValid) && (driveTorqueState == t.driveTorqueState) && (fabs(static_cast<double>(driveTorqueLimit - t.driveTorqueLimit)) < DBL_EPSILON) && (rapidRespEnable == t.rapidRespEnable) && (brakePreferEnable == t.brakePreferEnable) && (fabs(static_cast<double>(brakeTorqueCmd - t.brakeTorqueCmd)) < DBL_EPSILON) && (brakeTorqueValid == t.brakeTorqueValid) && (brakeTorqueState == t.brakeTorqueState) && (brakeTorqueMode == t.brakeTorqueMode) && (fabs(static_cast<double>(brakeTorqueLimit - t.brakeTorqueLimit)) < DBL_EPSILON) && (brakeLightOnReq == t.brakeLightOnReq) && (fabs(static_cast<double>(aebBrakeTorqueCmd - t.aebBrakeTorqueCmd)) < DBL_EPSILON) && (aebBrakeTorqueValid == t.aebBrakeTorqueValid) && (aebBrakeTorqueState == t.aebBrakeTorqueState) && (fabs(static_cast<double>(aebBrakeTorqueLimit - t.aebBrakeTorqueLimit)) < DBL_EPSILON) && (stopEnable == t.stopEnable) && (driveOffEnable == t.driveOffEnable) && (brakeHoldReq == t.brakeHoldReq) && (emergencyStopEnable == t.emergencyStopEnable) && (emergencyStopEnableValid == t.emergencyStopEnableValid) && (epbCmd == t.epbCmd) && (epbCmdState == t.epbCmdState) && (apaDistSpdEnable == t.apaDistSpdEnable) && (fabs(static_cast<double>(speedLimit - t.speedLimit)) < DBL_EPSILON) && (fabs(static_cast<double>(distanceRemain - t.distanceRemain)) < DBL_EPSILON) && (immediateStopEnable == t.immediateStopEnable) && (rpaEnable == t.rpaEnable) && (gearCmd == t.gearCmd) && (gearCmdValid == t.gearCmdValid) && (gearCmdState == t.gearCmdState) && (gearCmdClean == t.gearCmdClean) && (fabs(static_cast<double>(brakePedalCmd - t.brakePedalCmd)) < DBL_EPSILON) && (brakePedalValid == t.brakePedalValid) && (brakePedalState == t.brakePedalState) && (brakePedalClean == t.brakePedalClean) && (fabs(static_cast<double>(throttlePedalCmd - t.throttlePedalCmd)) < DBL_EPSILON) && (throttlePedalValid == t.throttlePedalValid) && (throttlePedalState == t.throttlePedalState) && (throttlePedalClean == t.throttlePedalClean) && (header == t.header);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_CHASSISCOMMANDMSG_H
