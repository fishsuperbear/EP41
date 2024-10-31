/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_BODY_IMPL_TYPE_BODYCOMMANDMSG_H
#define ARA_BODY_IMPL_TYPE_BODYCOMMANDMSG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"
#include "impl_type_boolean.h"
#include "impl_type_bodycmdheader.h"

namespace ara {
namespace body {
struct BodyCommandMsg {
    ::UInt8 source;
    ::UInt16 adsFunction;
    ::UInt16 adsWorkStatus;
    ::UInt16 moduleId;
    ::UInt8 turnLightCmd;
    ::UInt8 hazardLightCmd;
    ::UInt8 reverseLightCmd;
    ::UInt8 hornCmd;
    ::UInt8 highLowBeamLightCmd;
    ::UInt8 dayTimeRunLightCmd;
    ::UInt8 sidePositionLightCmd;
    ::UInt8 windShieldWashCmd;
    ::UInt8 windShieldWiperCmd;
    ::UInt8 rearViewMirrorCmd;
    ::UInt8 parkFinishRemind;
    ::UInt8 leftFrontWindowCmd;
    ::UInt8 rightFrontWindowCmd;
    ::UInt8 leftRearWindowCmd;
    ::UInt8 rightRearWindowCmd;
    ::UInt8 topWindowCmd;
    ::UInt8 turnLightState;
    ::UInt8 hazardLightState;
    ::UInt8 brakeLightCmd;
    ::UInt8 brakeLightValid;
    ::Boolean hazardLightValid;
    ::Boolean highLowBeamLighValid;
    ::Boolean hornValid;
    ::Boolean leftFrontWindowValid;
    ::Boolean leftRearWindowValid;
    ::Boolean rearViewMirrorValid;
    ::Boolean rightFrontWindowValid;
    ::Boolean rightRearWindowValid;
    ::Boolean topWindowValid;
    ::Boolean turnLightValid;
    ::Boolean windShieldWashValid;
    ::Boolean windShieldWiperValid;
    ::BodyCmdHeader header;
    ::UInt8 doorLockCmd;
    ::UInt8 turnLightCmdFrequency;
    ::UInt8 brakeLightCmdFrequency;
    ::UInt8 hazardLightCmdFrequency;
    ::Boolean doorLockCmdValid;
    ::UInt8 seatVibrationCmd;
    ::UInt8 safeBeltFastenCmd;
    ::UInt8 openDoorWarningFL;
    ::UInt8 openDoorWarningFR;
    ::UInt8 openDoorWarningRL;
    ::UInt8 openDoorWarningRR;
    ::UInt8 rearViewMirrorWarningLeft;
    ::UInt8 rearViewMirrorWarningRight;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(source);
        fun(adsFunction);
        fun(adsWorkStatus);
        fun(moduleId);
        fun(turnLightCmd);
        fun(hazardLightCmd);
        fun(reverseLightCmd);
        fun(hornCmd);
        fun(highLowBeamLightCmd);
        fun(dayTimeRunLightCmd);
        fun(sidePositionLightCmd);
        fun(windShieldWashCmd);
        fun(windShieldWiperCmd);
        fun(rearViewMirrorCmd);
        fun(parkFinishRemind);
        fun(leftFrontWindowCmd);
        fun(rightFrontWindowCmd);
        fun(leftRearWindowCmd);
        fun(rightRearWindowCmd);
        fun(topWindowCmd);
        fun(turnLightState);
        fun(hazardLightState);
        fun(brakeLightCmd);
        fun(brakeLightValid);
        fun(hazardLightValid);
        fun(highLowBeamLighValid);
        fun(hornValid);
        fun(leftFrontWindowValid);
        fun(leftRearWindowValid);
        fun(rearViewMirrorValid);
        fun(rightFrontWindowValid);
        fun(rightRearWindowValid);
        fun(topWindowValid);
        fun(turnLightValid);
        fun(windShieldWashValid);
        fun(windShieldWiperValid);
        fun(header);
        fun(doorLockCmd);
        fun(turnLightCmdFrequency);
        fun(brakeLightCmdFrequency);
        fun(hazardLightCmdFrequency);
        fun(doorLockCmdValid);
        fun(seatVibrationCmd);
        fun(safeBeltFastenCmd);
        fun(openDoorWarningFL);
        fun(openDoorWarningFR);
        fun(openDoorWarningRL);
        fun(openDoorWarningRR);
        fun(rearViewMirrorWarningLeft);
        fun(rearViewMirrorWarningRight);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(source);
        fun(adsFunction);
        fun(adsWorkStatus);
        fun(moduleId);
        fun(turnLightCmd);
        fun(hazardLightCmd);
        fun(reverseLightCmd);
        fun(hornCmd);
        fun(highLowBeamLightCmd);
        fun(dayTimeRunLightCmd);
        fun(sidePositionLightCmd);
        fun(windShieldWashCmd);
        fun(windShieldWiperCmd);
        fun(rearViewMirrorCmd);
        fun(parkFinishRemind);
        fun(leftFrontWindowCmd);
        fun(rightFrontWindowCmd);
        fun(leftRearWindowCmd);
        fun(rightRearWindowCmd);
        fun(topWindowCmd);
        fun(turnLightState);
        fun(hazardLightState);
        fun(brakeLightCmd);
        fun(brakeLightValid);
        fun(hazardLightValid);
        fun(highLowBeamLighValid);
        fun(hornValid);
        fun(leftFrontWindowValid);
        fun(leftRearWindowValid);
        fun(rearViewMirrorValid);
        fun(rightFrontWindowValid);
        fun(rightRearWindowValid);
        fun(topWindowValid);
        fun(turnLightValid);
        fun(windShieldWashValid);
        fun(windShieldWiperValid);
        fun(header);
        fun(doorLockCmd);
        fun(turnLightCmdFrequency);
        fun(brakeLightCmdFrequency);
        fun(hazardLightCmdFrequency);
        fun(doorLockCmdValid);
        fun(seatVibrationCmd);
        fun(safeBeltFastenCmd);
        fun(openDoorWarningFL);
        fun(openDoorWarningFR);
        fun(openDoorWarningRL);
        fun(openDoorWarningRR);
        fun(rearViewMirrorWarningLeft);
        fun(rearViewMirrorWarningRight);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("source", source);
        fun("adsFunction", adsFunction);
        fun("adsWorkStatus", adsWorkStatus);
        fun("moduleId", moduleId);
        fun("turnLightCmd", turnLightCmd);
        fun("hazardLightCmd", hazardLightCmd);
        fun("reverseLightCmd", reverseLightCmd);
        fun("hornCmd", hornCmd);
        fun("highLowBeamLightCmd", highLowBeamLightCmd);
        fun("dayTimeRunLightCmd", dayTimeRunLightCmd);
        fun("sidePositionLightCmd", sidePositionLightCmd);
        fun("windShieldWashCmd", windShieldWashCmd);
        fun("windShieldWiperCmd", windShieldWiperCmd);
        fun("rearViewMirrorCmd", rearViewMirrorCmd);
        fun("parkFinishRemind", parkFinishRemind);
        fun("leftFrontWindowCmd", leftFrontWindowCmd);
        fun("rightFrontWindowCmd", rightFrontWindowCmd);
        fun("leftRearWindowCmd", leftRearWindowCmd);
        fun("rightRearWindowCmd", rightRearWindowCmd);
        fun("topWindowCmd", topWindowCmd);
        fun("turnLightState", turnLightState);
        fun("hazardLightState", hazardLightState);
        fun("brakeLightCmd", brakeLightCmd);
        fun("brakeLightValid", brakeLightValid);
        fun("hazardLightValid", hazardLightValid);
        fun("highLowBeamLighValid", highLowBeamLighValid);
        fun("hornValid", hornValid);
        fun("leftFrontWindowValid", leftFrontWindowValid);
        fun("leftRearWindowValid", leftRearWindowValid);
        fun("rearViewMirrorValid", rearViewMirrorValid);
        fun("rightFrontWindowValid", rightFrontWindowValid);
        fun("rightRearWindowValid", rightRearWindowValid);
        fun("topWindowValid", topWindowValid);
        fun("turnLightValid", turnLightValid);
        fun("windShieldWashValid", windShieldWashValid);
        fun("windShieldWiperValid", windShieldWiperValid);
        fun("header", header);
        fun("doorLockCmd", doorLockCmd);
        fun("turnLightCmdFrequency", turnLightCmdFrequency);
        fun("brakeLightCmdFrequency", brakeLightCmdFrequency);
        fun("hazardLightCmdFrequency", hazardLightCmdFrequency);
        fun("doorLockCmdValid", doorLockCmdValid);
        fun("seatVibrationCmd", seatVibrationCmd);
        fun("safeBeltFastenCmd", safeBeltFastenCmd);
        fun("openDoorWarningFL", openDoorWarningFL);
        fun("openDoorWarningFR", openDoorWarningFR);
        fun("openDoorWarningRL", openDoorWarningRL);
        fun("openDoorWarningRR", openDoorWarningRR);
        fun("rearViewMirrorWarningLeft", rearViewMirrorWarningLeft);
        fun("rearViewMirrorWarningRight", rearViewMirrorWarningRight);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("source", source);
        fun("adsFunction", adsFunction);
        fun("adsWorkStatus", adsWorkStatus);
        fun("moduleId", moduleId);
        fun("turnLightCmd", turnLightCmd);
        fun("hazardLightCmd", hazardLightCmd);
        fun("reverseLightCmd", reverseLightCmd);
        fun("hornCmd", hornCmd);
        fun("highLowBeamLightCmd", highLowBeamLightCmd);
        fun("dayTimeRunLightCmd", dayTimeRunLightCmd);
        fun("sidePositionLightCmd", sidePositionLightCmd);
        fun("windShieldWashCmd", windShieldWashCmd);
        fun("windShieldWiperCmd", windShieldWiperCmd);
        fun("rearViewMirrorCmd", rearViewMirrorCmd);
        fun("parkFinishRemind", parkFinishRemind);
        fun("leftFrontWindowCmd", leftFrontWindowCmd);
        fun("rightFrontWindowCmd", rightFrontWindowCmd);
        fun("leftRearWindowCmd", leftRearWindowCmd);
        fun("rightRearWindowCmd", rightRearWindowCmd);
        fun("topWindowCmd", topWindowCmd);
        fun("turnLightState", turnLightState);
        fun("hazardLightState", hazardLightState);
        fun("brakeLightCmd", brakeLightCmd);
        fun("brakeLightValid", brakeLightValid);
        fun("hazardLightValid", hazardLightValid);
        fun("highLowBeamLighValid", highLowBeamLighValid);
        fun("hornValid", hornValid);
        fun("leftFrontWindowValid", leftFrontWindowValid);
        fun("leftRearWindowValid", leftRearWindowValid);
        fun("rearViewMirrorValid", rearViewMirrorValid);
        fun("rightFrontWindowValid", rightFrontWindowValid);
        fun("rightRearWindowValid", rightRearWindowValid);
        fun("topWindowValid", topWindowValid);
        fun("turnLightValid", turnLightValid);
        fun("windShieldWashValid", windShieldWashValid);
        fun("windShieldWiperValid", windShieldWiperValid);
        fun("header", header);
        fun("doorLockCmd", doorLockCmd);
        fun("turnLightCmdFrequency", turnLightCmdFrequency);
        fun("brakeLightCmdFrequency", brakeLightCmdFrequency);
        fun("hazardLightCmdFrequency", hazardLightCmdFrequency);
        fun("doorLockCmdValid", doorLockCmdValid);
        fun("seatVibrationCmd", seatVibrationCmd);
        fun("safeBeltFastenCmd", safeBeltFastenCmd);
        fun("openDoorWarningFL", openDoorWarningFL);
        fun("openDoorWarningFR", openDoorWarningFR);
        fun("openDoorWarningRL", openDoorWarningRL);
        fun("openDoorWarningRR", openDoorWarningRR);
        fun("rearViewMirrorWarningLeft", rearViewMirrorWarningLeft);
        fun("rearViewMirrorWarningRight", rearViewMirrorWarningRight);
    }

    bool operator==(const ::ara::body::BodyCommandMsg& t) const
    {
        return (source == t.source) && (adsFunction == t.adsFunction) && (adsWorkStatus == t.adsWorkStatus) && (moduleId == t.moduleId) && (turnLightCmd == t.turnLightCmd) && (hazardLightCmd == t.hazardLightCmd) && (reverseLightCmd == t.reverseLightCmd) && (hornCmd == t.hornCmd) && (highLowBeamLightCmd == t.highLowBeamLightCmd) && (dayTimeRunLightCmd == t.dayTimeRunLightCmd) && (sidePositionLightCmd == t.sidePositionLightCmd) && (windShieldWashCmd == t.windShieldWashCmd) && (windShieldWiperCmd == t.windShieldWiperCmd) && (rearViewMirrorCmd == t.rearViewMirrorCmd) && (parkFinishRemind == t.parkFinishRemind) && (leftFrontWindowCmd == t.leftFrontWindowCmd) && (rightFrontWindowCmd == t.rightFrontWindowCmd) && (leftRearWindowCmd == t.leftRearWindowCmd) && (rightRearWindowCmd == t.rightRearWindowCmd) && (topWindowCmd == t.topWindowCmd) && (turnLightState == t.turnLightState) && (hazardLightState == t.hazardLightState) && (brakeLightCmd == t.brakeLightCmd) && (brakeLightValid == t.brakeLightValid) && (hazardLightValid == t.hazardLightValid) && (highLowBeamLighValid == t.highLowBeamLighValid) && (hornValid == t.hornValid) && (leftFrontWindowValid == t.leftFrontWindowValid) && (leftRearWindowValid == t.leftRearWindowValid) && (rearViewMirrorValid == t.rearViewMirrorValid) && (rightFrontWindowValid == t.rightFrontWindowValid) && (rightRearWindowValid == t.rightRearWindowValid) && (topWindowValid == t.topWindowValid) && (turnLightValid == t.turnLightValid) && (windShieldWashValid == t.windShieldWashValid) && (windShieldWiperValid == t.windShieldWiperValid) && (header == t.header) && (doorLockCmd == t.doorLockCmd) && (turnLightCmdFrequency == t.turnLightCmdFrequency) && (brakeLightCmdFrequency == t.brakeLightCmdFrequency) && (hazardLightCmdFrequency == t.hazardLightCmdFrequency) && (doorLockCmdValid == t.doorLockCmdValid) && (seatVibrationCmd == t.seatVibrationCmd) && (safeBeltFastenCmd == t.safeBeltFastenCmd) && (openDoorWarningFL == t.openDoorWarningFL) && (openDoorWarningFR == t.openDoorWarningFR) && (openDoorWarningRL == t.openDoorWarningRL) && (openDoorWarningRR == t.openDoorWarningRR) && (rearViewMirrorWarningLeft == t.rearViewMirrorWarningLeft) && (rearViewMirrorWarningRight == t.rearViewMirrorWarningRight);
    }
};
} // namespace body
} // namespace ara


#endif // ARA_BODY_IMPL_TYPE_BODYCOMMANDMSG_H
