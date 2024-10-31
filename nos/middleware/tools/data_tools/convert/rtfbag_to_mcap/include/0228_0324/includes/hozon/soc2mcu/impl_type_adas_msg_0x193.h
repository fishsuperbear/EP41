/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X193_H
#define HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X193_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc2mcu {
struct Adas_MSG_0x193 {
    ::UInt8 ADCS8_LDWState;
    ::UInt8 ADCS8_LDWWarnState;
    ::UInt8 ADCS8_LDWMode;
    ::UInt8 ADCS8_ELKMode;
    ::UInt8 ADCS8_LDPELKWarnState;
    ::UInt8 ADCS8_lateralCtrHandsEyesOffWarning;
    ::UInt8 ADCS8_MissingExitAlert;
    ::UInt8 ADCS8_LKSSetspeed;
    ::UInt8 ADCS8_ACCState;
    ::UInt8 ADCS8_lateralCtrtakeover;
    ::UInt8 ADCS8_LateralCtrHandOffReleasewarning;
    ::UInt8 ADCS8_LDPMode;
    ::UInt8 ADCS8_LDPELKtakeoverWarningstate;
    ::UInt8 ADCS8_longitudCtrlSetSpeed;
    ::UInt8 ADCS8_NNP_Scenarios;
    ::UInt8 ADCS8_longitudCtrlTakeOverReq;
    ::UInt8 ADCS8_longitudCtrlSetDistance;
    ::UInt8 ADCS8_longitudCtrlTargetValidity;
    ::UInt8 ADCS8_ACCSystemFaultStatus;
    ::UInt8 ADCS8_HornReq;
    ::UInt8 ADCS8_DoorUnlock;
    ::UInt8 ADCS8_ACCOnOffState;
    ::UInt8 ADCS8_NNP_SOAAudioplay;
    ::UInt8 ADCS8_TSR_SLFStatefeedback;
    ::UInt8 ADCS8_VoiceNotifFrequency_mode;
    ::UInt8 ADCS8_NPilot_SystemFaultStatus;
    ::UInt8 ADCS8_NPilot_SysState;
    ::UInt8 ADCS8_NNP_AutoOnOffSet;
    ::UInt8 ADCS8_NNP_SystemFaultStatus;
    ::UInt8 ADCS8_NNPSysState;
    ::UInt8 ADCS8_Lanechangeinfor;
    ::UInt8 ADCS8_DCLCSysState;
    ::UInt8 ADCS8_LaneChangeSystemFaultStatus;
    ::UInt8 ADCS8_LaneChangeWarning;
    ::UInt8 ADCS8_ADAS_DriveOffPossilbe;
    ::UInt8 ADCS8_DriveOffinhibitionObjType;
    ::UInt8 ADCS8_VoiceMode;
    ::UInt8 ADCS8_ALC_mode;
    ::UInt8 ADCS8_NNP_State;
    ::UInt8 ADCS8_Lanechangedirection;
    ::UInt8 ADCS8_ADSDriving_mode;
    ::UInt8 ADCS8_NNP_MRM_status;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADCS8_LDWState);
        fun(ADCS8_LDWWarnState);
        fun(ADCS8_LDWMode);
        fun(ADCS8_ELKMode);
        fun(ADCS8_LDPELKWarnState);
        fun(ADCS8_lateralCtrHandsEyesOffWarning);
        fun(ADCS8_MissingExitAlert);
        fun(ADCS8_LKSSetspeed);
        fun(ADCS8_ACCState);
        fun(ADCS8_lateralCtrtakeover);
        fun(ADCS8_LateralCtrHandOffReleasewarning);
        fun(ADCS8_LDPMode);
        fun(ADCS8_LDPELKtakeoverWarningstate);
        fun(ADCS8_longitudCtrlSetSpeed);
        fun(ADCS8_NNP_Scenarios);
        fun(ADCS8_longitudCtrlTakeOverReq);
        fun(ADCS8_longitudCtrlSetDistance);
        fun(ADCS8_longitudCtrlTargetValidity);
        fun(ADCS8_ACCSystemFaultStatus);
        fun(ADCS8_HornReq);
        fun(ADCS8_DoorUnlock);
        fun(ADCS8_ACCOnOffState);
        fun(ADCS8_NNP_SOAAudioplay);
        fun(ADCS8_TSR_SLFStatefeedback);
        fun(ADCS8_VoiceNotifFrequency_mode);
        fun(ADCS8_NPilot_SystemFaultStatus);
        fun(ADCS8_NPilot_SysState);
        fun(ADCS8_NNP_AutoOnOffSet);
        fun(ADCS8_NNP_SystemFaultStatus);
        fun(ADCS8_NNPSysState);
        fun(ADCS8_Lanechangeinfor);
        fun(ADCS8_DCLCSysState);
        fun(ADCS8_LaneChangeSystemFaultStatus);
        fun(ADCS8_LaneChangeWarning);
        fun(ADCS8_ADAS_DriveOffPossilbe);
        fun(ADCS8_DriveOffinhibitionObjType);
        fun(ADCS8_VoiceMode);
        fun(ADCS8_ALC_mode);
        fun(ADCS8_NNP_State);
        fun(ADCS8_Lanechangedirection);
        fun(ADCS8_ADSDriving_mode);
        fun(ADCS8_NNP_MRM_status);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADCS8_LDWState);
        fun(ADCS8_LDWWarnState);
        fun(ADCS8_LDWMode);
        fun(ADCS8_ELKMode);
        fun(ADCS8_LDPELKWarnState);
        fun(ADCS8_lateralCtrHandsEyesOffWarning);
        fun(ADCS8_MissingExitAlert);
        fun(ADCS8_LKSSetspeed);
        fun(ADCS8_ACCState);
        fun(ADCS8_lateralCtrtakeover);
        fun(ADCS8_LateralCtrHandOffReleasewarning);
        fun(ADCS8_LDPMode);
        fun(ADCS8_LDPELKtakeoverWarningstate);
        fun(ADCS8_longitudCtrlSetSpeed);
        fun(ADCS8_NNP_Scenarios);
        fun(ADCS8_longitudCtrlTakeOverReq);
        fun(ADCS8_longitudCtrlSetDistance);
        fun(ADCS8_longitudCtrlTargetValidity);
        fun(ADCS8_ACCSystemFaultStatus);
        fun(ADCS8_HornReq);
        fun(ADCS8_DoorUnlock);
        fun(ADCS8_ACCOnOffState);
        fun(ADCS8_NNP_SOAAudioplay);
        fun(ADCS8_TSR_SLFStatefeedback);
        fun(ADCS8_VoiceNotifFrequency_mode);
        fun(ADCS8_NPilot_SystemFaultStatus);
        fun(ADCS8_NPilot_SysState);
        fun(ADCS8_NNP_AutoOnOffSet);
        fun(ADCS8_NNP_SystemFaultStatus);
        fun(ADCS8_NNPSysState);
        fun(ADCS8_Lanechangeinfor);
        fun(ADCS8_DCLCSysState);
        fun(ADCS8_LaneChangeSystemFaultStatus);
        fun(ADCS8_LaneChangeWarning);
        fun(ADCS8_ADAS_DriveOffPossilbe);
        fun(ADCS8_DriveOffinhibitionObjType);
        fun(ADCS8_VoiceMode);
        fun(ADCS8_ALC_mode);
        fun(ADCS8_NNP_State);
        fun(ADCS8_Lanechangedirection);
        fun(ADCS8_ADSDriving_mode);
        fun(ADCS8_NNP_MRM_status);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADCS8_LDWState", ADCS8_LDWState);
        fun("ADCS8_LDWWarnState", ADCS8_LDWWarnState);
        fun("ADCS8_LDWMode", ADCS8_LDWMode);
        fun("ADCS8_ELKMode", ADCS8_ELKMode);
        fun("ADCS8_LDPELKWarnState", ADCS8_LDPELKWarnState);
        fun("ADCS8_lateralCtrHandsEyesOffWarning", ADCS8_lateralCtrHandsEyesOffWarning);
        fun("ADCS8_MissingExitAlert", ADCS8_MissingExitAlert);
        fun("ADCS8_LKSSetspeed", ADCS8_LKSSetspeed);
        fun("ADCS8_ACCState", ADCS8_ACCState);
        fun("ADCS8_lateralCtrtakeover", ADCS8_lateralCtrtakeover);
        fun("ADCS8_LateralCtrHandOffReleasewarning", ADCS8_LateralCtrHandOffReleasewarning);
        fun("ADCS8_LDPMode", ADCS8_LDPMode);
        fun("ADCS8_LDPELKtakeoverWarningstate", ADCS8_LDPELKtakeoverWarningstate);
        fun("ADCS8_longitudCtrlSetSpeed", ADCS8_longitudCtrlSetSpeed);
        fun("ADCS8_NNP_Scenarios", ADCS8_NNP_Scenarios);
        fun("ADCS8_longitudCtrlTakeOverReq", ADCS8_longitudCtrlTakeOverReq);
        fun("ADCS8_longitudCtrlSetDistance", ADCS8_longitudCtrlSetDistance);
        fun("ADCS8_longitudCtrlTargetValidity", ADCS8_longitudCtrlTargetValidity);
        fun("ADCS8_ACCSystemFaultStatus", ADCS8_ACCSystemFaultStatus);
        fun("ADCS8_HornReq", ADCS8_HornReq);
        fun("ADCS8_DoorUnlock", ADCS8_DoorUnlock);
        fun("ADCS8_ACCOnOffState", ADCS8_ACCOnOffState);
        fun("ADCS8_NNP_SOAAudioplay", ADCS8_NNP_SOAAudioplay);
        fun("ADCS8_TSR_SLFStatefeedback", ADCS8_TSR_SLFStatefeedback);
        fun("ADCS8_VoiceNotifFrequency_mode", ADCS8_VoiceNotifFrequency_mode);
        fun("ADCS8_NPilot_SystemFaultStatus", ADCS8_NPilot_SystemFaultStatus);
        fun("ADCS8_NPilot_SysState", ADCS8_NPilot_SysState);
        fun("ADCS8_NNP_AutoOnOffSet", ADCS8_NNP_AutoOnOffSet);
        fun("ADCS8_NNP_SystemFaultStatus", ADCS8_NNP_SystemFaultStatus);
        fun("ADCS8_NNPSysState", ADCS8_NNPSysState);
        fun("ADCS8_Lanechangeinfor", ADCS8_Lanechangeinfor);
        fun("ADCS8_DCLCSysState", ADCS8_DCLCSysState);
        fun("ADCS8_LaneChangeSystemFaultStatus", ADCS8_LaneChangeSystemFaultStatus);
        fun("ADCS8_LaneChangeWarning", ADCS8_LaneChangeWarning);
        fun("ADCS8_ADAS_DriveOffPossilbe", ADCS8_ADAS_DriveOffPossilbe);
        fun("ADCS8_DriveOffinhibitionObjType", ADCS8_DriveOffinhibitionObjType);
        fun("ADCS8_VoiceMode", ADCS8_VoiceMode);
        fun("ADCS8_ALC_mode", ADCS8_ALC_mode);
        fun("ADCS8_NNP_State", ADCS8_NNP_State);
        fun("ADCS8_Lanechangedirection", ADCS8_Lanechangedirection);
        fun("ADCS8_ADSDriving_mode", ADCS8_ADSDriving_mode);
        fun("ADCS8_NNP_MRM_status", ADCS8_NNP_MRM_status);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADCS8_LDWState", ADCS8_LDWState);
        fun("ADCS8_LDWWarnState", ADCS8_LDWWarnState);
        fun("ADCS8_LDWMode", ADCS8_LDWMode);
        fun("ADCS8_ELKMode", ADCS8_ELKMode);
        fun("ADCS8_LDPELKWarnState", ADCS8_LDPELKWarnState);
        fun("ADCS8_lateralCtrHandsEyesOffWarning", ADCS8_lateralCtrHandsEyesOffWarning);
        fun("ADCS8_MissingExitAlert", ADCS8_MissingExitAlert);
        fun("ADCS8_LKSSetspeed", ADCS8_LKSSetspeed);
        fun("ADCS8_ACCState", ADCS8_ACCState);
        fun("ADCS8_lateralCtrtakeover", ADCS8_lateralCtrtakeover);
        fun("ADCS8_LateralCtrHandOffReleasewarning", ADCS8_LateralCtrHandOffReleasewarning);
        fun("ADCS8_LDPMode", ADCS8_LDPMode);
        fun("ADCS8_LDPELKtakeoverWarningstate", ADCS8_LDPELKtakeoverWarningstate);
        fun("ADCS8_longitudCtrlSetSpeed", ADCS8_longitudCtrlSetSpeed);
        fun("ADCS8_NNP_Scenarios", ADCS8_NNP_Scenarios);
        fun("ADCS8_longitudCtrlTakeOverReq", ADCS8_longitudCtrlTakeOverReq);
        fun("ADCS8_longitudCtrlSetDistance", ADCS8_longitudCtrlSetDistance);
        fun("ADCS8_longitudCtrlTargetValidity", ADCS8_longitudCtrlTargetValidity);
        fun("ADCS8_ACCSystemFaultStatus", ADCS8_ACCSystemFaultStatus);
        fun("ADCS8_HornReq", ADCS8_HornReq);
        fun("ADCS8_DoorUnlock", ADCS8_DoorUnlock);
        fun("ADCS8_ACCOnOffState", ADCS8_ACCOnOffState);
        fun("ADCS8_NNP_SOAAudioplay", ADCS8_NNP_SOAAudioplay);
        fun("ADCS8_TSR_SLFStatefeedback", ADCS8_TSR_SLFStatefeedback);
        fun("ADCS8_VoiceNotifFrequency_mode", ADCS8_VoiceNotifFrequency_mode);
        fun("ADCS8_NPilot_SystemFaultStatus", ADCS8_NPilot_SystemFaultStatus);
        fun("ADCS8_NPilot_SysState", ADCS8_NPilot_SysState);
        fun("ADCS8_NNP_AutoOnOffSet", ADCS8_NNP_AutoOnOffSet);
        fun("ADCS8_NNP_SystemFaultStatus", ADCS8_NNP_SystemFaultStatus);
        fun("ADCS8_NNPSysState", ADCS8_NNPSysState);
        fun("ADCS8_Lanechangeinfor", ADCS8_Lanechangeinfor);
        fun("ADCS8_DCLCSysState", ADCS8_DCLCSysState);
        fun("ADCS8_LaneChangeSystemFaultStatus", ADCS8_LaneChangeSystemFaultStatus);
        fun("ADCS8_LaneChangeWarning", ADCS8_LaneChangeWarning);
        fun("ADCS8_ADAS_DriveOffPossilbe", ADCS8_ADAS_DriveOffPossilbe);
        fun("ADCS8_DriveOffinhibitionObjType", ADCS8_DriveOffinhibitionObjType);
        fun("ADCS8_VoiceMode", ADCS8_VoiceMode);
        fun("ADCS8_ALC_mode", ADCS8_ALC_mode);
        fun("ADCS8_NNP_State", ADCS8_NNP_State);
        fun("ADCS8_Lanechangedirection", ADCS8_Lanechangedirection);
        fun("ADCS8_ADSDriving_mode", ADCS8_ADSDriving_mode);
        fun("ADCS8_NNP_MRM_status", ADCS8_NNP_MRM_status);
    }

    bool operator==(const ::hozon::soc2mcu::Adas_MSG_0x193& t) const
    {
        return (ADCS8_LDWState == t.ADCS8_LDWState) && (ADCS8_LDWWarnState == t.ADCS8_LDWWarnState) && (ADCS8_LDWMode == t.ADCS8_LDWMode) && (ADCS8_ELKMode == t.ADCS8_ELKMode) && (ADCS8_LDPELKWarnState == t.ADCS8_LDPELKWarnState) && (ADCS8_lateralCtrHandsEyesOffWarning == t.ADCS8_lateralCtrHandsEyesOffWarning) && (ADCS8_MissingExitAlert == t.ADCS8_MissingExitAlert) && (ADCS8_LKSSetspeed == t.ADCS8_LKSSetspeed) && (ADCS8_ACCState == t.ADCS8_ACCState) && (ADCS8_lateralCtrtakeover == t.ADCS8_lateralCtrtakeover) && (ADCS8_LateralCtrHandOffReleasewarning == t.ADCS8_LateralCtrHandOffReleasewarning) && (ADCS8_LDPMode == t.ADCS8_LDPMode) && (ADCS8_LDPELKtakeoverWarningstate == t.ADCS8_LDPELKtakeoverWarningstate) && (ADCS8_longitudCtrlSetSpeed == t.ADCS8_longitudCtrlSetSpeed) && (ADCS8_NNP_Scenarios == t.ADCS8_NNP_Scenarios) && (ADCS8_longitudCtrlTakeOverReq == t.ADCS8_longitudCtrlTakeOverReq) && (ADCS8_longitudCtrlSetDistance == t.ADCS8_longitudCtrlSetDistance) && (ADCS8_longitudCtrlTargetValidity == t.ADCS8_longitudCtrlTargetValidity) && (ADCS8_ACCSystemFaultStatus == t.ADCS8_ACCSystemFaultStatus) && (ADCS8_HornReq == t.ADCS8_HornReq) && (ADCS8_DoorUnlock == t.ADCS8_DoorUnlock) && (ADCS8_ACCOnOffState == t.ADCS8_ACCOnOffState) && (ADCS8_NNP_SOAAudioplay == t.ADCS8_NNP_SOAAudioplay) && (ADCS8_TSR_SLFStatefeedback == t.ADCS8_TSR_SLFStatefeedback) && (ADCS8_VoiceNotifFrequency_mode == t.ADCS8_VoiceNotifFrequency_mode) && (ADCS8_NPilot_SystemFaultStatus == t.ADCS8_NPilot_SystemFaultStatus) && (ADCS8_NPilot_SysState == t.ADCS8_NPilot_SysState) && (ADCS8_NNP_AutoOnOffSet == t.ADCS8_NNP_AutoOnOffSet) && (ADCS8_NNP_SystemFaultStatus == t.ADCS8_NNP_SystemFaultStatus) && (ADCS8_NNPSysState == t.ADCS8_NNPSysState) && (ADCS8_Lanechangeinfor == t.ADCS8_Lanechangeinfor) && (ADCS8_DCLCSysState == t.ADCS8_DCLCSysState) && (ADCS8_LaneChangeSystemFaultStatus == t.ADCS8_LaneChangeSystemFaultStatus) && (ADCS8_LaneChangeWarning == t.ADCS8_LaneChangeWarning) && (ADCS8_ADAS_DriveOffPossilbe == t.ADCS8_ADAS_DriveOffPossilbe) && (ADCS8_DriveOffinhibitionObjType == t.ADCS8_DriveOffinhibitionObjType) && (ADCS8_VoiceMode == t.ADCS8_VoiceMode) && (ADCS8_ALC_mode == t.ADCS8_ALC_mode) && (ADCS8_NNP_State == t.ADCS8_NNP_State) && (ADCS8_Lanechangedirection == t.ADCS8_Lanechangedirection) && (ADCS8_ADSDriving_mode == t.ADCS8_ADSDriving_mode) && (ADCS8_NNP_MRM_status == t.ADCS8_NNP_MRM_status);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X193_H
