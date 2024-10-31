/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X136_H
#define HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X136_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc2mcu {
struct Adas_MSG_0x136 {
    ::UInt8 ADCS12_DCLCAudioplay;
    ::UInt8 ADCS12_PilotAudioPlay;
    ::UInt8 ADCS12_HandOffAudioplay;
    ::UInt8 ADCS12_NNPCancelAudioPlay;
    ::UInt8 ADCS12_MRMAudioPlay;
    ::UInt8 ADCS12_NNP_ResumeAudioplay;
    ::UInt8 ADCS12_Lcsndrequest;
    ::UInt8 ADCS12_Lcsndrconfirm;
    ::UInt8 ADCS12_DriveOffBroadcast;
    ::UInt8 ADCS12_EyeOffBroadcast;
    ::UInt8 ADCS12_longitudDisableInfo;
    ::UInt8 ADCS12_SpeedAdaptComfirm;
    ::UInt8 ADCS12_PayModeConfirm;
    ::UInt8 ADCS12_longitudCtrlSysInfo;
    ::UInt8 ADCS12_ADAS_DriveOffinhibition;
    ::UInt8 ADCS12_P2N_State_Reminder;
    ::UInt8 ADCS12_NNP_State_Reminder;
    ::UInt8 ADCS12_NNP_LightRemind;
    ::UInt8 ADCS12_LaneChangePendingAlert;
    ::UInt8 ADCS12_NPilotSysInfo;
    ::UInt8 ADCS12_LCAudioPlay;
    ::UInt8 ADCS12_NNP_Scenarios_AudioPlay;
    ::UInt8 ADCS12_NNP_RINO;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADCS12_DCLCAudioplay);
        fun(ADCS12_PilotAudioPlay);
        fun(ADCS12_HandOffAudioplay);
        fun(ADCS12_NNPCancelAudioPlay);
        fun(ADCS12_MRMAudioPlay);
        fun(ADCS12_NNP_ResumeAudioplay);
        fun(ADCS12_Lcsndrequest);
        fun(ADCS12_Lcsndrconfirm);
        fun(ADCS12_DriveOffBroadcast);
        fun(ADCS12_EyeOffBroadcast);
        fun(ADCS12_longitudDisableInfo);
        fun(ADCS12_SpeedAdaptComfirm);
        fun(ADCS12_PayModeConfirm);
        fun(ADCS12_longitudCtrlSysInfo);
        fun(ADCS12_ADAS_DriveOffinhibition);
        fun(ADCS12_P2N_State_Reminder);
        fun(ADCS12_NNP_State_Reminder);
        fun(ADCS12_NNP_LightRemind);
        fun(ADCS12_LaneChangePendingAlert);
        fun(ADCS12_NPilotSysInfo);
        fun(ADCS12_LCAudioPlay);
        fun(ADCS12_NNP_Scenarios_AudioPlay);
        fun(ADCS12_NNP_RINO);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADCS12_DCLCAudioplay);
        fun(ADCS12_PilotAudioPlay);
        fun(ADCS12_HandOffAudioplay);
        fun(ADCS12_NNPCancelAudioPlay);
        fun(ADCS12_MRMAudioPlay);
        fun(ADCS12_NNP_ResumeAudioplay);
        fun(ADCS12_Lcsndrequest);
        fun(ADCS12_Lcsndrconfirm);
        fun(ADCS12_DriveOffBroadcast);
        fun(ADCS12_EyeOffBroadcast);
        fun(ADCS12_longitudDisableInfo);
        fun(ADCS12_SpeedAdaptComfirm);
        fun(ADCS12_PayModeConfirm);
        fun(ADCS12_longitudCtrlSysInfo);
        fun(ADCS12_ADAS_DriveOffinhibition);
        fun(ADCS12_P2N_State_Reminder);
        fun(ADCS12_NNP_State_Reminder);
        fun(ADCS12_NNP_LightRemind);
        fun(ADCS12_LaneChangePendingAlert);
        fun(ADCS12_NPilotSysInfo);
        fun(ADCS12_LCAudioPlay);
        fun(ADCS12_NNP_Scenarios_AudioPlay);
        fun(ADCS12_NNP_RINO);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADCS12_DCLCAudioplay", ADCS12_DCLCAudioplay);
        fun("ADCS12_PilotAudioPlay", ADCS12_PilotAudioPlay);
        fun("ADCS12_HandOffAudioplay", ADCS12_HandOffAudioplay);
        fun("ADCS12_NNPCancelAudioPlay", ADCS12_NNPCancelAudioPlay);
        fun("ADCS12_MRMAudioPlay", ADCS12_MRMAudioPlay);
        fun("ADCS12_NNP_ResumeAudioplay", ADCS12_NNP_ResumeAudioplay);
        fun("ADCS12_Lcsndrequest", ADCS12_Lcsndrequest);
        fun("ADCS12_Lcsndrconfirm", ADCS12_Lcsndrconfirm);
        fun("ADCS12_DriveOffBroadcast", ADCS12_DriveOffBroadcast);
        fun("ADCS12_EyeOffBroadcast", ADCS12_EyeOffBroadcast);
        fun("ADCS12_longitudDisableInfo", ADCS12_longitudDisableInfo);
        fun("ADCS12_SpeedAdaptComfirm", ADCS12_SpeedAdaptComfirm);
        fun("ADCS12_PayModeConfirm", ADCS12_PayModeConfirm);
        fun("ADCS12_longitudCtrlSysInfo", ADCS12_longitudCtrlSysInfo);
        fun("ADCS12_ADAS_DriveOffinhibition", ADCS12_ADAS_DriveOffinhibition);
        fun("ADCS12_P2N_State_Reminder", ADCS12_P2N_State_Reminder);
        fun("ADCS12_NNP_State_Reminder", ADCS12_NNP_State_Reminder);
        fun("ADCS12_NNP_LightRemind", ADCS12_NNP_LightRemind);
        fun("ADCS12_LaneChangePendingAlert", ADCS12_LaneChangePendingAlert);
        fun("ADCS12_NPilotSysInfo", ADCS12_NPilotSysInfo);
        fun("ADCS12_LCAudioPlay", ADCS12_LCAudioPlay);
        fun("ADCS12_NNP_Scenarios_AudioPlay", ADCS12_NNP_Scenarios_AudioPlay);
        fun("ADCS12_NNP_RINO", ADCS12_NNP_RINO);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADCS12_DCLCAudioplay", ADCS12_DCLCAudioplay);
        fun("ADCS12_PilotAudioPlay", ADCS12_PilotAudioPlay);
        fun("ADCS12_HandOffAudioplay", ADCS12_HandOffAudioplay);
        fun("ADCS12_NNPCancelAudioPlay", ADCS12_NNPCancelAudioPlay);
        fun("ADCS12_MRMAudioPlay", ADCS12_MRMAudioPlay);
        fun("ADCS12_NNP_ResumeAudioplay", ADCS12_NNP_ResumeAudioplay);
        fun("ADCS12_Lcsndrequest", ADCS12_Lcsndrequest);
        fun("ADCS12_Lcsndrconfirm", ADCS12_Lcsndrconfirm);
        fun("ADCS12_DriveOffBroadcast", ADCS12_DriveOffBroadcast);
        fun("ADCS12_EyeOffBroadcast", ADCS12_EyeOffBroadcast);
        fun("ADCS12_longitudDisableInfo", ADCS12_longitudDisableInfo);
        fun("ADCS12_SpeedAdaptComfirm", ADCS12_SpeedAdaptComfirm);
        fun("ADCS12_PayModeConfirm", ADCS12_PayModeConfirm);
        fun("ADCS12_longitudCtrlSysInfo", ADCS12_longitudCtrlSysInfo);
        fun("ADCS12_ADAS_DriveOffinhibition", ADCS12_ADAS_DriveOffinhibition);
        fun("ADCS12_P2N_State_Reminder", ADCS12_P2N_State_Reminder);
        fun("ADCS12_NNP_State_Reminder", ADCS12_NNP_State_Reminder);
        fun("ADCS12_NNP_LightRemind", ADCS12_NNP_LightRemind);
        fun("ADCS12_LaneChangePendingAlert", ADCS12_LaneChangePendingAlert);
        fun("ADCS12_NPilotSysInfo", ADCS12_NPilotSysInfo);
        fun("ADCS12_LCAudioPlay", ADCS12_LCAudioPlay);
        fun("ADCS12_NNP_Scenarios_AudioPlay", ADCS12_NNP_Scenarios_AudioPlay);
        fun("ADCS12_NNP_RINO", ADCS12_NNP_RINO);
    }

    bool operator==(const ::hozon::soc2mcu::Adas_MSG_0x136& t) const
    {
        return (ADCS12_DCLCAudioplay == t.ADCS12_DCLCAudioplay) && (ADCS12_PilotAudioPlay == t.ADCS12_PilotAudioPlay) && (ADCS12_HandOffAudioplay == t.ADCS12_HandOffAudioplay) && (ADCS12_NNPCancelAudioPlay == t.ADCS12_NNPCancelAudioPlay) && (ADCS12_MRMAudioPlay == t.ADCS12_MRMAudioPlay) && (ADCS12_NNP_ResumeAudioplay == t.ADCS12_NNP_ResumeAudioplay) && (ADCS12_Lcsndrequest == t.ADCS12_Lcsndrequest) && (ADCS12_Lcsndrconfirm == t.ADCS12_Lcsndrconfirm) && (ADCS12_DriveOffBroadcast == t.ADCS12_DriveOffBroadcast) && (ADCS12_EyeOffBroadcast == t.ADCS12_EyeOffBroadcast) && (ADCS12_longitudDisableInfo == t.ADCS12_longitudDisableInfo) && (ADCS12_SpeedAdaptComfirm == t.ADCS12_SpeedAdaptComfirm) && (ADCS12_PayModeConfirm == t.ADCS12_PayModeConfirm) && (ADCS12_longitudCtrlSysInfo == t.ADCS12_longitudCtrlSysInfo) && (ADCS12_ADAS_DriveOffinhibition == t.ADCS12_ADAS_DriveOffinhibition) && (ADCS12_P2N_State_Reminder == t.ADCS12_P2N_State_Reminder) && (ADCS12_NNP_State_Reminder == t.ADCS12_NNP_State_Reminder) && (ADCS12_NNP_LightRemind == t.ADCS12_NNP_LightRemind) && (ADCS12_LaneChangePendingAlert == t.ADCS12_LaneChangePendingAlert) && (ADCS12_NPilotSysInfo == t.ADCS12_NPilotSysInfo) && (ADCS12_LCAudioPlay == t.ADCS12_LCAudioPlay) && (ADCS12_NNP_Scenarios_AudioPlay == t.ADCS12_NNP_Scenarios_AudioPlay) && (ADCS12_NNP_RINO == t.ADCS12_NNP_RINO);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X136_H
