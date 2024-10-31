/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_RB_NVM_ASW_REMMBERSTATE_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_RB_NVM_ASW_REMMBERSTATE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_rb_NvM_ASW_RemmberState {
    ::UInt8 LDWMode;
    ::UInt8 LKSSetspeed;
    ::UInt8 LDPMode;
    ::UInt8 VoiceMode;
    ::UInt8 ACCOnOffState;
    ::UInt8 DCLCSysState;
    ::UInt8 NNPState;
    ::UInt8 AutoOnOffSet;
    ::UInt8 ALCMode;
    ::UInt8 ADSDrivingMode;
    ::UInt8 TSR_SLFStatefeedback;
    ::UInt8 RCTA_OnOffSet;
    ::UInt8 FCTA_OnOffSet;
    ::UInt8 DOW_OnOffSet;
    ::UInt8 RCW_OnOffSet;
    ::UInt8 LCA_OnOffSet;
    ::UInt8 TSR_OnOffSet;
    ::UInt8 TSR_OverspeedOnOffSet;
    ::UInt8 IHBC_OnOffSet;
    ::Float CtrlYawrateOffset;
    ::Float CtrlYawOffset;
    ::Float CtrlAxOffset;
    ::Float CtrlSteerOffset;
    ::Float CtrlAccelDeadzone;
    ::UInt8 ADCS8_FCWSensitiveLevel;
    ::UInt8 AEB_OnOffSet;
    ::UInt8 FCW_OnOffSet;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(LDWMode);
        fun(LKSSetspeed);
        fun(LDPMode);
        fun(VoiceMode);
        fun(ACCOnOffState);
        fun(DCLCSysState);
        fun(NNPState);
        fun(AutoOnOffSet);
        fun(ALCMode);
        fun(ADSDrivingMode);
        fun(TSR_SLFStatefeedback);
        fun(RCTA_OnOffSet);
        fun(FCTA_OnOffSet);
        fun(DOW_OnOffSet);
        fun(RCW_OnOffSet);
        fun(LCA_OnOffSet);
        fun(TSR_OnOffSet);
        fun(TSR_OverspeedOnOffSet);
        fun(IHBC_OnOffSet);
        fun(CtrlYawrateOffset);
        fun(CtrlYawOffset);
        fun(CtrlAxOffset);
        fun(CtrlSteerOffset);
        fun(CtrlAccelDeadzone);
        fun(ADCS8_FCWSensitiveLevel);
        fun(AEB_OnOffSet);
        fun(FCW_OnOffSet);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(LDWMode);
        fun(LKSSetspeed);
        fun(LDPMode);
        fun(VoiceMode);
        fun(ACCOnOffState);
        fun(DCLCSysState);
        fun(NNPState);
        fun(AutoOnOffSet);
        fun(ALCMode);
        fun(ADSDrivingMode);
        fun(TSR_SLFStatefeedback);
        fun(RCTA_OnOffSet);
        fun(FCTA_OnOffSet);
        fun(DOW_OnOffSet);
        fun(RCW_OnOffSet);
        fun(LCA_OnOffSet);
        fun(TSR_OnOffSet);
        fun(TSR_OverspeedOnOffSet);
        fun(IHBC_OnOffSet);
        fun(CtrlYawrateOffset);
        fun(CtrlYawOffset);
        fun(CtrlAxOffset);
        fun(CtrlSteerOffset);
        fun(CtrlAccelDeadzone);
        fun(ADCS8_FCWSensitiveLevel);
        fun(AEB_OnOffSet);
        fun(FCW_OnOffSet);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("LDWMode", LDWMode);
        fun("LKSSetspeed", LKSSetspeed);
        fun("LDPMode", LDPMode);
        fun("VoiceMode", VoiceMode);
        fun("ACCOnOffState", ACCOnOffState);
        fun("DCLCSysState", DCLCSysState);
        fun("NNPState", NNPState);
        fun("AutoOnOffSet", AutoOnOffSet);
        fun("ALCMode", ALCMode);
        fun("ADSDrivingMode", ADSDrivingMode);
        fun("TSR_SLFStatefeedback", TSR_SLFStatefeedback);
        fun("RCTA_OnOffSet", RCTA_OnOffSet);
        fun("FCTA_OnOffSet", FCTA_OnOffSet);
        fun("DOW_OnOffSet", DOW_OnOffSet);
        fun("RCW_OnOffSet", RCW_OnOffSet);
        fun("LCA_OnOffSet", LCA_OnOffSet);
        fun("TSR_OnOffSet", TSR_OnOffSet);
        fun("TSR_OverspeedOnOffSet", TSR_OverspeedOnOffSet);
        fun("IHBC_OnOffSet", IHBC_OnOffSet);
        fun("CtrlYawrateOffset", CtrlYawrateOffset);
        fun("CtrlYawOffset", CtrlYawOffset);
        fun("CtrlAxOffset", CtrlAxOffset);
        fun("CtrlSteerOffset", CtrlSteerOffset);
        fun("CtrlAccelDeadzone", CtrlAccelDeadzone);
        fun("ADCS8_FCWSensitiveLevel", ADCS8_FCWSensitiveLevel);
        fun("AEB_OnOffSet", AEB_OnOffSet);
        fun("FCW_OnOffSet", FCW_OnOffSet);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("LDWMode", LDWMode);
        fun("LKSSetspeed", LKSSetspeed);
        fun("LDPMode", LDPMode);
        fun("VoiceMode", VoiceMode);
        fun("ACCOnOffState", ACCOnOffState);
        fun("DCLCSysState", DCLCSysState);
        fun("NNPState", NNPState);
        fun("AutoOnOffSet", AutoOnOffSet);
        fun("ALCMode", ALCMode);
        fun("ADSDrivingMode", ADSDrivingMode);
        fun("TSR_SLFStatefeedback", TSR_SLFStatefeedback);
        fun("RCTA_OnOffSet", RCTA_OnOffSet);
        fun("FCTA_OnOffSet", FCTA_OnOffSet);
        fun("DOW_OnOffSet", DOW_OnOffSet);
        fun("RCW_OnOffSet", RCW_OnOffSet);
        fun("LCA_OnOffSet", LCA_OnOffSet);
        fun("TSR_OnOffSet", TSR_OnOffSet);
        fun("TSR_OverspeedOnOffSet", TSR_OverspeedOnOffSet);
        fun("IHBC_OnOffSet", IHBC_OnOffSet);
        fun("CtrlYawrateOffset", CtrlYawrateOffset);
        fun("CtrlYawOffset", CtrlYawOffset);
        fun("CtrlAxOffset", CtrlAxOffset);
        fun("CtrlSteerOffset", CtrlSteerOffset);
        fun("CtrlAccelDeadzone", CtrlAccelDeadzone);
        fun("ADCS8_FCWSensitiveLevel", ADCS8_FCWSensitiveLevel);
        fun("AEB_OnOffSet", AEB_OnOffSet);
        fun("FCW_OnOffSet", FCW_OnOffSet);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_rb_NvM_ASW_RemmberState& t) const
    {
        return (LDWMode == t.LDWMode) && (LKSSetspeed == t.LKSSetspeed) && (LDPMode == t.LDPMode) && (VoiceMode == t.VoiceMode) && (ACCOnOffState == t.ACCOnOffState) && (DCLCSysState == t.DCLCSysState) && (NNPState == t.NNPState) && (AutoOnOffSet == t.AutoOnOffSet) && (ALCMode == t.ALCMode) && (ADSDrivingMode == t.ADSDrivingMode) && (TSR_SLFStatefeedback == t.TSR_SLFStatefeedback) && (RCTA_OnOffSet == t.RCTA_OnOffSet) && (FCTA_OnOffSet == t.FCTA_OnOffSet) && (DOW_OnOffSet == t.DOW_OnOffSet) && (RCW_OnOffSet == t.RCW_OnOffSet) && (LCA_OnOffSet == t.LCA_OnOffSet) && (TSR_OnOffSet == t.TSR_OnOffSet) && (TSR_OverspeedOnOffSet == t.TSR_OverspeedOnOffSet) && (IHBC_OnOffSet == t.IHBC_OnOffSet) && (fabs(static_cast<double>(CtrlYawrateOffset - t.CtrlYawrateOffset)) < DBL_EPSILON) && (fabs(static_cast<double>(CtrlYawOffset - t.CtrlYawOffset)) < DBL_EPSILON) && (fabs(static_cast<double>(CtrlAxOffset - t.CtrlAxOffset)) < DBL_EPSILON) && (fabs(static_cast<double>(CtrlSteerOffset - t.CtrlSteerOffset)) < DBL_EPSILON) && (fabs(static_cast<double>(CtrlAccelDeadzone - t.CtrlAccelDeadzone)) < DBL_EPSILON) && (ADCS8_FCWSensitiveLevel == t.ADCS8_FCWSensitiveLevel) && (AEB_OnOffSet == t.AEB_OnOffSet) && (FCW_OnOffSet == t.FCW_OnOffSet);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_RB_NVM_ASW_REMMBERSTATE_H
