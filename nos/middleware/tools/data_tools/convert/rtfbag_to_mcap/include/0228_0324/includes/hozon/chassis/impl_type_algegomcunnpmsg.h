/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGEGOMCUNNPMSG_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGEGOMCUNNPMSG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "impl_type_float.h"

namespace hozon {
namespace chassis {
struct AlgEgoMcuNNPMsg {
    ::UInt8 NNP_Active_OnOffSet;
    ::UInt8 Lanechangeinfor;
    ::UInt8 Lanechangedirection;
    ::UInt8 LCAudioPlay;
    ::UInt8 Lcsndrequest;
    ::UInt8 DCLCAudioplay;
    ::UInt8 LaneChangeWarning;
    ::UInt8 LightRequest;
    ::UInt8 LaneChangePendingAlert ;
    ::UInt8 NNP_LightRemind;
    ::UInt8 lateralCtrtakeover;
    ::UInt8 NNP_Scenarios_AudioPlay;
    ::UInt8 NNP_Scenarios;
    ::UInt8 NNP_RINO_Status;
    ::UInt8 PayModeConfirmFeedBack;
    ::UInt8 SpdAdaptComfirmFeedback;
    ::UInt8 TSR_SpeedSign;
    ::UInt8 ALC_mode;
    ::UInt8 NNP_indx_HDMapLocationNavi_u8;
    ::UInt8 NNP_indx_CrrntLaneCond_u8;
    ::UInt32 NNP_d_Distance2OnRamp_sg;
    ::UInt32 NNP_d_Distance2DownRamp_sg;
    ::UInt32 NNP_d_DistanceIntoODD_sg;
    ::UInt32 NNP_d_DistanceOutofODD_sg;
    ::Float NNP_d_CrrntLaneWidth_sg;
    ::Float NNP_crv_CrrntLaneCurve_sg;
    ::Float NNP_rad_CrrntLaneHead_sg;
    ::UInt8 NNP_is_NNPMRMFlf_bl;
    ::UInt8 NNP_is_NNPMRMDoneFlf_bl;
    ::UInt8 NNP_is_NNPEMFlf_bl;
    ::UInt8 NNP_is_NNPEMDoneFlf_bl;
    ::UInt8 NNP_indx_NNPSoftwareFault_u8;
    ::UInt8 HighBeamReqSt ;
    ::UInt8 LowBeamReqSt ;
    ::UInt8 LowHighBeamReqSt ;
    ::UInt8 HazardLampReqSt ;
    ::UInt8 HornReqSt ;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(NNP_Active_OnOffSet);
        fun(Lanechangeinfor);
        fun(Lanechangedirection);
        fun(LCAudioPlay);
        fun(Lcsndrequest);
        fun(DCLCAudioplay);
        fun(LaneChangeWarning);
        fun(LightRequest);
        fun(LaneChangePendingAlert );
        fun(NNP_LightRemind);
        fun(lateralCtrtakeover);
        fun(NNP_Scenarios_AudioPlay);
        fun(NNP_Scenarios);
        fun(NNP_RINO_Status);
        fun(PayModeConfirmFeedBack);
        fun(SpdAdaptComfirmFeedback);
        fun(TSR_SpeedSign);
        fun(ALC_mode);
        fun(NNP_indx_HDMapLocationNavi_u8);
        fun(NNP_indx_CrrntLaneCond_u8);
        fun(NNP_d_Distance2OnRamp_sg);
        fun(NNP_d_Distance2DownRamp_sg);
        fun(NNP_d_DistanceIntoODD_sg);
        fun(NNP_d_DistanceOutofODD_sg);
        fun(NNP_d_CrrntLaneWidth_sg);
        fun(NNP_crv_CrrntLaneCurve_sg);
        fun(NNP_rad_CrrntLaneHead_sg);
        fun(NNP_is_NNPMRMFlf_bl);
        fun(NNP_is_NNPMRMDoneFlf_bl);
        fun(NNP_is_NNPEMFlf_bl);
        fun(NNP_is_NNPEMDoneFlf_bl);
        fun(NNP_indx_NNPSoftwareFault_u8);
        fun(HighBeamReqSt );
        fun(LowBeamReqSt );
        fun(LowHighBeamReqSt );
        fun(HazardLampReqSt );
        fun(HornReqSt );
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(NNP_Active_OnOffSet);
        fun(Lanechangeinfor);
        fun(Lanechangedirection);
        fun(LCAudioPlay);
        fun(Lcsndrequest);
        fun(DCLCAudioplay);
        fun(LaneChangeWarning);
        fun(LightRequest);
        fun(LaneChangePendingAlert );
        fun(NNP_LightRemind);
        fun(lateralCtrtakeover);
        fun(NNP_Scenarios_AudioPlay);
        fun(NNP_Scenarios);
        fun(NNP_RINO_Status);
        fun(PayModeConfirmFeedBack);
        fun(SpdAdaptComfirmFeedback);
        fun(TSR_SpeedSign);
        fun(ALC_mode);
        fun(NNP_indx_HDMapLocationNavi_u8);
        fun(NNP_indx_CrrntLaneCond_u8);
        fun(NNP_d_Distance2OnRamp_sg);
        fun(NNP_d_Distance2DownRamp_sg);
        fun(NNP_d_DistanceIntoODD_sg);
        fun(NNP_d_DistanceOutofODD_sg);
        fun(NNP_d_CrrntLaneWidth_sg);
        fun(NNP_crv_CrrntLaneCurve_sg);
        fun(NNP_rad_CrrntLaneHead_sg);
        fun(NNP_is_NNPMRMFlf_bl);
        fun(NNP_is_NNPMRMDoneFlf_bl);
        fun(NNP_is_NNPEMFlf_bl);
        fun(NNP_is_NNPEMDoneFlf_bl);
        fun(NNP_indx_NNPSoftwareFault_u8);
        fun(HighBeamReqSt );
        fun(LowBeamReqSt );
        fun(LowHighBeamReqSt );
        fun(HazardLampReqSt );
        fun(HornReqSt );
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("NNP_Active_OnOffSet", NNP_Active_OnOffSet);
        fun("Lanechangeinfor", Lanechangeinfor);
        fun("Lanechangedirection", Lanechangedirection);
        fun("LCAudioPlay", LCAudioPlay);
        fun("Lcsndrequest", Lcsndrequest);
        fun("DCLCAudioplay", DCLCAudioplay);
        fun("LaneChangeWarning", LaneChangeWarning);
        fun("LightRequest", LightRequest);
        fun("LaneChangePendingAlert ", LaneChangePendingAlert );
        fun("NNP_LightRemind", NNP_LightRemind);
        fun("lateralCtrtakeover", lateralCtrtakeover);
        fun("NNP_Scenarios_AudioPlay", NNP_Scenarios_AudioPlay);
        fun("NNP_Scenarios", NNP_Scenarios);
        fun("NNP_RINO_Status", NNP_RINO_Status);
        fun("PayModeConfirmFeedBack", PayModeConfirmFeedBack);
        fun("SpdAdaptComfirmFeedback", SpdAdaptComfirmFeedback);
        fun("TSR_SpeedSign", TSR_SpeedSign);
        fun("ALC_mode", ALC_mode);
        fun("NNP_indx_HDMapLocationNavi_u8", NNP_indx_HDMapLocationNavi_u8);
        fun("NNP_indx_CrrntLaneCond_u8", NNP_indx_CrrntLaneCond_u8);
        fun("NNP_d_Distance2OnRamp_sg", NNP_d_Distance2OnRamp_sg);
        fun("NNP_d_Distance2DownRamp_sg", NNP_d_Distance2DownRamp_sg);
        fun("NNP_d_DistanceIntoODD_sg", NNP_d_DistanceIntoODD_sg);
        fun("NNP_d_DistanceOutofODD_sg", NNP_d_DistanceOutofODD_sg);
        fun("NNP_d_CrrntLaneWidth_sg", NNP_d_CrrntLaneWidth_sg);
        fun("NNP_crv_CrrntLaneCurve_sg", NNP_crv_CrrntLaneCurve_sg);
        fun("NNP_rad_CrrntLaneHead_sg", NNP_rad_CrrntLaneHead_sg);
        fun("NNP_is_NNPMRMFlf_bl", NNP_is_NNPMRMFlf_bl);
        fun("NNP_is_NNPMRMDoneFlf_bl", NNP_is_NNPMRMDoneFlf_bl);
        fun("NNP_is_NNPEMFlf_bl", NNP_is_NNPEMFlf_bl);
        fun("NNP_is_NNPEMDoneFlf_bl", NNP_is_NNPEMDoneFlf_bl);
        fun("NNP_indx_NNPSoftwareFault_u8", NNP_indx_NNPSoftwareFault_u8);
        fun("HighBeamReqSt ", HighBeamReqSt );
        fun("LowBeamReqSt ", LowBeamReqSt );
        fun("LowHighBeamReqSt ", LowHighBeamReqSt );
        fun("HazardLampReqSt ", HazardLampReqSt );
        fun("HornReqSt ", HornReqSt );
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("NNP_Active_OnOffSet", NNP_Active_OnOffSet);
        fun("Lanechangeinfor", Lanechangeinfor);
        fun("Lanechangedirection", Lanechangedirection);
        fun("LCAudioPlay", LCAudioPlay);
        fun("Lcsndrequest", Lcsndrequest);
        fun("DCLCAudioplay", DCLCAudioplay);
        fun("LaneChangeWarning", LaneChangeWarning);
        fun("LightRequest", LightRequest);
        fun("LaneChangePendingAlert ", LaneChangePendingAlert );
        fun("NNP_LightRemind", NNP_LightRemind);
        fun("lateralCtrtakeover", lateralCtrtakeover);
        fun("NNP_Scenarios_AudioPlay", NNP_Scenarios_AudioPlay);
        fun("NNP_Scenarios", NNP_Scenarios);
        fun("NNP_RINO_Status", NNP_RINO_Status);
        fun("PayModeConfirmFeedBack", PayModeConfirmFeedBack);
        fun("SpdAdaptComfirmFeedback", SpdAdaptComfirmFeedback);
        fun("TSR_SpeedSign", TSR_SpeedSign);
        fun("ALC_mode", ALC_mode);
        fun("NNP_indx_HDMapLocationNavi_u8", NNP_indx_HDMapLocationNavi_u8);
        fun("NNP_indx_CrrntLaneCond_u8", NNP_indx_CrrntLaneCond_u8);
        fun("NNP_d_Distance2OnRamp_sg", NNP_d_Distance2OnRamp_sg);
        fun("NNP_d_Distance2DownRamp_sg", NNP_d_Distance2DownRamp_sg);
        fun("NNP_d_DistanceIntoODD_sg", NNP_d_DistanceIntoODD_sg);
        fun("NNP_d_DistanceOutofODD_sg", NNP_d_DistanceOutofODD_sg);
        fun("NNP_d_CrrntLaneWidth_sg", NNP_d_CrrntLaneWidth_sg);
        fun("NNP_crv_CrrntLaneCurve_sg", NNP_crv_CrrntLaneCurve_sg);
        fun("NNP_rad_CrrntLaneHead_sg", NNP_rad_CrrntLaneHead_sg);
        fun("NNP_is_NNPMRMFlf_bl", NNP_is_NNPMRMFlf_bl);
        fun("NNP_is_NNPMRMDoneFlf_bl", NNP_is_NNPMRMDoneFlf_bl);
        fun("NNP_is_NNPEMFlf_bl", NNP_is_NNPEMFlf_bl);
        fun("NNP_is_NNPEMDoneFlf_bl", NNP_is_NNPEMDoneFlf_bl);
        fun("NNP_indx_NNPSoftwareFault_u8", NNP_indx_NNPSoftwareFault_u8);
        fun("HighBeamReqSt ", HighBeamReqSt );
        fun("LowBeamReqSt ", LowBeamReqSt );
        fun("LowHighBeamReqSt ", LowHighBeamReqSt );
        fun("HazardLampReqSt ", HazardLampReqSt );
        fun("HornReqSt ", HornReqSt );
    }

    bool operator==(const ::hozon::chassis::AlgEgoMcuNNPMsg& t) const
    {
        return (NNP_Active_OnOffSet == t.NNP_Active_OnOffSet) && (Lanechangeinfor == t.Lanechangeinfor) && (Lanechangedirection == t.Lanechangedirection) && (LCAudioPlay == t.LCAudioPlay) && (Lcsndrequest == t.Lcsndrequest) && (DCLCAudioplay == t.DCLCAudioplay) && (LaneChangeWarning == t.LaneChangeWarning) && (LightRequest == t.LightRequest) && (LaneChangePendingAlert  == t.LaneChangePendingAlert ) && (NNP_LightRemind == t.NNP_LightRemind) && (lateralCtrtakeover == t.lateralCtrtakeover) && (NNP_Scenarios_AudioPlay == t.NNP_Scenarios_AudioPlay) && (NNP_Scenarios == t.NNP_Scenarios) && (NNP_RINO_Status == t.NNP_RINO_Status) && (PayModeConfirmFeedBack == t.PayModeConfirmFeedBack) && (SpdAdaptComfirmFeedback == t.SpdAdaptComfirmFeedback) && (TSR_SpeedSign == t.TSR_SpeedSign) && (ALC_mode == t.ALC_mode) && (NNP_indx_HDMapLocationNavi_u8 == t.NNP_indx_HDMapLocationNavi_u8) && (NNP_indx_CrrntLaneCond_u8 == t.NNP_indx_CrrntLaneCond_u8) && (NNP_d_Distance2OnRamp_sg == t.NNP_d_Distance2OnRamp_sg) && (NNP_d_Distance2DownRamp_sg == t.NNP_d_Distance2DownRamp_sg) && (NNP_d_DistanceIntoODD_sg == t.NNP_d_DistanceIntoODD_sg) && (NNP_d_DistanceOutofODD_sg == t.NNP_d_DistanceOutofODD_sg) && (fabs(static_cast<double>(NNP_d_CrrntLaneWidth_sg - t.NNP_d_CrrntLaneWidth_sg)) < DBL_EPSILON) && (fabs(static_cast<double>(NNP_crv_CrrntLaneCurve_sg - t.NNP_crv_CrrntLaneCurve_sg)) < DBL_EPSILON) && (fabs(static_cast<double>(NNP_rad_CrrntLaneHead_sg - t.NNP_rad_CrrntLaneHead_sg)) < DBL_EPSILON) && (NNP_is_NNPMRMFlf_bl == t.NNP_is_NNPMRMFlf_bl) && (NNP_is_NNPMRMDoneFlf_bl == t.NNP_is_NNPMRMDoneFlf_bl) && (NNP_is_NNPEMFlf_bl == t.NNP_is_NNPEMFlf_bl) && (NNP_is_NNPEMDoneFlf_bl == t.NNP_is_NNPEMDoneFlf_bl) && (NNP_indx_NNPSoftwareFault_u8 == t.NNP_indx_NNPSoftwareFault_u8) && (HighBeamReqSt  == t.HighBeamReqSt ) && (LowBeamReqSt  == t.LowBeamReqSt ) && (LowHighBeamReqSt  == t.LowHighBeamReqSt ) && (HazardLampReqSt  == t.HazardLampReqSt ) && (HornReqSt  == t.HornReqSt );
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGEGOMCUNNPMSG_H
