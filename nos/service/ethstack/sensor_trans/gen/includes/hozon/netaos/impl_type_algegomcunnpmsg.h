/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file impl_type_algegomcunnpmsg.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGEGOMCUNNPMSG_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGEGOMCUNNPMSG_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgEgoMcuNNPMsg {
    std::uint8_t NNP_Active_OnOffSet;
    std::uint8_t Lanechangeinfor;
    std::uint8_t Lanechangedirection;
    std::uint8_t LCAudioPlay;
    std::uint8_t Lcsndrequest;
    std::uint8_t DCLCAudioplay;
    std::uint8_t LaneChangeWarning;
    std::uint8_t LightRequest;
    std::uint8_t LaneChangePendingAlert;
    std::uint8_t NNP_LightRemind;
    std::uint8_t lateralCtrtakeover;
    std::uint8_t NNP_Scenarios_AudioPlay;
    std::uint8_t NNP_Scenarios;
    std::uint8_t NNP_RINO_Status;
    std::uint8_t PayModeConfirmFeedBack;
    std::uint8_t SpdAdaptComfirmFeedback;
    std::uint8_t TSR_SpeedSign;
    std::uint8_t ALC_mode;
    std::uint8_t NNP_indx_HDMapLocationNavi_u8;
    std::uint8_t NNP_indx_CrrntLaneCond_u8;
    std::uint32_t NNP_d_Distance2OnRamp_sg;
    std::uint32_t NNP_d_Distance2DownRamp_sg;
    std::uint32_t NNP_d_DistanceIntoODD_sg;
    std::uint32_t NNP_d_DistanceOutofODD_sg;
    float NNP_d_CrrntLaneWidth_sg;
    float NNP_crv_CrrntLaneCurve_sg;
    float NNP_rad_CrrntLaneHead_sg;
    std::uint8_t NNP_is_NNPMRMFlf_bl;
    std::uint8_t NNP_is_NNPMRMDoneFlf_bl;
    std::uint8_t NNP_is_NNPEMFlf_bl;
    std::uint8_t NNP_is_NNPEMDoneFlf_bl;
    std::uint8_t NNP_indx_NNPSoftwareFault_u8;
    std::uint8_t HighBeamReqSt;
    std::uint8_t LowBeamReqSt;
    std::uint8_t LowHighBeamReqSt;
    std::uint8_t HazardLampReqSt;
    std::uint8_t HornReqSt;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgEgoMcuNNPMsg,NNP_Active_OnOffSet,Lanechangeinfor,Lanechangedirection,LCAudioPlay,Lcsndrequest,DCLCAudioplay,LaneChangeWarning,LightRequest,LaneChangePendingAlert,NNP_LightRemind,lateralCtrtakeover,NNP_Scenarios_AudioPlay,NNP_Scenarios,NNP_RINO_Status,PayModeConfirmFeedBack,SpdAdaptComfirmFeedback,TSR_SpeedSign,ALC_mode,NNP_indx_HDMapLocationNavi_u8,NNP_indx_CrrntLaneCond_u8,NNP_d_Distance2OnRamp_sg,NNP_d_Distance2DownRamp_sg,NNP_d_DistanceIntoODD_sg,NNP_d_DistanceOutofODD_sg,NNP_d_CrrntLaneWidth_sg,NNP_crv_CrrntLaneCurve_sg,NNP_rad_CrrntLaneHead_sg,NNP_is_NNPMRMFlf_bl,NNP_is_NNPMRMDoneFlf_bl,NNP_is_NNPEMFlf_bl,NNP_is_NNPEMDoneFlf_bl,NNP_indx_NNPSoftwareFault_u8,HighBeamReqSt,LowBeamReqSt,LowHighBeamReqSt,HazardLampReqSt,HornReqSt);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGEGOMCUNNPMSG_H_
/* EOF */