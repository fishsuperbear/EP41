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
 * @file impl_type_algavmpdsinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGAVMPDSINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGAVMPDSINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgAvmPdsInfo {
    std::uint16_t CDCS3_Xcoordinate;
    std::uint16_t CDCS3_Ycoordinate;
    std::uint8_t CDCS3_XGestureSpd;
    std::uint8_t CDCS3_YGestureSpd;
    std::uint8_t CDCS3_TouchEvtTyp;
    std::uint8_t CDCS10_WorK_status;
    std::uint8_t CDCS11_LowSpeedBSDSw;
    std::uint8_t CDCS11_AVM_3DRotate;
    std::uint8_t CDCS11_AVM_Turnlightstart;
    std::uint8_t CDCS11_AVM_Steeringstart;
    std::uint8_t CDCS11_AVM_Magnifiedview;
    std::uint8_t CDCS11_AVM_MODdetection;
    std::uint8_t CDCS11_GM_OnOffSet;
    std::uint8_t CDCS11_AVM_auxiliaryline;
    std::uint8_t CDCS11_AVM_Raderdisplay;
    std::uint8_t CDCS11_AVM_Transparentchassis;
    std::uint8_t CDCS11_GM_VidedIn;
    std::uint8_t CDCS11_APA_FunctionMode;
    std::uint8_t CDCS11_TrackreverseSW;
    std::uint8_t CDCS11_SetPathwaySW;
    std::uint8_t CDCS11_AVMSw;
    std::uint8_t CDCS11_RPASw;
    std::uint8_t CDCS11_SetSw;
    std::uint8_t CDCS11_location_sw;
    std::uint8_t CDCS11_Pathrefresh;
    std::uint8_t CDCS11_HPAGuideSW;
    std::uint8_t CDCS11_HPAPathwaytoCloudSW;
    std::uint8_t CDCS11_PathlearningSw;
    std::uint8_t CDCS11_PASw;
    std::uint8_t CDCS11_PA_Recover;
    std::uint8_t CDCS11_SelectSlotID;
    std::uint8_t CDCS11_ParkingInReq;
    std::uint8_t CDCS11_ParkingOutReq;
    std::uint8_t CDCS11_AVM_ViewMode;
    std::uint8_t CDCS11_RPA_FunctionMode;
    std::uint8_t CDCS11_AVM_Initialization;
    std::uint8_t CDCS11_AVM_SpeView;
    std::uint8_t CDCS11_AVM_bodySync;
    std::uint8_t CDCS11_AVM_licenseplateabbr;
    std::uint8_t CDCS11_AVM_licenseplateArea;
    std::uint8_t CDCS11_AVM_LicensePlateNO1;
    std::uint8_t CDCS11_AVM_LicensePlateNO2;
    std::uint8_t CDCS11_AVM_LicensePlateNO3;
    std::uint8_t CDCS11_AVM_LicensePlateNO4;
    std::uint8_t CDCS11_AVM_LicensePlateNO5;
    std::uint8_t CDCS11_AVM_LicensePlateNO6;
    std::uint8_t CDCS11_AVM_LicensePlateNO7;
    std::uint8_t CDCS11_AVM_LicensePlateNO8;
    std::uint8_t CDCS11_AVM_DiaphaneitySw;
    std::uint8_t CDCS11_MOD_Sound_Sw;
    std::uint8_t CDCS11_MOD_BSD_Sw;
    std::uint8_t CDCS11_PA_FRS_OnOffSet;
    std::uint8_t CDCS11_PA_3D_OnOffSet;
    std::uint8_t CDCS11_PA_measure_OnOffSet;
    std::uint8_t CDCS11_MODSw;
    std::uint8_t CDCS11_HPASw;
    std::uint8_t CDCS11_DeleteMapID;
    std::uint8_t CDCS11_UploadMapID;
    std::uint8_t CDCS11_HPApreparkingSw;
    std::uint8_t CDCS11_HPAPathwaytoCloud_WorkSts;
    std::uint8_t CDCS11_ParkingoutSlot_Type;
    std::uint8_t CDCS11_ParkingoutSlot_Dire;
    std::uint8_t CDCS11_OptionalSlot_Type;
    std::uint8_t CDCS11_OptionalSlot_Dire;
    float CDCS11_OptionalSlotAngle;
    std::uint16_t CDCS11_OptionalSlotCoor_P0_X;
    std::uint16_t CDCS11_OptionalSlotCoor_P0_Y;
    std::uint16_t CDCS11_OptionalSlotCoor_P1_X;
    std::uint16_t CDCS11_OptionalSlotCoor_P1_Y;
    std::uint16_t CDCS11_OptionalSlotCoor_P2_X;
    std::uint16_t CDCS11_OptionalSlotCoor_P2_Y;
    std::uint16_t CDCS11_OptionalSlotCoor_P3_X;
    std::uint16_t CDCS11_OptionalSlotCoor_P3_Y;
    std::uint8_t DDCU1_FLDoorAjar;
    std::uint8_t DDCU1_RLDoorAjar;
    std::uint8_t PDCU1_FRDoorAjar;
    std::uint8_t PDCU1_RRDoorAjar;
    std::uint8_t BTM1_SecuritySts;
    std::uint8_t BTM1_PhoneBatSts;
    std::uint8_t BTM1_RemoteIntModSel;
    std::uint8_t BTM1_SelectSlotID;
    std::uint8_t BTM1_Retry;
    std::uint8_t BTM1_Fixslot;
    std::uint8_t BTM1_parkingoutSlot_Dire;
    std::uint8_t BTM1_parkingoutSlotType;
    std::uint8_t BTM1_Recover;
    std::uint8_t BTM1_ParkingReq;
    std::uint8_t BTM1_FunctionMode;
    std::uint8_t BTM1_OptionalSlot_Dire;
    std::uint8_t BTM1_OptionalSlotType;
    std::uint8_t BTM1_RollingCounter;
    std::uint8_t BTM1_RemoteParkReq;
    std::uint8_t BTM1_Movectrl;
    std::uint8_t BTM1_Checksum;
    float BTM1_OptionalSlotAngle;
    std::uint16_t BTM1_OptionalSlotCoor_P0_X;
    std::uint16_t BTM1_OptionalSlotCoor_P0_Y;
    std::uint16_t BTM1_OptionalSlotCoor_P1_X;
    std::uint16_t BTM1_OptionalSlotCoor_P1_Y;
    std::uint16_t BTM1_OptionalSlotCoor_P2_X;
    std::uint16_t BTM1_OptionalSlotCoor_P2_Y;
    std::uint16_t BTM1_OptionalSlotCoor_P3_X;
    std::uint16_t BTM1_OptionalSlotCoor_P3_Y;
    std::uint8_t TBOX2_AVMOnReq;
    std::uint8_t TBOX2_RemoteParkStartPressReqSts;
    std::uint8_t TBOX2_RemoteHPP;
    std::uint8_t TBOX2_AVMviewReq;
    std::uint8_t TBOX2_RemoteParkReq;
    std::uint8_t TBOX2_SecuritySts;
    std::uint8_t TBOX2_ConnectSts;
    std::uint8_t TBOX2_PhoneInCar;
    std::uint8_t TBOX2_RemoteParkAbortReq;
    std::uint8_t TBOX2_RemoteParkStartPressReq;
    std::uint8_t TBOX2_RemoteParkOutModSel;
    std::uint8_t TBOX2_PhoneBatSts;
    std::uint8_t TBOX2_RemoteIntModSel;
    std::uint8_t TBOX2_RemoteSw;
    std::uint8_t TBOX2_RemoteCtrModSel;
    std::uint8_t TBOX2_RemoteCtrSw;
    std::uint8_t TBOX2_Remotepick;
    std::uint8_t TBOX2_NNSsuspend;
    std::uint8_t TBOX2_RemoteVhpower;
    std::uint8_t TBOX2_NRPReq;
    std::uint8_t TBOX2_SDMapReq;
    std::uint8_t TBOX2_NNSSW;
    std::uint8_t TBOX2_RollingCounter;
    std::uint8_t TBOX2_Checksum;
    std::uint8_t TBOX2_GMOnreq;
    std::uint8_t TBOX2_360viewReq;
    std::uint8_t TBOX2_360SpecView;
    std::uint8_t TBOX2_PA_PauseRecover_Req;
    std::uint8_t CDCS11_tryHPP;
    std::uint8_t CDCS11_AVM_Diaphaneity;
    std::uint8_t CDCS11_HPA_ONPath_ID1;
    std::uint8_t CDCS11_HPA_ONPath_ID2;
    std::uint8_t CDCS11_HPA_ONPath_ID3;
    std::uint8_t CDCS11_HPA_ONPath_ID4;
    std::uint8_t BDCS1_PowerManageMode;
    std::uint8_t BTM2_ConnectSts;
    std::uint8_t BTM2_Key_Position;
    std::uint8_t BTM3_RKECmd;
    std::uint16_t BMS3_StateOfCharge;
    std::uint8_t BMS3_Charge_State;
    std::uint8_t BDCS13_RLS_RQ_WiperSPD;
    std::uint8_t DDCU1_FL_WinPos;
    std::uint8_t DDCU1_RL_WinPos;
    std::uint8_t DDCU1_MirrorFoldSt;
    std::uint8_t PDCU1_FR_WinPos;
    std::uint8_t PDCU1_RR_WinPos;
    std::uint8_t CDCS11_HPA_Select_ID;
    std::uint8_t BDCS1_PowerMode;
    std::uint8_t BDCS1_TurnLightSW;
    std::uint8_t BDCS1_TrunkLockSts;
    std::uint8_t CDCS11_AVM_Powertype;
    std::uint8_t ACU1_CrashStatus;
    std::uint8_t TBOX2_HPA_ONPath_ID;
    std::uint8_t TBOX2_HPA_PickPath_ID;
    std::uint8_t CDCS11_learnpath_St;
    std::uint8_t CDCS11_FindCarAvmReq;
    std::uint8_t CDCS15_UserSceneMode;
    std::uint8_t CDCS11_DCWposition;
    std::uint8_t CDCS15_Storage_St;
    std::uint8_t CDCS15_USB_St;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgAvmPdsInfo,CDCS3_Xcoordinate,CDCS3_Ycoordinate,CDCS3_XGestureSpd,CDCS3_YGestureSpd,CDCS3_TouchEvtTyp,CDCS10_WorK_status,CDCS11_LowSpeedBSDSw,CDCS11_AVM_3DRotate,CDCS11_AVM_Turnlightstart,CDCS11_AVM_Steeringstart,CDCS11_AVM_Magnifiedview,CDCS11_AVM_MODdetection,CDCS11_GM_OnOffSet,CDCS11_AVM_auxiliaryline,CDCS11_AVM_Raderdisplay,CDCS11_AVM_Transparentchassis,CDCS11_GM_VidedIn,CDCS11_APA_FunctionMode,CDCS11_TrackreverseSW,CDCS11_SetPathwaySW,CDCS11_AVMSw,CDCS11_RPASw,CDCS11_SetSw,CDCS11_location_sw,CDCS11_Pathrefresh,CDCS11_HPAGuideSW,CDCS11_HPAPathwaytoCloudSW,CDCS11_PathlearningSw,CDCS11_PASw,CDCS11_PA_Recover,CDCS11_SelectSlotID,CDCS11_ParkingInReq,CDCS11_ParkingOutReq,CDCS11_AVM_ViewMode,CDCS11_RPA_FunctionMode,CDCS11_AVM_Initialization,CDCS11_AVM_SpeView,CDCS11_AVM_bodySync,CDCS11_AVM_licenseplateabbr,CDCS11_AVM_licenseplateArea,CDCS11_AVM_LicensePlateNO1,CDCS11_AVM_LicensePlateNO2,CDCS11_AVM_LicensePlateNO3,CDCS11_AVM_LicensePlateNO4,CDCS11_AVM_LicensePlateNO5,CDCS11_AVM_LicensePlateNO6,CDCS11_AVM_LicensePlateNO7,CDCS11_AVM_LicensePlateNO8,CDCS11_AVM_DiaphaneitySw,CDCS11_MOD_Sound_Sw,CDCS11_MOD_BSD_Sw,CDCS11_PA_FRS_OnOffSet,CDCS11_PA_3D_OnOffSet,CDCS11_PA_measure_OnOffSet,CDCS11_MODSw,CDCS11_HPASw,CDCS11_DeleteMapID,CDCS11_UploadMapID,CDCS11_HPApreparkingSw,CDCS11_HPAPathwaytoCloud_WorkSts,CDCS11_ParkingoutSlot_Type,CDCS11_ParkingoutSlot_Dire,CDCS11_OptionalSlot_Type,CDCS11_OptionalSlot_Dire,CDCS11_OptionalSlotAngle,CDCS11_OptionalSlotCoor_P0_X,CDCS11_OptionalSlotCoor_P0_Y,CDCS11_OptionalSlotCoor_P1_X,CDCS11_OptionalSlotCoor_P1_Y,CDCS11_OptionalSlotCoor_P2_X,CDCS11_OptionalSlotCoor_P2_Y,CDCS11_OptionalSlotCoor_P3_X,CDCS11_OptionalSlotCoor_P3_Y,DDCU1_FLDoorAjar,DDCU1_RLDoorAjar,PDCU1_FRDoorAjar,PDCU1_RRDoorAjar,BTM1_SecuritySts,BTM1_PhoneBatSts,BTM1_RemoteIntModSel,BTM1_SelectSlotID,BTM1_Retry,BTM1_Fixslot,BTM1_parkingoutSlot_Dire,BTM1_parkingoutSlotType,BTM1_Recover,BTM1_ParkingReq,BTM1_FunctionMode,BTM1_OptionalSlot_Dire,BTM1_OptionalSlotType,BTM1_RollingCounter,BTM1_RemoteParkReq,BTM1_Movectrl,BTM1_Checksum,BTM1_OptionalSlotAngle,BTM1_OptionalSlotCoor_P0_X,BTM1_OptionalSlotCoor_P0_Y,BTM1_OptionalSlotCoor_P1_X,BTM1_OptionalSlotCoor_P1_Y,BTM1_OptionalSlotCoor_P2_X,BTM1_OptionalSlotCoor_P2_Y,BTM1_OptionalSlotCoor_P3_X,BTM1_OptionalSlotCoor_P3_Y,TBOX2_AVMOnReq,TBOX2_RemoteParkStartPressReqSts,TBOX2_RemoteHPP,TBOX2_AVMviewReq,TBOX2_RemoteParkReq,TBOX2_SecuritySts,TBOX2_ConnectSts,TBOX2_PhoneInCar,TBOX2_RemoteParkAbortReq,TBOX2_RemoteParkStartPressReq,TBOX2_RemoteParkOutModSel,TBOX2_PhoneBatSts,TBOX2_RemoteIntModSel,TBOX2_RemoteSw,TBOX2_RemoteCtrModSel,TBOX2_RemoteCtrSw,TBOX2_Remotepick,TBOX2_NNSsuspend,TBOX2_RemoteVhpower,TBOX2_NRPReq,TBOX2_SDMapReq,TBOX2_NNSSW,TBOX2_RollingCounter,TBOX2_Checksum,TBOX2_GMOnreq,TBOX2_360viewReq,TBOX2_360SpecView,TBOX2_PA_PauseRecover_Req,CDCS11_tryHPP,CDCS11_AVM_Diaphaneity,CDCS11_HPA_ONPath_ID1,CDCS11_HPA_ONPath_ID2,CDCS11_HPA_ONPath_ID3,CDCS11_HPA_ONPath_ID4,BDCS1_PowerManageMode,BTM2_ConnectSts,BTM2_Key_Position,BTM3_RKECmd,BMS3_StateOfCharge,BMS3_Charge_State,BDCS13_RLS_RQ_WiperSPD,DDCU1_FL_WinPos,DDCU1_RL_WinPos,DDCU1_MirrorFoldSt,PDCU1_FR_WinPos,PDCU1_RR_WinPos,CDCS11_HPA_Select_ID,BDCS1_PowerMode,BDCS1_TurnLightSW,BDCS1_TrunkLockSts,CDCS11_AVM_Powertype,ACU1_CrashStatus,TBOX2_HPA_ONPath_ID,TBOX2_HPA_PickPath_ID,CDCS11_learnpath_St,CDCS11_FindCarAvmReq,CDCS15_UserSceneMode,CDCS11_DCWposition,CDCS15_Storage_St,CDCS15_USB_St);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGAVMPDSINFO_H_
/* EOF */