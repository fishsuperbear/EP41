/* *
 * FUNCTION: Define chassis_info data Types
 * Copyright (c) Hozon auto, Ltd. 2019-2022. All rights reserved.
 */

#pragma once

#include "adf/include/data_types/common/types.h"

namespace hozon {
namespace netaos {
namespace adf {

struct AlgVcuInfo {
    uint8_t VCU_ActGearPosition;
    bool VCU_ActGearPosition_Valid;
    uint8_t VCU_Real_ThrottlePosition;
    bool VCU_Real_ThrottlePos_Valid;
    uint8_t VCU5_PwrTrainFailureLevel;
    uint8_t VCU5_READYLightSts;
};

struct AlgSteeringInfo {
    double SteeringAngle;
    bool SteeringAngleValid;
    double SteeringAngleSpeed;
    bool SteeringAngleSpeedValid;
};

struct AlgWheelInfo {
    double ESC_FLWheelSpeed;
    bool ESC_FLWheelSpeedValid;
    uint8_t ESC_FLWheelDirection;
    double ESC_FRWheelSpeed;
    bool ESC_FRWheelSpeedValid;
    uint8_t ESC_FRWheelDirection;
    double ESC_RLWheelSpeed;
    bool ESC_RLWheelSpeedValid;
    uint8_t ESC_RLWheelDirection;
    double ESC_RRWheelSpeed;
    bool ESC_RRWheelSpeedValid;
    uint8_t ESC_RRWheelDirection;
    float ESC_FL_WhlPulCnt;
    float ESC_FR_WhlPulCnt;
    float ESC_RL_WhlPulCnt;
    float ESC_RR_WhlPulCnt;
    bool ESC_FL_WhlPulCntValid;
    bool ESC_FR_WhlPulCntValid;
    bool ESC_RL_WhlPulCntValid;
    bool ESC_RR_WhlPulCntValid;
};

struct AlgEscDrivingInfo {
    double ESC_VehicleSpeed;
    bool ESC_VehicleSpeedValid;
    bool ESC_BrakePedalSwitchStatus;
    bool ESC_BrakePedalSwitchValid;
    float BrkPedVal;
    double VehicleSpdDisplay;
    bool VehicleSpdDisplayValid;
    bool ESC_ApaStandStill;
    double ESC_LongAccValue;
    bool ESC_LongAccValue_Valid;
    bool ESC_LongAccValueOffset_Valid;
    double ESC_LatAccValue;
    bool ESC_LatAccValue_Valid;
    double ESC_YawRate;
    bool ESC_YawRate_Valid;
    uint8_t IDB1_FailedState;
    float IDB3_MasterCylPressure;
    bool IDB5_BrakeDiscTempSts;
    uint8_t IDB5_VehicleStanstill;
    bool IDB5_JerkActive;
    bool IDB5_Jerkfail;
    uint8_t IDB5_ADAS_PrefillSts;
    bool IDB5_ABAavailable;
    bool IDB5_ABPavailable;
    bool IDB5_ABAactive;
    bool IDB5_AEBactive;
    uint8_t IDB5_AEB_Enable;
    uint8_t IDB6_EPBAvailable;
    bool IDB7_ESCActive;
    bool IDB7_ABSActive;
    bool IDB7_HBAActive;
    bool IDB7_HBAFail;
    bool IDB7_TCSActive;
    bool IDB7_TCSDisable;
    bool IDB7_ARPACTIVE;
    bool IDB7_ARPFailure;
};

struct AlgBodyStateInfo {
    uint8_t BCM_FLDrOpn;
    uint8_t BCM_FRDrOpn;
    uint8_t BCM_RLDrOpn;
    uint8_t BCM_RRDrOpn;
    uint8_t BCM_TGOpn;
    bool BCM_HodOpen;
    uint8_t BCM_DrvSeatbeltBucklesta;
    uint8_t BCM_FrontWiperSt;
    uint8_t BCM_FrontWiperWorkSts;
    uint8_t BCM_HighBeamSt;
    uint8_t CS1_HighBeamReqSt;
    uint8_t BCM_LowBeamSt;
    uint8_t HazardLampSt;
    bool BCM_FrontFogLampSt;
    bool BCM_RearFogLampSt;
    bool BCM_LeftTurnLightSt;
    bool BCM_RightTurnLightSt;
    uint8_t BCM_TurnLightSW;
    uint8_t BCM_FrontLampSt;
};

struct AlgCenterConsoleInfo {
    uint8_t TSR_OnOffSet;
    uint8_t TSR_OverspeedOnoffSet;
    uint8_t IHBC_SysSwState;

    uint8_t FactoryReset;
    uint8_t ResetAllSetup;
};

struct AlgParkInfo {
    bool TCSActive;
    bool ABSActive;
    bool ARPActive;
    bool ESCActive;
    uint8_t EPBStatus;
};

struct AlgSWSwitchInfo {
    uint8_t SWSM_A_CruiseSpeed_Add;
    uint8_t SWSM_A_CruiseSpeed_Minus;
    uint8_t SWSM_A_CruiseDistance_Add;
    uint8_t SWSM_A_CruiseDistance_Minus;
};

enum AlgChassisErrorCode {
    NO_ERROR = 0,
    CMD_NOT_IN_PERIOD = 1,
    MANUAL_INTERVENTION = 2,
    CHASSIS_CAN_NOT_IN_PERIOD = 3,
    CHASSIS_ERROR_ON_STEER = 4,
    CHASSIS_ERROR_ON_BRAKE = 5,
    CHASSIS_ERROR_ON_THROTTLE = 6,
    CHASSIS_ERROR_ON_GEAR = 7,
    UNKNOW_ERROR = 8
};

struct AlgWarnningHmiInfo {
    uint8_t ADCS8_VoiceMode;
    uint8_t RCTA_OnOffSet;
    uint8_t FCTA_OnOffSet;
    uint8_t DOW_OnOffSet;
    uint8_t RCW_OnOffSet;
    uint8_t LCA_OnOffSet;
};

struct AlgAvmPdsInfo {
    uint16_t CDCS3_Xcoordinate;
    uint16_t CDCS3_Ycoordinate;
    uint8_t CDCS3_XGestureSpd;
    uint8_t CDCS3_YGestureSpd;
    uint8_t CDCS3_TouchEvtTyp;
    uint8_t CDCS10_WorK_status;
    uint8_t CDCS11_LowSpeedBSDSw;
    uint8_t CDCS11_AVM_3DRotate;
    uint8_t CDCS11_AVM_Turnlightstart;
    uint8_t CDCS11_AVM_Steeringstart;
    uint8_t CDCS11_AVM_Magnifiedview;
    uint8_t CDCS11_AVM_MODdetection;
    uint8_t CDCS11_GM_OnOffSet;
    uint8_t CDCS11_AVM_auxiliaryline;
    uint8_t CDCS11_AVM_Raderdisplay;
    uint8_t CDCS11_AVM_Transparentchassis;
    uint8_t CDCS11_GM_VidedIn;
    uint8_t CDCS11_APA_FunctionMode;
    uint8_t CDCS11_TrackreverseSW;
    uint8_t CDCS11_SetPathwaySW;
    uint8_t CDCS11_AVMSw;
    uint8_t CDCS11_RPASw;
    uint8_t CDCS11_SetSw;
    uint8_t CDCS11_location_sw;
    uint8_t CDCS11_Pathrefresh;
    uint8_t CDCS11_HPAGuideSW;
    uint8_t CDCS11_HPAPathwaytoCloudSW;
    uint8_t CDCS11_PathlearningSw;
    uint8_t CDCS11_PASw;
    uint8_t CDCS11_PA_Recover;
    uint8_t CDCS11_SelectSlotID;
    uint8_t CDCS11_ParkingInReq;
    uint8_t CDCS11_ParkingOutReq;
    uint8_t CDCS11_AVM_ViewMode;
    uint8_t CDCS11_RPA_FunctionMode;
    uint8_t CDCS11_AVM_Initialization;
    uint8_t CDCS11_AVM_SpeView;
    uint8_t CDCS11_AVM_bodySync;
    uint8_t CDCS11_AVM_licenseplateabbr;
    uint8_t CDCS11_AVM_licenseplateArea;
    uint8_t CDCS11_AVM_LicensePlateNO1;
    uint8_t CDCS11_AVM_LicensePlateNO2;
    uint8_t CDCS11_AVM_LicensePlateNO3;
    uint8_t CDCS11_AVM_LicensePlateNO4;
    uint8_t CDCS11_AVM_LicensePlateNO5;
    uint8_t CDCS11_AVM_LicensePlateNO6;
    uint8_t CDCS11_AVM_LicensePlateNO7;
    uint8_t CDCS11_AVM_LicensePlateNO8;
    uint8_t CDCS11_AVM_DiaphaneitySw;
    uint8_t CDCS11_MOD_Sound_Sw;
    uint8_t CDCS11_MOD_BSD_Sw;
    uint8_t CDCS11_PA_FRS_OnOffSet;
    uint8_t CDCS11_PA_3D_OnOffSet;
    uint8_t CDCS11_PA_measure_OnOffSet;
    uint8_t CDCS11_MODSw;
    uint8_t CDCS11_HPASw;
    uint8_t CDCS11_DeleteMapID;
    uint8_t CDCS11_UploadMapID;
    uint8_t CDCS11_HPApreparkingSw;
    uint8_t CDCS11_HPAPathwaytoCloud_WorkSts;
    uint8_t CDCS11_ParkingoutSlot_Type;
    uint8_t CDCS11_ParkingoutSlot_Dire;
    uint8_t CDCS11_OptionalSlot_Type;
    uint8_t CDCS11_OptionalSlot_Dire;
    double CDCS11_OptionalSlotAngle;
    uint16_t CDCS11_OptionalSlotCoor_P0_X;
    uint16_t CDCS11_OptionalSlotCoor_P0_Y;
    uint16_t CDCS11_OptionalSlotCoor_P1_X;
    uint16_t CDCS11_OptionalSlotCoor_P1_Y;
    uint16_t CDCS11_OptionalSlotCoor_P2_X;
    uint16_t CDCS11_OptionalSlotCoor_P2_Y;
    uint16_t CDCS11_OptionalSlotCoor_P3_X;
    uint16_t CDCS11_OptionalSlotCoor_P3_Y;
    uint8_t DDCU1_FLDoorAjar;
    uint8_t DDCU1_RLDoorAjar;
    uint8_t PDCU1_FRDoorAjar;
    uint8_t PDCU1_RRDoorAjar;
    uint8_t BTM1_SecuritySts;
    uint8_t BTM1_PhoneBatSts;
    uint8_t BTM1_RemoteIntModSel;
    uint8_t BTM1_SelectSlotID;
    uint8_t BTM1_Retry;
    uint8_t BTM1_Fixslot;
    uint8_t BTM1_parkingoutSlot_Dire;
    uint8_t BTM1_parkingoutSlotType;
    uint8_t BTM1_Recover;
    uint8_t BTM1_ParkingReq;
    uint8_t BTM1_FunctionMode;
    uint8_t BTM1_OptionalSlot_Dire;
    uint8_t BTM1_OptionalSlotType;
    uint8_t BTM1_RollingCounter;
    uint8_t BTM1_RemoteParkReq;
    uint8_t BTM1_Movectrl;
    uint8_t BTM1_Checksum;
    double BTM1_OptionalSlotAngle;
    uint16_t BTM1_OptionalSlotCoor_P0_X;
    uint16_t BTM1_OptionalSlotCoor_P0_Y;
    uint16_t BTM1_OptionalSlotCoor_P1_X;
    uint16_t BTM1_OptionalSlotCoor_P1_Y;
    uint16_t BTM1_OptionalSlotCoor_P2_X;
    uint16_t BTM1_OptionalSlotCoor_P2_Y;
    uint16_t BTM1_OptionalSlotCoor_P3_X;
    uint16_t BTM1_OptionalSlotCoor_P3_Y;
    uint8_t TBOX2_AVMOnReq;
    uint8_t TBOX2_RemoteParkStartPressReqSts;
    uint8_t TBOX2_RemoteHPP;
    uint8_t TBOX2_AVMviewReq;
    uint8_t TBOX2_RemoteParkReq;
    uint8_t TBOX2_SecuritySts;
    uint8_t TBOX2_ConnectSts;
    uint8_t TBOX2_PhoneInCar;
    uint8_t TBOX2_RemoteParkAbortReq;
    uint8_t TBOX2_RemoteParkStartPressReq;
    uint8_t TBOX2_RemoteParkOutModSel;
    uint8_t TBOX2_PhoneBatSts;
    uint8_t TBOX2_RemoteIntModSel;
    uint8_t TBOX2_RemoteSw;
    uint8_t TBOX2_RemoteCtrModSel;
    uint8_t TBOX2_RemoteCtrSw;
    uint8_t TBOX2_Remotepick;
    uint8_t TBOX2_NNSsuspend;
    uint8_t TBOX2_RemoteVhpower;
    uint8_t TBOX2_NRPReq;
    uint8_t TBOX2_SDMapReq;
    uint8_t TBOX2_NNSSW;
    uint8_t TBOX2_RollingCounter;
    uint8_t TBOX2_Checksum;
    uint8_t TBOX2_GMOnreq;
    uint8_t TBOX2_360viewReq;
    uint8_t TBOX2_360SpecView;
    uint8_t TBOX2_PA_PauseRecover_Req;
    uint8_t CDCS11_tryHPP;
    uint8_t CDCS11_AVM_Diaphaneity;
    uint8_t CDCS11_HPA_ONPath_ID1;
    uint8_t CDCS11_HPA_ONPath_ID2;
    uint8_t CDCS11_HPA_ONPath_ID3;
    uint8_t CDCS11_HPA_ONPath_ID4;
    uint8_t BDCS1_PowerManageMode;
    uint8_t BTM2_ConnectSts;
    uint8_t BTM2_Key_Position;
    uint8_t BTM3_RKECmd;
    float BMS3_StateOfCharge;
    uint8_t BMS3_Charge_State;
    uint8_t BDCS13_RLS_RQ_WiperSPD;
    uint8_t DDCU1_FL_WinPos;
    uint8_t DDCU1_RL_WinPos;
    uint8_t DDCU1_MirrorFoldSt;
    uint8_t PDCU1_FR_WinPos;
    uint8_t PDCU1_RR_WinPos;
    uint8_t CDCS11_HPA_Select_ID;
    uint8_t BDCS1_PowerMode;
    uint8_t BDCS10_AC_OutsideTempValid;
    uint8_t BDCS10_AC_OutsideTemp;
    uint8_t ADCS9_PA_FRS_Onoffset;
    uint8_t BDCS1_TurnLightSW;
    uint8_t BDCS1_TrunkLockSts;
    uint8_t CDCS11_AVM_Powertype;
    uint8_t ACU1_CrashStatus;
    uint8_t TBOX2_HPA_ONPath_ID;
    uint8_t TBOX2_HPA_PickPath_ID;
    uint8_t CDCS11_learnpath_St;
    uint8_t CDCS11_FindCarAvmReq;
    uint8_t CDCS15_UserSceneMode;
    uint8_t CDCS11_DCWposition;
    uint8_t CDCS15_Storage_St;
    uint8_t CDCS15_USB_St;
};

struct AlgFaultDidInfo {
    bool BDCS10_AC_OutsideTempValid;
    float BDCS10_AC_OutsideTemp;
    uint8_t Power_Supply_Voltage;
    bool ICU1_VehicleSpdDisplayValid;
    float ICU1_VehicleSpdDisplay;
    float ICU2_Odometer;
    uint8_t BDCS1_PowerManageMode;
    uint8_t Ignition_status;
};

struct AlgIgStInfo {
    bool IG_OFF;
    bool ACC;
    bool IG_ON;
    bool Start;
    bool Remote_IG_ON;
    bool reserve_1;
    bool reserve_2;
};

struct AlgChassisTimeInfo {
    uint8_t CalendarYear;
    uint8_t CalendarMonth;
    uint8_t CalendarDay;
    uint8_t HourOfDay;
    uint8_t MinuteOfHour;
    uint8_t SecsOfMinute;
    uint8_t TimeDspFmt;
};

/* ******************************************************************************
    结构 名        :  AlgChassisInfo
    功能描述       :  车辆底盘命令信息
****************************************************************************** */
struct AlgChassisInfo : public AlgDataBase {
    AlgHeader header;
    bool isValid = false;
    AlgVcuInfo vcu_info;
    AlgSteeringInfo steering_info;
    AlgWheelInfo wheel_info;
    AlgEscDrivingInfo esc_driving_info;
    AlgBodyStateInfo body_state_info;
    AlgCenterConsoleInfo center_console_info;
    AlgParkInfo park_info;
    AlgSWSwitchInfo swswitch_info;
    AlgAvmPdsInfo avm_pds_info;
    AlgFaultDidInfo fault_did_info;
    AlgIgStInfo ig_status_info;
    AlgChassisTimeInfo chassis_time_info;
    AlgWarnningHmiInfo warnning_hmi_info;
    AlgChassisErrorCode error_code;
};

struct AlgEgoWarningInfo {
    uint8_t LCARightWarnSt;
    uint8_t LCALeftWarnSt;
    uint8_t LCAFaultStatus;
    uint8_t LCAState;
    uint8_t DOWState;
    uint8_t DOWWarnAudioplay;
    uint8_t DOWLeftWarnSt;
    uint8_t DOWRightWarnSt;
    uint8_t DOWFaultStatus;
    uint8_t RCTAState;
    uint8_t RCTAWarnAudioplay;
    uint8_t RCTAObjType;
    uint8_t RCTALeftWarnSt;
    uint8_t RCTARightWarnSt;
    uint8_t RCTAFaultStatus;
    uint8_t FCTAState;
    uint8_t FCTAWarnAudioplay;
    uint8_t FCTAObjType;
    uint8_t FCTALeftWarnSt;
    uint8_t FCTARightWarnSt;
    uint8_t FCTAFaultStatus;
    uint8_t RCWState;
    uint8_t RCWWarnAudioplay;
    uint8_t RCWWarnSt;
    uint8_t RCWFaultStatus;
    uint8_t Voice_Mode;
};

struct AlgEgoParkHmiInfo {
    uint8_t PA_ParkBarPercent;
    float32_t PA_GuideLineE_a;
    float32_t PA_GuideLineE_b;
    float32_t PA_GuideLineE_c;
    float32_t PA_GuideLineE_d;
    float32_t PA_GuideLineE_Xmin;
    float32_t PA_GuideLineE_Xmax;
    uint8_t HourOfDay;
    uint8_t MinuteOfHour;
    uint8_t SecondOfMinute;
    uint16_t NNS_distance;
    uint16_t HPA_distance;
    uint16_t Parkingtimeremaining;
};

/* ******************************************************************************
    结构 名        :  AlgEgoHmiFrame
    功能描述       :  规控模块写到底盘的HMI相关信息
****************************************************************************** */
struct AlgEgoHmiFrame : public AlgDataBase {
    AlgHeader header;
    bool isValid = false;
    AlgEgoWarningInfo warnning_info;
    AlgEgoParkHmiInfo park_hmi_info;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon