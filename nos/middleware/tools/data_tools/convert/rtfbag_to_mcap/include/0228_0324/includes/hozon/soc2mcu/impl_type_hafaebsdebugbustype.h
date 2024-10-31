/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_HAFAEBSDEBUGBUSTYPE_H
#define HOZON_SOC2MCU_IMPL_TYPE_HAFAEBSDEBUGBUSTYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "impl_type_float.h"
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "impl_type_uint16.h"

namespace hozon {
namespace soc2mcu {
struct HafAEBSDebugBusType {
    ::Boolean CIB_GasPedal_OR;
    ::Boolean CIB_HighBrakeTimer_On;
    ::Boolean CIB_OverrideTimer_On;
    ::Boolean CIB_Steering_OR;
    ::Boolean CIB_Inhibit_Steering_OR;
    ::Boolean CIB_StationaryObjectDelay_On;
    ::Boolean CIB_Brake_Release;
    ::Boolean CIB_FCW;
    ::Float CIB_Filt_Accel_mpss;
    ::UInt8 CIB_Steer_Rate_Flag;
    ::Float CIB_Time_To_Brake;
    ::Float CIB_Time_To_Turn;
    ::Boolean CIB_PCSAlgoEntered;
    ::UInt8 CIB_Activation_Target_ID;
    ::Boolean CIB_NewTrackData;
    ::Boolean CIB_ProcessEachTrack;
    ::Boolean CIB_ProcessCmbbTrack;
    ::Boolean CIB_PCSEntryGatePassed;
    ::Boolean CIB_InPathGatePassed;
    ::Boolean CIB_HighBrakeDeploy;
    ::Boolean CIB_LowBrakeDeploy;
    ::UInt8 CIB_Brake_Level;
    ::Boolean IMUSecondaryPlausibilityFault;
    ::Boolean IMUPrimaryPlausibilityFault;
    ::Boolean WSSPlausibilityFault;
    ::Boolean SASPlausibilityFault;
    ::Float CIB_indicator;
    ::UInt8 CIB_Steer_Rate_Flag1;
    ::Float CIB_Time_To_Brake1;
    ::Float CIB_Time_To_Turn1;
    ::UInt32 CIB_Inhibit_1_Flags;
    ::UInt32 CIB_Inhibit_0_Flags;
    ::UInt32 CIB_Inhibit_3_Flags;
    ::UInt32 CIB_Inhibit_2_Flags;
    ::UInt32 CIB_Inhibit_5_Flags;
    ::UInt32 CIB_Inhibit_4_Flags;
    ::Float CIBMovObjRange;
    ::Float CIBMovObjRangeAccel;
    ::Float CIBMovObjRangeRate;
    ::UInt8 CIBMovObjTrackID;
    ::Float CIBMovObjAngle;
    ::Boolean LowSpeed_ClutterON;
    ::Boolean InTunnelFlag_bool;
    ::UInt32 CIBMovObjTrackPower;
    ::Float CIBMovOffsetTrackRange;
    ::Float CIBStaObjRange;
    ::Float CIBStaObjRangeAccel;
    ::Float CIBStaObjRangeRate;
    ::UInt8 CIBStaObjTrackID;
    ::Float CIBStaObjAngle;
    ::UInt32 CIBStaObjTrackPower;
    ::Float CIBStaOffsetTrackRange;
    ::UInt16 FCW_Ref_indicator;
    ::UInt32 FCW_Inhibit_1_Flags;
    ::UInt32 FCW_Inhibit_0_Flags;
    ::UInt32 FCW_Inhibit_3_Flags;
    ::UInt32 FCW_Inhibit_2_Flags;
    ::UInt32 FCW_Inhibit_5_Flags;
    ::UInt32 FCW_Inhibit_4_Flags;
    ::UInt8 FCWMovObjTrackID;
    ::UInt8 FCWMovComponent;
    ::Float FCWMovWarnRange;
    ::Float FCWCIPVRange;
    ::Float FCWCIPVRangeRate;
    ::Float FCWLeadSpeed_mps;
    ::Float FCWLeadAccel_mpss;
    ::UInt8 FCA_Target_ID;
    ::Boolean Allow_FCA;
    ::UInt32 FCWMovPower;
    ::Boolean FCWChimeSuppress;
    ::Boolean FCA_State;
    ::Float FCWCaution;
    ::Float FCWHostAccel;
    ::Float FCWReactionTime;
    ::Boolean FCWAllowFCWCaution;
    ::Float FCWStatAngle;
    ::UInt8 FCA_Warning_State;
    ::Float FCWMovAngle;
    ::UInt8 FCWStatObjTrackID;
    ::UInt8 FCWStatComponent;
    ::Float FCWCIPSRange;
    ::Float FCWStatWarnRange;
    ::UInt8 FCWCIPSLatTimer;
    ::Float FCWCIPSRangeRate;
    ::UInt32 CIB_TSL_1_Flags;
    ::UInt32 CIB_TSL_0_Flags;
    ::UInt32 FCW_TSL_1_Flags;
    ::UInt32 FCW_TSL_0_Flags;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(CIB_GasPedal_OR);
        fun(CIB_HighBrakeTimer_On);
        fun(CIB_OverrideTimer_On);
        fun(CIB_Steering_OR);
        fun(CIB_Inhibit_Steering_OR);
        fun(CIB_StationaryObjectDelay_On);
        fun(CIB_Brake_Release);
        fun(CIB_FCW);
        fun(CIB_Filt_Accel_mpss);
        fun(CIB_Steer_Rate_Flag);
        fun(CIB_Time_To_Brake);
        fun(CIB_Time_To_Turn);
        fun(CIB_PCSAlgoEntered);
        fun(CIB_Activation_Target_ID);
        fun(CIB_NewTrackData);
        fun(CIB_ProcessEachTrack);
        fun(CIB_ProcessCmbbTrack);
        fun(CIB_PCSEntryGatePassed);
        fun(CIB_InPathGatePassed);
        fun(CIB_HighBrakeDeploy);
        fun(CIB_LowBrakeDeploy);
        fun(CIB_Brake_Level);
        fun(IMUSecondaryPlausibilityFault);
        fun(IMUPrimaryPlausibilityFault);
        fun(WSSPlausibilityFault);
        fun(SASPlausibilityFault);
        fun(CIB_indicator);
        fun(CIB_Steer_Rate_Flag1);
        fun(CIB_Time_To_Brake1);
        fun(CIB_Time_To_Turn1);
        fun(CIB_Inhibit_1_Flags);
        fun(CIB_Inhibit_0_Flags);
        fun(CIB_Inhibit_3_Flags);
        fun(CIB_Inhibit_2_Flags);
        fun(CIB_Inhibit_5_Flags);
        fun(CIB_Inhibit_4_Flags);
        fun(CIBMovObjRange);
        fun(CIBMovObjRangeAccel);
        fun(CIBMovObjRangeRate);
        fun(CIBMovObjTrackID);
        fun(CIBMovObjAngle);
        fun(LowSpeed_ClutterON);
        fun(InTunnelFlag_bool);
        fun(CIBMovObjTrackPower);
        fun(CIBMovOffsetTrackRange);
        fun(CIBStaObjRange);
        fun(CIBStaObjRangeAccel);
        fun(CIBStaObjRangeRate);
        fun(CIBStaObjTrackID);
        fun(CIBStaObjAngle);
        fun(CIBStaObjTrackPower);
        fun(CIBStaOffsetTrackRange);
        fun(FCW_Ref_indicator);
        fun(FCW_Inhibit_1_Flags);
        fun(FCW_Inhibit_0_Flags);
        fun(FCW_Inhibit_3_Flags);
        fun(FCW_Inhibit_2_Flags);
        fun(FCW_Inhibit_5_Flags);
        fun(FCW_Inhibit_4_Flags);
        fun(FCWMovObjTrackID);
        fun(FCWMovComponent);
        fun(FCWMovWarnRange);
        fun(FCWCIPVRange);
        fun(FCWCIPVRangeRate);
        fun(FCWLeadSpeed_mps);
        fun(FCWLeadAccel_mpss);
        fun(FCA_Target_ID);
        fun(Allow_FCA);
        fun(FCWMovPower);
        fun(FCWChimeSuppress);
        fun(FCA_State);
        fun(FCWCaution);
        fun(FCWHostAccel);
        fun(FCWReactionTime);
        fun(FCWAllowFCWCaution);
        fun(FCWStatAngle);
        fun(FCA_Warning_State);
        fun(FCWMovAngle);
        fun(FCWStatObjTrackID);
        fun(FCWStatComponent);
        fun(FCWCIPSRange);
        fun(FCWStatWarnRange);
        fun(FCWCIPSLatTimer);
        fun(FCWCIPSRangeRate);
        fun(CIB_TSL_1_Flags);
        fun(CIB_TSL_0_Flags);
        fun(FCW_TSL_1_Flags);
        fun(FCW_TSL_0_Flags);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(CIB_GasPedal_OR);
        fun(CIB_HighBrakeTimer_On);
        fun(CIB_OverrideTimer_On);
        fun(CIB_Steering_OR);
        fun(CIB_Inhibit_Steering_OR);
        fun(CIB_StationaryObjectDelay_On);
        fun(CIB_Brake_Release);
        fun(CIB_FCW);
        fun(CIB_Filt_Accel_mpss);
        fun(CIB_Steer_Rate_Flag);
        fun(CIB_Time_To_Brake);
        fun(CIB_Time_To_Turn);
        fun(CIB_PCSAlgoEntered);
        fun(CIB_Activation_Target_ID);
        fun(CIB_NewTrackData);
        fun(CIB_ProcessEachTrack);
        fun(CIB_ProcessCmbbTrack);
        fun(CIB_PCSEntryGatePassed);
        fun(CIB_InPathGatePassed);
        fun(CIB_HighBrakeDeploy);
        fun(CIB_LowBrakeDeploy);
        fun(CIB_Brake_Level);
        fun(IMUSecondaryPlausibilityFault);
        fun(IMUPrimaryPlausibilityFault);
        fun(WSSPlausibilityFault);
        fun(SASPlausibilityFault);
        fun(CIB_indicator);
        fun(CIB_Steer_Rate_Flag1);
        fun(CIB_Time_To_Brake1);
        fun(CIB_Time_To_Turn1);
        fun(CIB_Inhibit_1_Flags);
        fun(CIB_Inhibit_0_Flags);
        fun(CIB_Inhibit_3_Flags);
        fun(CIB_Inhibit_2_Flags);
        fun(CIB_Inhibit_5_Flags);
        fun(CIB_Inhibit_4_Flags);
        fun(CIBMovObjRange);
        fun(CIBMovObjRangeAccel);
        fun(CIBMovObjRangeRate);
        fun(CIBMovObjTrackID);
        fun(CIBMovObjAngle);
        fun(LowSpeed_ClutterON);
        fun(InTunnelFlag_bool);
        fun(CIBMovObjTrackPower);
        fun(CIBMovOffsetTrackRange);
        fun(CIBStaObjRange);
        fun(CIBStaObjRangeAccel);
        fun(CIBStaObjRangeRate);
        fun(CIBStaObjTrackID);
        fun(CIBStaObjAngle);
        fun(CIBStaObjTrackPower);
        fun(CIBStaOffsetTrackRange);
        fun(FCW_Ref_indicator);
        fun(FCW_Inhibit_1_Flags);
        fun(FCW_Inhibit_0_Flags);
        fun(FCW_Inhibit_3_Flags);
        fun(FCW_Inhibit_2_Flags);
        fun(FCW_Inhibit_5_Flags);
        fun(FCW_Inhibit_4_Flags);
        fun(FCWMovObjTrackID);
        fun(FCWMovComponent);
        fun(FCWMovWarnRange);
        fun(FCWCIPVRange);
        fun(FCWCIPVRangeRate);
        fun(FCWLeadSpeed_mps);
        fun(FCWLeadAccel_mpss);
        fun(FCA_Target_ID);
        fun(Allow_FCA);
        fun(FCWMovPower);
        fun(FCWChimeSuppress);
        fun(FCA_State);
        fun(FCWCaution);
        fun(FCWHostAccel);
        fun(FCWReactionTime);
        fun(FCWAllowFCWCaution);
        fun(FCWStatAngle);
        fun(FCA_Warning_State);
        fun(FCWMovAngle);
        fun(FCWStatObjTrackID);
        fun(FCWStatComponent);
        fun(FCWCIPSRange);
        fun(FCWStatWarnRange);
        fun(FCWCIPSLatTimer);
        fun(FCWCIPSRangeRate);
        fun(CIB_TSL_1_Flags);
        fun(CIB_TSL_0_Flags);
        fun(FCW_TSL_1_Flags);
        fun(FCW_TSL_0_Flags);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("CIB_GasPedal_OR", CIB_GasPedal_OR);
        fun("CIB_HighBrakeTimer_On", CIB_HighBrakeTimer_On);
        fun("CIB_OverrideTimer_On", CIB_OverrideTimer_On);
        fun("CIB_Steering_OR", CIB_Steering_OR);
        fun("CIB_Inhibit_Steering_OR", CIB_Inhibit_Steering_OR);
        fun("CIB_StationaryObjectDelay_On", CIB_StationaryObjectDelay_On);
        fun("CIB_Brake_Release", CIB_Brake_Release);
        fun("CIB_FCW", CIB_FCW);
        fun("CIB_Filt_Accel_mpss", CIB_Filt_Accel_mpss);
        fun("CIB_Steer_Rate_Flag", CIB_Steer_Rate_Flag);
        fun("CIB_Time_To_Brake", CIB_Time_To_Brake);
        fun("CIB_Time_To_Turn", CIB_Time_To_Turn);
        fun("CIB_PCSAlgoEntered", CIB_PCSAlgoEntered);
        fun("CIB_Activation_Target_ID", CIB_Activation_Target_ID);
        fun("CIB_NewTrackData", CIB_NewTrackData);
        fun("CIB_ProcessEachTrack", CIB_ProcessEachTrack);
        fun("CIB_ProcessCmbbTrack", CIB_ProcessCmbbTrack);
        fun("CIB_PCSEntryGatePassed", CIB_PCSEntryGatePassed);
        fun("CIB_InPathGatePassed", CIB_InPathGatePassed);
        fun("CIB_HighBrakeDeploy", CIB_HighBrakeDeploy);
        fun("CIB_LowBrakeDeploy", CIB_LowBrakeDeploy);
        fun("CIB_Brake_Level", CIB_Brake_Level);
        fun("IMUSecondaryPlausibilityFault", IMUSecondaryPlausibilityFault);
        fun("IMUPrimaryPlausibilityFault", IMUPrimaryPlausibilityFault);
        fun("WSSPlausibilityFault", WSSPlausibilityFault);
        fun("SASPlausibilityFault", SASPlausibilityFault);
        fun("CIB_indicator", CIB_indicator);
        fun("CIB_Steer_Rate_Flag1", CIB_Steer_Rate_Flag1);
        fun("CIB_Time_To_Brake1", CIB_Time_To_Brake1);
        fun("CIB_Time_To_Turn1", CIB_Time_To_Turn1);
        fun("CIB_Inhibit_1_Flags", CIB_Inhibit_1_Flags);
        fun("CIB_Inhibit_0_Flags", CIB_Inhibit_0_Flags);
        fun("CIB_Inhibit_3_Flags", CIB_Inhibit_3_Flags);
        fun("CIB_Inhibit_2_Flags", CIB_Inhibit_2_Flags);
        fun("CIB_Inhibit_5_Flags", CIB_Inhibit_5_Flags);
        fun("CIB_Inhibit_4_Flags", CIB_Inhibit_4_Flags);
        fun("CIBMovObjRange", CIBMovObjRange);
        fun("CIBMovObjRangeAccel", CIBMovObjRangeAccel);
        fun("CIBMovObjRangeRate", CIBMovObjRangeRate);
        fun("CIBMovObjTrackID", CIBMovObjTrackID);
        fun("CIBMovObjAngle", CIBMovObjAngle);
        fun("LowSpeed_ClutterON", LowSpeed_ClutterON);
        fun("InTunnelFlag_bool", InTunnelFlag_bool);
        fun("CIBMovObjTrackPower", CIBMovObjTrackPower);
        fun("CIBMovOffsetTrackRange", CIBMovOffsetTrackRange);
        fun("CIBStaObjRange", CIBStaObjRange);
        fun("CIBStaObjRangeAccel", CIBStaObjRangeAccel);
        fun("CIBStaObjRangeRate", CIBStaObjRangeRate);
        fun("CIBStaObjTrackID", CIBStaObjTrackID);
        fun("CIBStaObjAngle", CIBStaObjAngle);
        fun("CIBStaObjTrackPower", CIBStaObjTrackPower);
        fun("CIBStaOffsetTrackRange", CIBStaOffsetTrackRange);
        fun("FCW_Ref_indicator", FCW_Ref_indicator);
        fun("FCW_Inhibit_1_Flags", FCW_Inhibit_1_Flags);
        fun("FCW_Inhibit_0_Flags", FCW_Inhibit_0_Flags);
        fun("FCW_Inhibit_3_Flags", FCW_Inhibit_3_Flags);
        fun("FCW_Inhibit_2_Flags", FCW_Inhibit_2_Flags);
        fun("FCW_Inhibit_5_Flags", FCW_Inhibit_5_Flags);
        fun("FCW_Inhibit_4_Flags", FCW_Inhibit_4_Flags);
        fun("FCWMovObjTrackID", FCWMovObjTrackID);
        fun("FCWMovComponent", FCWMovComponent);
        fun("FCWMovWarnRange", FCWMovWarnRange);
        fun("FCWCIPVRange", FCWCIPVRange);
        fun("FCWCIPVRangeRate", FCWCIPVRangeRate);
        fun("FCWLeadSpeed_mps", FCWLeadSpeed_mps);
        fun("FCWLeadAccel_mpss", FCWLeadAccel_mpss);
        fun("FCA_Target_ID", FCA_Target_ID);
        fun("Allow_FCA", Allow_FCA);
        fun("FCWMovPower", FCWMovPower);
        fun("FCWChimeSuppress", FCWChimeSuppress);
        fun("FCA_State", FCA_State);
        fun("FCWCaution", FCWCaution);
        fun("FCWHostAccel", FCWHostAccel);
        fun("FCWReactionTime", FCWReactionTime);
        fun("FCWAllowFCWCaution", FCWAllowFCWCaution);
        fun("FCWStatAngle", FCWStatAngle);
        fun("FCA_Warning_State", FCA_Warning_State);
        fun("FCWMovAngle", FCWMovAngle);
        fun("FCWStatObjTrackID", FCWStatObjTrackID);
        fun("FCWStatComponent", FCWStatComponent);
        fun("FCWCIPSRange", FCWCIPSRange);
        fun("FCWStatWarnRange", FCWStatWarnRange);
        fun("FCWCIPSLatTimer", FCWCIPSLatTimer);
        fun("FCWCIPSRangeRate", FCWCIPSRangeRate);
        fun("CIB_TSL_1_Flags", CIB_TSL_1_Flags);
        fun("CIB_TSL_0_Flags", CIB_TSL_0_Flags);
        fun("FCW_TSL_1_Flags", FCW_TSL_1_Flags);
        fun("FCW_TSL_0_Flags", FCW_TSL_0_Flags);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("CIB_GasPedal_OR", CIB_GasPedal_OR);
        fun("CIB_HighBrakeTimer_On", CIB_HighBrakeTimer_On);
        fun("CIB_OverrideTimer_On", CIB_OverrideTimer_On);
        fun("CIB_Steering_OR", CIB_Steering_OR);
        fun("CIB_Inhibit_Steering_OR", CIB_Inhibit_Steering_OR);
        fun("CIB_StationaryObjectDelay_On", CIB_StationaryObjectDelay_On);
        fun("CIB_Brake_Release", CIB_Brake_Release);
        fun("CIB_FCW", CIB_FCW);
        fun("CIB_Filt_Accel_mpss", CIB_Filt_Accel_mpss);
        fun("CIB_Steer_Rate_Flag", CIB_Steer_Rate_Flag);
        fun("CIB_Time_To_Brake", CIB_Time_To_Brake);
        fun("CIB_Time_To_Turn", CIB_Time_To_Turn);
        fun("CIB_PCSAlgoEntered", CIB_PCSAlgoEntered);
        fun("CIB_Activation_Target_ID", CIB_Activation_Target_ID);
        fun("CIB_NewTrackData", CIB_NewTrackData);
        fun("CIB_ProcessEachTrack", CIB_ProcessEachTrack);
        fun("CIB_ProcessCmbbTrack", CIB_ProcessCmbbTrack);
        fun("CIB_PCSEntryGatePassed", CIB_PCSEntryGatePassed);
        fun("CIB_InPathGatePassed", CIB_InPathGatePassed);
        fun("CIB_HighBrakeDeploy", CIB_HighBrakeDeploy);
        fun("CIB_LowBrakeDeploy", CIB_LowBrakeDeploy);
        fun("CIB_Brake_Level", CIB_Brake_Level);
        fun("IMUSecondaryPlausibilityFault", IMUSecondaryPlausibilityFault);
        fun("IMUPrimaryPlausibilityFault", IMUPrimaryPlausibilityFault);
        fun("WSSPlausibilityFault", WSSPlausibilityFault);
        fun("SASPlausibilityFault", SASPlausibilityFault);
        fun("CIB_indicator", CIB_indicator);
        fun("CIB_Steer_Rate_Flag1", CIB_Steer_Rate_Flag1);
        fun("CIB_Time_To_Brake1", CIB_Time_To_Brake1);
        fun("CIB_Time_To_Turn1", CIB_Time_To_Turn1);
        fun("CIB_Inhibit_1_Flags", CIB_Inhibit_1_Flags);
        fun("CIB_Inhibit_0_Flags", CIB_Inhibit_0_Flags);
        fun("CIB_Inhibit_3_Flags", CIB_Inhibit_3_Flags);
        fun("CIB_Inhibit_2_Flags", CIB_Inhibit_2_Flags);
        fun("CIB_Inhibit_5_Flags", CIB_Inhibit_5_Flags);
        fun("CIB_Inhibit_4_Flags", CIB_Inhibit_4_Flags);
        fun("CIBMovObjRange", CIBMovObjRange);
        fun("CIBMovObjRangeAccel", CIBMovObjRangeAccel);
        fun("CIBMovObjRangeRate", CIBMovObjRangeRate);
        fun("CIBMovObjTrackID", CIBMovObjTrackID);
        fun("CIBMovObjAngle", CIBMovObjAngle);
        fun("LowSpeed_ClutterON", LowSpeed_ClutterON);
        fun("InTunnelFlag_bool", InTunnelFlag_bool);
        fun("CIBMovObjTrackPower", CIBMovObjTrackPower);
        fun("CIBMovOffsetTrackRange", CIBMovOffsetTrackRange);
        fun("CIBStaObjRange", CIBStaObjRange);
        fun("CIBStaObjRangeAccel", CIBStaObjRangeAccel);
        fun("CIBStaObjRangeRate", CIBStaObjRangeRate);
        fun("CIBStaObjTrackID", CIBStaObjTrackID);
        fun("CIBStaObjAngle", CIBStaObjAngle);
        fun("CIBStaObjTrackPower", CIBStaObjTrackPower);
        fun("CIBStaOffsetTrackRange", CIBStaOffsetTrackRange);
        fun("FCW_Ref_indicator", FCW_Ref_indicator);
        fun("FCW_Inhibit_1_Flags", FCW_Inhibit_1_Flags);
        fun("FCW_Inhibit_0_Flags", FCW_Inhibit_0_Flags);
        fun("FCW_Inhibit_3_Flags", FCW_Inhibit_3_Flags);
        fun("FCW_Inhibit_2_Flags", FCW_Inhibit_2_Flags);
        fun("FCW_Inhibit_5_Flags", FCW_Inhibit_5_Flags);
        fun("FCW_Inhibit_4_Flags", FCW_Inhibit_4_Flags);
        fun("FCWMovObjTrackID", FCWMovObjTrackID);
        fun("FCWMovComponent", FCWMovComponent);
        fun("FCWMovWarnRange", FCWMovWarnRange);
        fun("FCWCIPVRange", FCWCIPVRange);
        fun("FCWCIPVRangeRate", FCWCIPVRangeRate);
        fun("FCWLeadSpeed_mps", FCWLeadSpeed_mps);
        fun("FCWLeadAccel_mpss", FCWLeadAccel_mpss);
        fun("FCA_Target_ID", FCA_Target_ID);
        fun("Allow_FCA", Allow_FCA);
        fun("FCWMovPower", FCWMovPower);
        fun("FCWChimeSuppress", FCWChimeSuppress);
        fun("FCA_State", FCA_State);
        fun("FCWCaution", FCWCaution);
        fun("FCWHostAccel", FCWHostAccel);
        fun("FCWReactionTime", FCWReactionTime);
        fun("FCWAllowFCWCaution", FCWAllowFCWCaution);
        fun("FCWStatAngle", FCWStatAngle);
        fun("FCA_Warning_State", FCA_Warning_State);
        fun("FCWMovAngle", FCWMovAngle);
        fun("FCWStatObjTrackID", FCWStatObjTrackID);
        fun("FCWStatComponent", FCWStatComponent);
        fun("FCWCIPSRange", FCWCIPSRange);
        fun("FCWStatWarnRange", FCWStatWarnRange);
        fun("FCWCIPSLatTimer", FCWCIPSLatTimer);
        fun("FCWCIPSRangeRate", FCWCIPSRangeRate);
        fun("CIB_TSL_1_Flags", CIB_TSL_1_Flags);
        fun("CIB_TSL_0_Flags", CIB_TSL_0_Flags);
        fun("FCW_TSL_1_Flags", FCW_TSL_1_Flags);
        fun("FCW_TSL_0_Flags", FCW_TSL_0_Flags);
    }

    bool operator==(const ::hozon::soc2mcu::HafAEBSDebugBusType& t) const
    {
        return (CIB_GasPedal_OR == t.CIB_GasPedal_OR) && (CIB_HighBrakeTimer_On == t.CIB_HighBrakeTimer_On) && (CIB_OverrideTimer_On == t.CIB_OverrideTimer_On) && (CIB_Steering_OR == t.CIB_Steering_OR) && (CIB_Inhibit_Steering_OR == t.CIB_Inhibit_Steering_OR) && (CIB_StationaryObjectDelay_On == t.CIB_StationaryObjectDelay_On) && (CIB_Brake_Release == t.CIB_Brake_Release) && (CIB_FCW == t.CIB_FCW) && (fabs(static_cast<double>(CIB_Filt_Accel_mpss - t.CIB_Filt_Accel_mpss)) < DBL_EPSILON) && (CIB_Steer_Rate_Flag == t.CIB_Steer_Rate_Flag) && (fabs(static_cast<double>(CIB_Time_To_Brake - t.CIB_Time_To_Brake)) < DBL_EPSILON) && (fabs(static_cast<double>(CIB_Time_To_Turn - t.CIB_Time_To_Turn)) < DBL_EPSILON) && (CIB_PCSAlgoEntered == t.CIB_PCSAlgoEntered) && (CIB_Activation_Target_ID == t.CIB_Activation_Target_ID) && (CIB_NewTrackData == t.CIB_NewTrackData) && (CIB_ProcessEachTrack == t.CIB_ProcessEachTrack) && (CIB_ProcessCmbbTrack == t.CIB_ProcessCmbbTrack) && (CIB_PCSEntryGatePassed == t.CIB_PCSEntryGatePassed) && (CIB_InPathGatePassed == t.CIB_InPathGatePassed) && (CIB_HighBrakeDeploy == t.CIB_HighBrakeDeploy) && (CIB_LowBrakeDeploy == t.CIB_LowBrakeDeploy) && (CIB_Brake_Level == t.CIB_Brake_Level) && (IMUSecondaryPlausibilityFault == t.IMUSecondaryPlausibilityFault) && (IMUPrimaryPlausibilityFault == t.IMUPrimaryPlausibilityFault) && (WSSPlausibilityFault == t.WSSPlausibilityFault) && (SASPlausibilityFault == t.SASPlausibilityFault) && (fabs(static_cast<double>(CIB_indicator - t.CIB_indicator)) < DBL_EPSILON) && (CIB_Steer_Rate_Flag1 == t.CIB_Steer_Rate_Flag1) && (fabs(static_cast<double>(CIB_Time_To_Brake1 - t.CIB_Time_To_Brake1)) < DBL_EPSILON) && (fabs(static_cast<double>(CIB_Time_To_Turn1 - t.CIB_Time_To_Turn1)) < DBL_EPSILON) && (CIB_Inhibit_1_Flags == t.CIB_Inhibit_1_Flags) && (CIB_Inhibit_0_Flags == t.CIB_Inhibit_0_Flags) && (CIB_Inhibit_3_Flags == t.CIB_Inhibit_3_Flags) && (CIB_Inhibit_2_Flags == t.CIB_Inhibit_2_Flags) && (CIB_Inhibit_5_Flags == t.CIB_Inhibit_5_Flags) && (CIB_Inhibit_4_Flags == t.CIB_Inhibit_4_Flags) && (fabs(static_cast<double>(CIBMovObjRange - t.CIBMovObjRange)) < DBL_EPSILON) && (fabs(static_cast<double>(CIBMovObjRangeAccel - t.CIBMovObjRangeAccel)) < DBL_EPSILON) && (fabs(static_cast<double>(CIBMovObjRangeRate - t.CIBMovObjRangeRate)) < DBL_EPSILON) && (CIBMovObjTrackID == t.CIBMovObjTrackID) && (fabs(static_cast<double>(CIBMovObjAngle - t.CIBMovObjAngle)) < DBL_EPSILON) && (LowSpeed_ClutterON == t.LowSpeed_ClutterON) && (InTunnelFlag_bool == t.InTunnelFlag_bool) && (CIBMovObjTrackPower == t.CIBMovObjTrackPower) && (fabs(static_cast<double>(CIBMovOffsetTrackRange - t.CIBMovOffsetTrackRange)) < DBL_EPSILON) && (fabs(static_cast<double>(CIBStaObjRange - t.CIBStaObjRange)) < DBL_EPSILON) && (fabs(static_cast<double>(CIBStaObjRangeAccel - t.CIBStaObjRangeAccel)) < DBL_EPSILON) && (fabs(static_cast<double>(CIBStaObjRangeRate - t.CIBStaObjRangeRate)) < DBL_EPSILON) && (CIBStaObjTrackID == t.CIBStaObjTrackID) && (fabs(static_cast<double>(CIBStaObjAngle - t.CIBStaObjAngle)) < DBL_EPSILON) && (CIBStaObjTrackPower == t.CIBStaObjTrackPower) && (fabs(static_cast<double>(CIBStaOffsetTrackRange - t.CIBStaOffsetTrackRange)) < DBL_EPSILON) && (FCW_Ref_indicator == t.FCW_Ref_indicator) && (FCW_Inhibit_1_Flags == t.FCW_Inhibit_1_Flags) && (FCW_Inhibit_0_Flags == t.FCW_Inhibit_0_Flags) && (FCW_Inhibit_3_Flags == t.FCW_Inhibit_3_Flags) && (FCW_Inhibit_2_Flags == t.FCW_Inhibit_2_Flags) && (FCW_Inhibit_5_Flags == t.FCW_Inhibit_5_Flags) && (FCW_Inhibit_4_Flags == t.FCW_Inhibit_4_Flags) && (FCWMovObjTrackID == t.FCWMovObjTrackID) && (FCWMovComponent == t.FCWMovComponent) && (fabs(static_cast<double>(FCWMovWarnRange - t.FCWMovWarnRange)) < DBL_EPSILON) && (fabs(static_cast<double>(FCWCIPVRange - t.FCWCIPVRange)) < DBL_EPSILON) && (fabs(static_cast<double>(FCWCIPVRangeRate - t.FCWCIPVRangeRate)) < DBL_EPSILON) && (fabs(static_cast<double>(FCWLeadSpeed_mps - t.FCWLeadSpeed_mps)) < DBL_EPSILON) && (fabs(static_cast<double>(FCWLeadAccel_mpss - t.FCWLeadAccel_mpss)) < DBL_EPSILON) && (FCA_Target_ID == t.FCA_Target_ID) && (Allow_FCA == t.Allow_FCA) && (FCWMovPower == t.FCWMovPower) && (FCWChimeSuppress == t.FCWChimeSuppress) && (FCA_State == t.FCA_State) && (fabs(static_cast<double>(FCWCaution - t.FCWCaution)) < DBL_EPSILON) && (fabs(static_cast<double>(FCWHostAccel - t.FCWHostAccel)) < DBL_EPSILON) && (fabs(static_cast<double>(FCWReactionTime - t.FCWReactionTime)) < DBL_EPSILON) && (FCWAllowFCWCaution == t.FCWAllowFCWCaution) && (fabs(static_cast<double>(FCWStatAngle - t.FCWStatAngle)) < DBL_EPSILON) && (FCA_Warning_State == t.FCA_Warning_State) && (fabs(static_cast<double>(FCWMovAngle - t.FCWMovAngle)) < DBL_EPSILON) && (FCWStatObjTrackID == t.FCWStatObjTrackID) && (FCWStatComponent == t.FCWStatComponent) && (fabs(static_cast<double>(FCWCIPSRange - t.FCWCIPSRange)) < DBL_EPSILON) && (fabs(static_cast<double>(FCWStatWarnRange - t.FCWStatWarnRange)) < DBL_EPSILON) && (FCWCIPSLatTimer == t.FCWCIPSLatTimer) && (fabs(static_cast<double>(FCWCIPSRangeRate - t.FCWCIPSRangeRate)) < DBL_EPSILON) && (CIB_TSL_1_Flags == t.CIB_TSL_1_Flags) && (CIB_TSL_0_Flags == t.CIB_TSL_0_Flags) && (FCW_TSL_1_Flags == t.FCW_TSL_1_Flags) && (FCW_TSL_0_Flags == t.FCW_TSL_0_Flags);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_HAFAEBSDEBUGBUSTYPE_H
