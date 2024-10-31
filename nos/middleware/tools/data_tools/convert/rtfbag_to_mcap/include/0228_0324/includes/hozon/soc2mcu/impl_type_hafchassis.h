/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_HAFCHASSIS_H
#define HOZON_SOC2MCU_IMPL_TYPE_HAFCHASSIS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace hozon {
namespace soc2mcu {
struct HafChassis {
    ::UInt8 YawrateFdbkValid;
    ::UInt8 AxFdbkValid;
    ::UInt8 HsaActive;
    ::UInt8 HsaFail;
    ::UInt8 VehicleStandstill;
    ::UInt8 AccEnable;
    ::Float AccCmdFdbk;
    ::UInt8 VehicleSpdValid;
    ::UInt8 PaPrefillFlag;
    ::UInt8 FLWhlVelocityValid;
    ::Float FLWhlVelocity;
    ::UInt8 FRWhlVelocityValid;
    ::Float FRWhlVelocity;
    ::UInt8 RRWhlDir;
    ::UInt8 RLWhlDir;
    ::UInt8 FRWhlDir;
    ::UInt8 FLWhlDir;
    ::UInt8 RLWhlVelocityValid;
    ::Float RLWhlVelocity;
    ::UInt8 RRWhlVelocityValid;
    ::Float RRWhlVelocity;
    ::UInt8 SteerFdbkValid;
    ::UInt8 SteerFdbkSpdValid;
    ::UInt8 SteerCalibrated;
    ::UInt8 WarningRed;
    ::UInt8 WarningYellow;
    ::Float SteerFdbkSpd;
    ::UInt8 AyFdbkValid;
    ::Float AyFdbk;
    ::Float VxFdbk;
    ::Float VehicleSpdDisplay;
    ::Float AxFdbk;
    ::Float PitchFdbk;
    ::Float YawFdbk;
    ::Float YawRateFdbk;
    ::Float SteerFdbk;
    ::UInt8 GearFdbk;
    ::Float VehicleGyroX;
    ::Float VehicleGyroY;
    ::Float VehicleGyroZ;
    ::Float VehicleAccelX;
    ::Float VehicleAccelY;
    ::Float VehicleAccelZ;
    ::UInt8 IDB6_EPBStatus;
    ::UInt8 IDB1_BrakePedalApplied;
    ::UInt8 IDB9_PA_ApaCtrlSts;
    ::UInt8 EPS2_ADAS_Active;
    ::UInt8 EPS2_PA_Active;
    ::UInt8 VCU14_PAActiv;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(YawrateFdbkValid);
        fun(AxFdbkValid);
        fun(HsaActive);
        fun(HsaFail);
        fun(VehicleStandstill);
        fun(AccEnable);
        fun(AccCmdFdbk);
        fun(VehicleSpdValid);
        fun(PaPrefillFlag);
        fun(FLWhlVelocityValid);
        fun(FLWhlVelocity);
        fun(FRWhlVelocityValid);
        fun(FRWhlVelocity);
        fun(RRWhlDir);
        fun(RLWhlDir);
        fun(FRWhlDir);
        fun(FLWhlDir);
        fun(RLWhlVelocityValid);
        fun(RLWhlVelocity);
        fun(RRWhlVelocityValid);
        fun(RRWhlVelocity);
        fun(SteerFdbkValid);
        fun(SteerFdbkSpdValid);
        fun(SteerCalibrated);
        fun(WarningRed);
        fun(WarningYellow);
        fun(SteerFdbkSpd);
        fun(AyFdbkValid);
        fun(AyFdbk);
        fun(VxFdbk);
        fun(VehicleSpdDisplay);
        fun(AxFdbk);
        fun(PitchFdbk);
        fun(YawFdbk);
        fun(YawRateFdbk);
        fun(SteerFdbk);
        fun(GearFdbk);
        fun(VehicleGyroX);
        fun(VehicleGyroY);
        fun(VehicleGyroZ);
        fun(VehicleAccelX);
        fun(VehicleAccelY);
        fun(VehicleAccelZ);
        fun(IDB6_EPBStatus);
        fun(IDB1_BrakePedalApplied);
        fun(IDB9_PA_ApaCtrlSts);
        fun(EPS2_ADAS_Active);
        fun(EPS2_PA_Active);
        fun(VCU14_PAActiv);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(YawrateFdbkValid);
        fun(AxFdbkValid);
        fun(HsaActive);
        fun(HsaFail);
        fun(VehicleStandstill);
        fun(AccEnable);
        fun(AccCmdFdbk);
        fun(VehicleSpdValid);
        fun(PaPrefillFlag);
        fun(FLWhlVelocityValid);
        fun(FLWhlVelocity);
        fun(FRWhlVelocityValid);
        fun(FRWhlVelocity);
        fun(RRWhlDir);
        fun(RLWhlDir);
        fun(FRWhlDir);
        fun(FLWhlDir);
        fun(RLWhlVelocityValid);
        fun(RLWhlVelocity);
        fun(RRWhlVelocityValid);
        fun(RRWhlVelocity);
        fun(SteerFdbkValid);
        fun(SteerFdbkSpdValid);
        fun(SteerCalibrated);
        fun(WarningRed);
        fun(WarningYellow);
        fun(SteerFdbkSpd);
        fun(AyFdbkValid);
        fun(AyFdbk);
        fun(VxFdbk);
        fun(VehicleSpdDisplay);
        fun(AxFdbk);
        fun(PitchFdbk);
        fun(YawFdbk);
        fun(YawRateFdbk);
        fun(SteerFdbk);
        fun(GearFdbk);
        fun(VehicleGyroX);
        fun(VehicleGyroY);
        fun(VehicleGyroZ);
        fun(VehicleAccelX);
        fun(VehicleAccelY);
        fun(VehicleAccelZ);
        fun(IDB6_EPBStatus);
        fun(IDB1_BrakePedalApplied);
        fun(IDB9_PA_ApaCtrlSts);
        fun(EPS2_ADAS_Active);
        fun(EPS2_PA_Active);
        fun(VCU14_PAActiv);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("YawrateFdbkValid", YawrateFdbkValid);
        fun("AxFdbkValid", AxFdbkValid);
        fun("HsaActive", HsaActive);
        fun("HsaFail", HsaFail);
        fun("VehicleStandstill", VehicleStandstill);
        fun("AccEnable", AccEnable);
        fun("AccCmdFdbk", AccCmdFdbk);
        fun("VehicleSpdValid", VehicleSpdValid);
        fun("PaPrefillFlag", PaPrefillFlag);
        fun("FLWhlVelocityValid", FLWhlVelocityValid);
        fun("FLWhlVelocity", FLWhlVelocity);
        fun("FRWhlVelocityValid", FRWhlVelocityValid);
        fun("FRWhlVelocity", FRWhlVelocity);
        fun("RRWhlDir", RRWhlDir);
        fun("RLWhlDir", RLWhlDir);
        fun("FRWhlDir", FRWhlDir);
        fun("FLWhlDir", FLWhlDir);
        fun("RLWhlVelocityValid", RLWhlVelocityValid);
        fun("RLWhlVelocity", RLWhlVelocity);
        fun("RRWhlVelocityValid", RRWhlVelocityValid);
        fun("RRWhlVelocity", RRWhlVelocity);
        fun("SteerFdbkValid", SteerFdbkValid);
        fun("SteerFdbkSpdValid", SteerFdbkSpdValid);
        fun("SteerCalibrated", SteerCalibrated);
        fun("WarningRed", WarningRed);
        fun("WarningYellow", WarningYellow);
        fun("SteerFdbkSpd", SteerFdbkSpd);
        fun("AyFdbkValid", AyFdbkValid);
        fun("AyFdbk", AyFdbk);
        fun("VxFdbk", VxFdbk);
        fun("VehicleSpdDisplay", VehicleSpdDisplay);
        fun("AxFdbk", AxFdbk);
        fun("PitchFdbk", PitchFdbk);
        fun("YawFdbk", YawFdbk);
        fun("YawRateFdbk", YawRateFdbk);
        fun("SteerFdbk", SteerFdbk);
        fun("GearFdbk", GearFdbk);
        fun("VehicleGyroX", VehicleGyroX);
        fun("VehicleGyroY", VehicleGyroY);
        fun("VehicleGyroZ", VehicleGyroZ);
        fun("VehicleAccelX", VehicleAccelX);
        fun("VehicleAccelY", VehicleAccelY);
        fun("VehicleAccelZ", VehicleAccelZ);
        fun("IDB6_EPBStatus", IDB6_EPBStatus);
        fun("IDB1_BrakePedalApplied", IDB1_BrakePedalApplied);
        fun("IDB9_PA_ApaCtrlSts", IDB9_PA_ApaCtrlSts);
        fun("EPS2_ADAS_Active", EPS2_ADAS_Active);
        fun("EPS2_PA_Active", EPS2_PA_Active);
        fun("VCU14_PAActiv", VCU14_PAActiv);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("YawrateFdbkValid", YawrateFdbkValid);
        fun("AxFdbkValid", AxFdbkValid);
        fun("HsaActive", HsaActive);
        fun("HsaFail", HsaFail);
        fun("VehicleStandstill", VehicleStandstill);
        fun("AccEnable", AccEnable);
        fun("AccCmdFdbk", AccCmdFdbk);
        fun("VehicleSpdValid", VehicleSpdValid);
        fun("PaPrefillFlag", PaPrefillFlag);
        fun("FLWhlVelocityValid", FLWhlVelocityValid);
        fun("FLWhlVelocity", FLWhlVelocity);
        fun("FRWhlVelocityValid", FRWhlVelocityValid);
        fun("FRWhlVelocity", FRWhlVelocity);
        fun("RRWhlDir", RRWhlDir);
        fun("RLWhlDir", RLWhlDir);
        fun("FRWhlDir", FRWhlDir);
        fun("FLWhlDir", FLWhlDir);
        fun("RLWhlVelocityValid", RLWhlVelocityValid);
        fun("RLWhlVelocity", RLWhlVelocity);
        fun("RRWhlVelocityValid", RRWhlVelocityValid);
        fun("RRWhlVelocity", RRWhlVelocity);
        fun("SteerFdbkValid", SteerFdbkValid);
        fun("SteerFdbkSpdValid", SteerFdbkSpdValid);
        fun("SteerCalibrated", SteerCalibrated);
        fun("WarningRed", WarningRed);
        fun("WarningYellow", WarningYellow);
        fun("SteerFdbkSpd", SteerFdbkSpd);
        fun("AyFdbkValid", AyFdbkValid);
        fun("AyFdbk", AyFdbk);
        fun("VxFdbk", VxFdbk);
        fun("VehicleSpdDisplay", VehicleSpdDisplay);
        fun("AxFdbk", AxFdbk);
        fun("PitchFdbk", PitchFdbk);
        fun("YawFdbk", YawFdbk);
        fun("YawRateFdbk", YawRateFdbk);
        fun("SteerFdbk", SteerFdbk);
        fun("GearFdbk", GearFdbk);
        fun("VehicleGyroX", VehicleGyroX);
        fun("VehicleGyroY", VehicleGyroY);
        fun("VehicleGyroZ", VehicleGyroZ);
        fun("VehicleAccelX", VehicleAccelX);
        fun("VehicleAccelY", VehicleAccelY);
        fun("VehicleAccelZ", VehicleAccelZ);
        fun("IDB6_EPBStatus", IDB6_EPBStatus);
        fun("IDB1_BrakePedalApplied", IDB1_BrakePedalApplied);
        fun("IDB9_PA_ApaCtrlSts", IDB9_PA_ApaCtrlSts);
        fun("EPS2_ADAS_Active", EPS2_ADAS_Active);
        fun("EPS2_PA_Active", EPS2_PA_Active);
        fun("VCU14_PAActiv", VCU14_PAActiv);
    }

    bool operator==(const ::hozon::soc2mcu::HafChassis& t) const
    {
        return (YawrateFdbkValid == t.YawrateFdbkValid) && (AxFdbkValid == t.AxFdbkValid) && (HsaActive == t.HsaActive) && (HsaFail == t.HsaFail) && (VehicleStandstill == t.VehicleStandstill) && (AccEnable == t.AccEnable) && (fabs(static_cast<double>(AccCmdFdbk - t.AccCmdFdbk)) < DBL_EPSILON) && (VehicleSpdValid == t.VehicleSpdValid) && (PaPrefillFlag == t.PaPrefillFlag) && (FLWhlVelocityValid == t.FLWhlVelocityValid) && (fabs(static_cast<double>(FLWhlVelocity - t.FLWhlVelocity)) < DBL_EPSILON) && (FRWhlVelocityValid == t.FRWhlVelocityValid) && (fabs(static_cast<double>(FRWhlVelocity - t.FRWhlVelocity)) < DBL_EPSILON) && (RRWhlDir == t.RRWhlDir) && (RLWhlDir == t.RLWhlDir) && (FRWhlDir == t.FRWhlDir) && (FLWhlDir == t.FLWhlDir) && (RLWhlVelocityValid == t.RLWhlVelocityValid) && (fabs(static_cast<double>(RLWhlVelocity - t.RLWhlVelocity)) < DBL_EPSILON) && (RRWhlVelocityValid == t.RRWhlVelocityValid) && (fabs(static_cast<double>(RRWhlVelocity - t.RRWhlVelocity)) < DBL_EPSILON) && (SteerFdbkValid == t.SteerFdbkValid) && (SteerFdbkSpdValid == t.SteerFdbkSpdValid) && (SteerCalibrated == t.SteerCalibrated) && (WarningRed == t.WarningRed) && (WarningYellow == t.WarningYellow) && (fabs(static_cast<double>(SteerFdbkSpd - t.SteerFdbkSpd)) < DBL_EPSILON) && (AyFdbkValid == t.AyFdbkValid) && (fabs(static_cast<double>(AyFdbk - t.AyFdbk)) < DBL_EPSILON) && (fabs(static_cast<double>(VxFdbk - t.VxFdbk)) < DBL_EPSILON) && (fabs(static_cast<double>(VehicleSpdDisplay - t.VehicleSpdDisplay)) < DBL_EPSILON) && (fabs(static_cast<double>(AxFdbk - t.AxFdbk)) < DBL_EPSILON) && (fabs(static_cast<double>(PitchFdbk - t.PitchFdbk)) < DBL_EPSILON) && (fabs(static_cast<double>(YawFdbk - t.YawFdbk)) < DBL_EPSILON) && (fabs(static_cast<double>(YawRateFdbk - t.YawRateFdbk)) < DBL_EPSILON) && (fabs(static_cast<double>(SteerFdbk - t.SteerFdbk)) < DBL_EPSILON) && (GearFdbk == t.GearFdbk) && (fabs(static_cast<double>(VehicleGyroX - t.VehicleGyroX)) < DBL_EPSILON) && (fabs(static_cast<double>(VehicleGyroY - t.VehicleGyroY)) < DBL_EPSILON) && (fabs(static_cast<double>(VehicleGyroZ - t.VehicleGyroZ)) < DBL_EPSILON) && (fabs(static_cast<double>(VehicleAccelX - t.VehicleAccelX)) < DBL_EPSILON) && (fabs(static_cast<double>(VehicleAccelY - t.VehicleAccelY)) < DBL_EPSILON) && (fabs(static_cast<double>(VehicleAccelZ - t.VehicleAccelZ)) < DBL_EPSILON) && (IDB6_EPBStatus == t.IDB6_EPBStatus) && (IDB1_BrakePedalApplied == t.IDB1_BrakePedalApplied) && (IDB9_PA_ApaCtrlSts == t.IDB9_PA_ApaCtrlSts) && (EPS2_ADAS_Active == t.EPS2_ADAS_Active) && (EPS2_PA_Active == t.EPS2_PA_Active) && (VCU14_PAActiv == t.VCU14_PAActiv);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_HAFCHASSIS_H
