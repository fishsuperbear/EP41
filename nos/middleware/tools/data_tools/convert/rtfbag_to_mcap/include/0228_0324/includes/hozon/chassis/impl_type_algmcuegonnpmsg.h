/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGMCUEGONNPMSG_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGMCUEGONNPMSG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace chassis {
struct AlgMcuEgoNNPMsg {
    ::UInt8 LongitudCtrlDecToStopReq;
    ::UInt8 LongitudCtrlDriveOff;
    ::UInt8 DriveOffinhibition;
    ::UInt8 DriveOffinhibitionObjType;
    ::UInt8 Lcsndconfirm;
    ::UInt8 TurnLightReqSt;
    ::UInt8 Lcsndrequest;
    ::UInt8 PayModeConfirm;
    ::UInt8 SpdAdaptComfirm;
    ::UInt8 ALC_mode;
    ::UInt8 ADSDriving_mode;
    ::UInt8 longitudCtrlSetSpeed;
    ::UInt8 longitudCtrlSetDistance;
    ::UInt8 LowBeamSt;
    ::UInt8 HighBeamSt;
    ::UInt8 HazardLampSt;
    ::UInt8 LowHighBeamSt;
    ::UInt8 HornSt;
    ::UInt8 NNPSysState;
    ::uint8_t acc_target_id;
    ::uint8_t alc_warnning_target_id;
    ::uint8_t alc_warnning_state;
    ::uint8_t drive_mode;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(LongitudCtrlDecToStopReq);
        fun(LongitudCtrlDriveOff);
        fun(DriveOffinhibition);
        fun(DriveOffinhibitionObjType);
        fun(Lcsndconfirm);
        fun(TurnLightReqSt);
        fun(Lcsndrequest);
        fun(PayModeConfirm);
        fun(SpdAdaptComfirm);
        fun(ALC_mode);
        fun(ADSDriving_mode);
        fun(longitudCtrlSetSpeed);
        fun(longitudCtrlSetDistance);
        fun(LowBeamSt);
        fun(HighBeamSt);
        fun(HazardLampSt);
        fun(LowHighBeamSt);
        fun(HornSt);
        fun(NNPSysState);
        fun(acc_target_id);
        fun(alc_warnning_target_id);
        fun(alc_warnning_state);
        fun(drive_mode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(LongitudCtrlDecToStopReq);
        fun(LongitudCtrlDriveOff);
        fun(DriveOffinhibition);
        fun(DriveOffinhibitionObjType);
        fun(Lcsndconfirm);
        fun(TurnLightReqSt);
        fun(Lcsndrequest);
        fun(PayModeConfirm);
        fun(SpdAdaptComfirm);
        fun(ALC_mode);
        fun(ADSDriving_mode);
        fun(longitudCtrlSetSpeed);
        fun(longitudCtrlSetDistance);
        fun(LowBeamSt);
        fun(HighBeamSt);
        fun(HazardLampSt);
        fun(LowHighBeamSt);
        fun(HornSt);
        fun(NNPSysState);
        fun(acc_target_id);
        fun(alc_warnning_target_id);
        fun(alc_warnning_state);
        fun(drive_mode);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("LongitudCtrlDecToStopReq", LongitudCtrlDecToStopReq);
        fun("LongitudCtrlDriveOff", LongitudCtrlDriveOff);
        fun("DriveOffinhibition", DriveOffinhibition);
        fun("DriveOffinhibitionObjType", DriveOffinhibitionObjType);
        fun("Lcsndconfirm", Lcsndconfirm);
        fun("TurnLightReqSt", TurnLightReqSt);
        fun("Lcsndrequest", Lcsndrequest);
        fun("PayModeConfirm", PayModeConfirm);
        fun("SpdAdaptComfirm", SpdAdaptComfirm);
        fun("ALC_mode", ALC_mode);
        fun("ADSDriving_mode", ADSDriving_mode);
        fun("longitudCtrlSetSpeed", longitudCtrlSetSpeed);
        fun("longitudCtrlSetDistance", longitudCtrlSetDistance);
        fun("LowBeamSt", LowBeamSt);
        fun("HighBeamSt", HighBeamSt);
        fun("HazardLampSt", HazardLampSt);
        fun("LowHighBeamSt", LowHighBeamSt);
        fun("HornSt", HornSt);
        fun("NNPSysState", NNPSysState);
        fun("acc_target_id", acc_target_id);
        fun("alc_warnning_target_id", alc_warnning_target_id);
        fun("alc_warnning_state", alc_warnning_state);
        fun("drive_mode", drive_mode);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("LongitudCtrlDecToStopReq", LongitudCtrlDecToStopReq);
        fun("LongitudCtrlDriveOff", LongitudCtrlDriveOff);
        fun("DriveOffinhibition", DriveOffinhibition);
        fun("DriveOffinhibitionObjType", DriveOffinhibitionObjType);
        fun("Lcsndconfirm", Lcsndconfirm);
        fun("TurnLightReqSt", TurnLightReqSt);
        fun("Lcsndrequest", Lcsndrequest);
        fun("PayModeConfirm", PayModeConfirm);
        fun("SpdAdaptComfirm", SpdAdaptComfirm);
        fun("ALC_mode", ALC_mode);
        fun("ADSDriving_mode", ADSDriving_mode);
        fun("longitudCtrlSetSpeed", longitudCtrlSetSpeed);
        fun("longitudCtrlSetDistance", longitudCtrlSetDistance);
        fun("LowBeamSt", LowBeamSt);
        fun("HighBeamSt", HighBeamSt);
        fun("HazardLampSt", HazardLampSt);
        fun("LowHighBeamSt", LowHighBeamSt);
        fun("HornSt", HornSt);
        fun("NNPSysState", NNPSysState);
        fun("acc_target_id", acc_target_id);
        fun("alc_warnning_target_id", alc_warnning_target_id);
        fun("alc_warnning_state", alc_warnning_state);
        fun("drive_mode", drive_mode);
    }

    bool operator==(const ::hozon::chassis::AlgMcuEgoNNPMsg& t) const
    {
        return (LongitudCtrlDecToStopReq == t.LongitudCtrlDecToStopReq) && (LongitudCtrlDriveOff == t.LongitudCtrlDriveOff) && (DriveOffinhibition == t.DriveOffinhibition) && (DriveOffinhibitionObjType == t.DriveOffinhibitionObjType) && (Lcsndconfirm == t.Lcsndconfirm) && (TurnLightReqSt == t.TurnLightReqSt) && (Lcsndrequest == t.Lcsndrequest) && (PayModeConfirm == t.PayModeConfirm) && (SpdAdaptComfirm == t.SpdAdaptComfirm) && (ALC_mode == t.ALC_mode) && (ADSDriving_mode == t.ADSDriving_mode) && (longitudCtrlSetSpeed == t.longitudCtrlSetSpeed) && (longitudCtrlSetDistance == t.longitudCtrlSetDistance) && (LowBeamSt == t.LowBeamSt) && (HighBeamSt == t.HighBeamSt) && (HazardLampSt == t.HazardLampSt) && (LowHighBeamSt == t.LowHighBeamSt) && (HornSt == t.HornSt) && (NNPSysState == t.NNPSysState) && (acc_target_id == t.acc_target_id) && (alc_warnning_target_id == t.alc_warnning_target_id) && (alc_warnning_state == t.alc_warnning_state) && (drive_mode == t.drive_mode);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGMCUEGONNPMSG_H
