/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_SOADATA_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_SOADATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint16.h"
#include "impl_type_uint8.h"
#include "impl_type_uint64.h"
#include "hozon/soc_mcu/impl_type_uint8array_58.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_SOAData {
    ::UInt32 TrajData_locSeq;
    ::UInt32 TrajData_sec;
    ::UInt32 TrajData_nsec;
    ::UInt16 TrajData_utmZoneID;
    ::UInt32 PoseData_seq;
    ::UInt32 PoseData_sec;
    ::UInt32 PoseData_nsec;
    ::UInt8 PoseData_locationState;
    ::UInt32 SnsrFsnLaneDate_locSeq;
    ::UInt32 SnsrFsnLaneDate_sec;
    ::UInt32 SnsrFsnLaneDate_nsec;
    ::UInt32 SnsrFsnObj_locSeq;
    ::UInt32 SnsrFsnObj_sec;
    ::UInt32 SnsrFsnObj_nsec;
    ::UInt64 HMI_Dpdata;
    ::UInt8 Lowpower_Rqevent;
    ::hozon::soc_mcu::uint8Array_58 VehicleCfgF170_CfgTmp;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(TrajData_locSeq);
        fun(TrajData_sec);
        fun(TrajData_nsec);
        fun(TrajData_utmZoneID);
        fun(PoseData_seq);
        fun(PoseData_sec);
        fun(PoseData_nsec);
        fun(PoseData_locationState);
        fun(SnsrFsnLaneDate_locSeq);
        fun(SnsrFsnLaneDate_sec);
        fun(SnsrFsnLaneDate_nsec);
        fun(SnsrFsnObj_locSeq);
        fun(SnsrFsnObj_sec);
        fun(SnsrFsnObj_nsec);
        fun(HMI_Dpdata);
        fun(Lowpower_Rqevent);
        fun(VehicleCfgF170_CfgTmp);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(TrajData_locSeq);
        fun(TrajData_sec);
        fun(TrajData_nsec);
        fun(TrajData_utmZoneID);
        fun(PoseData_seq);
        fun(PoseData_sec);
        fun(PoseData_nsec);
        fun(PoseData_locationState);
        fun(SnsrFsnLaneDate_locSeq);
        fun(SnsrFsnLaneDate_sec);
        fun(SnsrFsnLaneDate_nsec);
        fun(SnsrFsnObj_locSeq);
        fun(SnsrFsnObj_sec);
        fun(SnsrFsnObj_nsec);
        fun(HMI_Dpdata);
        fun(Lowpower_Rqevent);
        fun(VehicleCfgF170_CfgTmp);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("TrajData_locSeq", TrajData_locSeq);
        fun("TrajData_sec", TrajData_sec);
        fun("TrajData_nsec", TrajData_nsec);
        fun("TrajData_utmZoneID", TrajData_utmZoneID);
        fun("PoseData_seq", PoseData_seq);
        fun("PoseData_sec", PoseData_sec);
        fun("PoseData_nsec", PoseData_nsec);
        fun("PoseData_locationState", PoseData_locationState);
        fun("SnsrFsnLaneDate_locSeq", SnsrFsnLaneDate_locSeq);
        fun("SnsrFsnLaneDate_sec", SnsrFsnLaneDate_sec);
        fun("SnsrFsnLaneDate_nsec", SnsrFsnLaneDate_nsec);
        fun("SnsrFsnObj_locSeq", SnsrFsnObj_locSeq);
        fun("SnsrFsnObj_sec", SnsrFsnObj_sec);
        fun("SnsrFsnObj_nsec", SnsrFsnObj_nsec);
        fun("HMI_Dpdata", HMI_Dpdata);
        fun("Lowpower_Rqevent", Lowpower_Rqevent);
        fun("VehicleCfgF170_CfgTmp", VehicleCfgF170_CfgTmp);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("TrajData_locSeq", TrajData_locSeq);
        fun("TrajData_sec", TrajData_sec);
        fun("TrajData_nsec", TrajData_nsec);
        fun("TrajData_utmZoneID", TrajData_utmZoneID);
        fun("PoseData_seq", PoseData_seq);
        fun("PoseData_sec", PoseData_sec);
        fun("PoseData_nsec", PoseData_nsec);
        fun("PoseData_locationState", PoseData_locationState);
        fun("SnsrFsnLaneDate_locSeq", SnsrFsnLaneDate_locSeq);
        fun("SnsrFsnLaneDate_sec", SnsrFsnLaneDate_sec);
        fun("SnsrFsnLaneDate_nsec", SnsrFsnLaneDate_nsec);
        fun("SnsrFsnObj_locSeq", SnsrFsnObj_locSeq);
        fun("SnsrFsnObj_sec", SnsrFsnObj_sec);
        fun("SnsrFsnObj_nsec", SnsrFsnObj_nsec);
        fun("HMI_Dpdata", HMI_Dpdata);
        fun("Lowpower_Rqevent", Lowpower_Rqevent);
        fun("VehicleCfgF170_CfgTmp", VehicleCfgF170_CfgTmp);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_SOAData& t) const
    {
        return (TrajData_locSeq == t.TrajData_locSeq) && (TrajData_sec == t.TrajData_sec) && (TrajData_nsec == t.TrajData_nsec) && (TrajData_utmZoneID == t.TrajData_utmZoneID) && (PoseData_seq == t.PoseData_seq) && (PoseData_sec == t.PoseData_sec) && (PoseData_nsec == t.PoseData_nsec) && (PoseData_locationState == t.PoseData_locationState) && (SnsrFsnLaneDate_locSeq == t.SnsrFsnLaneDate_locSeq) && (SnsrFsnLaneDate_sec == t.SnsrFsnLaneDate_sec) && (SnsrFsnLaneDate_nsec == t.SnsrFsnLaneDate_nsec) && (SnsrFsnObj_locSeq == t.SnsrFsnObj_locSeq) && (SnsrFsnObj_sec == t.SnsrFsnObj_sec) && (SnsrFsnObj_nsec == t.SnsrFsnObj_nsec) && (HMI_Dpdata == t.HMI_Dpdata) && (Lowpower_Rqevent == t.Lowpower_Rqevent) && (VehicleCfgF170_CfgTmp == t.VehicleCfgF170_CfgTmp);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_SOADATA_H
