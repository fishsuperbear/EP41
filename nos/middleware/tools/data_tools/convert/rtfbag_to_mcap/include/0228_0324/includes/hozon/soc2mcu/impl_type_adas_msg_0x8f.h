/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X8F_H
#define HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X8F_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace hozon {
namespace soc2mcu {
struct Adas_MSG_0x8F {
    ::UInt8 ADCS11_PA_GearReq;
    ::UInt8 ADCS11_PA_GearPosReqVail;
    ::UInt8 ADCS11_PA_prefillReq;
    ::UInt8 ADCS11_PA_GearPosReq;
    ::UInt8 ADCS11_PA_BrakemodeReq;
    ::UInt8 ADCS11_PA_EpbReq;
    ::Float ADCS11_PA_StopDistance;
    ::Float ADCS11_PA_EPSAngleReq;
    ::UInt8 ADCS11_PA_IDBControlReqV;
    ::Float ADCS11_PA_TarDecel;
    ::UInt8 ADCS11_PA_EPSAngleReqSt;
    ::UInt8 ADCS11_PA_EPSAngleReqV;
    ::Float ADCS11_PA_PAMaxSpd;
    ::UInt8 ADCS11_PA_IDBControlReq;
    ::UInt8 ADCS11_PA_TorqReq;
    ::UInt8 ADCS11_PA_TorqReqValidity;
    ::UInt8 ADCS11_PA_TarDecelreq;
    ::UInt8 ADCS11_PA_ParkingFnMd;
    ::UInt8 ADCS11_PA_StopReq;
    ::UInt8 ADCS11_PA_StopDistanceValid;
    ::Float ADCS11_PA_TorqReqValue;
    ::UInt8 ADCS11_PA_IDB_TorqReqValidity;
    ::Float ADCS11_PA_IDB_TorqReqValue;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADCS11_PA_GearReq);
        fun(ADCS11_PA_GearPosReqVail);
        fun(ADCS11_PA_prefillReq);
        fun(ADCS11_PA_GearPosReq);
        fun(ADCS11_PA_BrakemodeReq);
        fun(ADCS11_PA_EpbReq);
        fun(ADCS11_PA_StopDistance);
        fun(ADCS11_PA_EPSAngleReq);
        fun(ADCS11_PA_IDBControlReqV);
        fun(ADCS11_PA_TarDecel);
        fun(ADCS11_PA_EPSAngleReqSt);
        fun(ADCS11_PA_EPSAngleReqV);
        fun(ADCS11_PA_PAMaxSpd);
        fun(ADCS11_PA_IDBControlReq);
        fun(ADCS11_PA_TorqReq);
        fun(ADCS11_PA_TorqReqValidity);
        fun(ADCS11_PA_TarDecelreq);
        fun(ADCS11_PA_ParkingFnMd);
        fun(ADCS11_PA_StopReq);
        fun(ADCS11_PA_StopDistanceValid);
        fun(ADCS11_PA_TorqReqValue);
        fun(ADCS11_PA_IDB_TorqReqValidity);
        fun(ADCS11_PA_IDB_TorqReqValue);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADCS11_PA_GearReq);
        fun(ADCS11_PA_GearPosReqVail);
        fun(ADCS11_PA_prefillReq);
        fun(ADCS11_PA_GearPosReq);
        fun(ADCS11_PA_BrakemodeReq);
        fun(ADCS11_PA_EpbReq);
        fun(ADCS11_PA_StopDistance);
        fun(ADCS11_PA_EPSAngleReq);
        fun(ADCS11_PA_IDBControlReqV);
        fun(ADCS11_PA_TarDecel);
        fun(ADCS11_PA_EPSAngleReqSt);
        fun(ADCS11_PA_EPSAngleReqV);
        fun(ADCS11_PA_PAMaxSpd);
        fun(ADCS11_PA_IDBControlReq);
        fun(ADCS11_PA_TorqReq);
        fun(ADCS11_PA_TorqReqValidity);
        fun(ADCS11_PA_TarDecelreq);
        fun(ADCS11_PA_ParkingFnMd);
        fun(ADCS11_PA_StopReq);
        fun(ADCS11_PA_StopDistanceValid);
        fun(ADCS11_PA_TorqReqValue);
        fun(ADCS11_PA_IDB_TorqReqValidity);
        fun(ADCS11_PA_IDB_TorqReqValue);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADCS11_PA_GearReq", ADCS11_PA_GearReq);
        fun("ADCS11_PA_GearPosReqVail", ADCS11_PA_GearPosReqVail);
        fun("ADCS11_PA_prefillReq", ADCS11_PA_prefillReq);
        fun("ADCS11_PA_GearPosReq", ADCS11_PA_GearPosReq);
        fun("ADCS11_PA_BrakemodeReq", ADCS11_PA_BrakemodeReq);
        fun("ADCS11_PA_EpbReq", ADCS11_PA_EpbReq);
        fun("ADCS11_PA_StopDistance", ADCS11_PA_StopDistance);
        fun("ADCS11_PA_EPSAngleReq", ADCS11_PA_EPSAngleReq);
        fun("ADCS11_PA_IDBControlReqV", ADCS11_PA_IDBControlReqV);
        fun("ADCS11_PA_TarDecel", ADCS11_PA_TarDecel);
        fun("ADCS11_PA_EPSAngleReqSt", ADCS11_PA_EPSAngleReqSt);
        fun("ADCS11_PA_EPSAngleReqV", ADCS11_PA_EPSAngleReqV);
        fun("ADCS11_PA_PAMaxSpd", ADCS11_PA_PAMaxSpd);
        fun("ADCS11_PA_IDBControlReq", ADCS11_PA_IDBControlReq);
        fun("ADCS11_PA_TorqReq", ADCS11_PA_TorqReq);
        fun("ADCS11_PA_TorqReqValidity", ADCS11_PA_TorqReqValidity);
        fun("ADCS11_PA_TarDecelreq", ADCS11_PA_TarDecelreq);
        fun("ADCS11_PA_ParkingFnMd", ADCS11_PA_ParkingFnMd);
        fun("ADCS11_PA_StopReq", ADCS11_PA_StopReq);
        fun("ADCS11_PA_StopDistanceValid", ADCS11_PA_StopDistanceValid);
        fun("ADCS11_PA_TorqReqValue", ADCS11_PA_TorqReqValue);
        fun("ADCS11_PA_IDB_TorqReqValidity", ADCS11_PA_IDB_TorqReqValidity);
        fun("ADCS11_PA_IDB_TorqReqValue", ADCS11_PA_IDB_TorqReqValue);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADCS11_PA_GearReq", ADCS11_PA_GearReq);
        fun("ADCS11_PA_GearPosReqVail", ADCS11_PA_GearPosReqVail);
        fun("ADCS11_PA_prefillReq", ADCS11_PA_prefillReq);
        fun("ADCS11_PA_GearPosReq", ADCS11_PA_GearPosReq);
        fun("ADCS11_PA_BrakemodeReq", ADCS11_PA_BrakemodeReq);
        fun("ADCS11_PA_EpbReq", ADCS11_PA_EpbReq);
        fun("ADCS11_PA_StopDistance", ADCS11_PA_StopDistance);
        fun("ADCS11_PA_EPSAngleReq", ADCS11_PA_EPSAngleReq);
        fun("ADCS11_PA_IDBControlReqV", ADCS11_PA_IDBControlReqV);
        fun("ADCS11_PA_TarDecel", ADCS11_PA_TarDecel);
        fun("ADCS11_PA_EPSAngleReqSt", ADCS11_PA_EPSAngleReqSt);
        fun("ADCS11_PA_EPSAngleReqV", ADCS11_PA_EPSAngleReqV);
        fun("ADCS11_PA_PAMaxSpd", ADCS11_PA_PAMaxSpd);
        fun("ADCS11_PA_IDBControlReq", ADCS11_PA_IDBControlReq);
        fun("ADCS11_PA_TorqReq", ADCS11_PA_TorqReq);
        fun("ADCS11_PA_TorqReqValidity", ADCS11_PA_TorqReqValidity);
        fun("ADCS11_PA_TarDecelreq", ADCS11_PA_TarDecelreq);
        fun("ADCS11_PA_ParkingFnMd", ADCS11_PA_ParkingFnMd);
        fun("ADCS11_PA_StopReq", ADCS11_PA_StopReq);
        fun("ADCS11_PA_StopDistanceValid", ADCS11_PA_StopDistanceValid);
        fun("ADCS11_PA_TorqReqValue", ADCS11_PA_TorqReqValue);
        fun("ADCS11_PA_IDB_TorqReqValidity", ADCS11_PA_IDB_TorqReqValidity);
        fun("ADCS11_PA_IDB_TorqReqValue", ADCS11_PA_IDB_TorqReqValue);
    }

    bool operator==(const ::hozon::soc2mcu::Adas_MSG_0x8F& t) const
    {
        return (ADCS11_PA_GearReq == t.ADCS11_PA_GearReq) && (ADCS11_PA_GearPosReqVail == t.ADCS11_PA_GearPosReqVail) && (ADCS11_PA_prefillReq == t.ADCS11_PA_prefillReq) && (ADCS11_PA_GearPosReq == t.ADCS11_PA_GearPosReq) && (ADCS11_PA_BrakemodeReq == t.ADCS11_PA_BrakemodeReq) && (ADCS11_PA_EpbReq == t.ADCS11_PA_EpbReq) && (fabs(static_cast<double>(ADCS11_PA_StopDistance - t.ADCS11_PA_StopDistance)) < DBL_EPSILON) && (fabs(static_cast<double>(ADCS11_PA_EPSAngleReq - t.ADCS11_PA_EPSAngleReq)) < DBL_EPSILON) && (ADCS11_PA_IDBControlReqV == t.ADCS11_PA_IDBControlReqV) && (fabs(static_cast<double>(ADCS11_PA_TarDecel - t.ADCS11_PA_TarDecel)) < DBL_EPSILON) && (ADCS11_PA_EPSAngleReqSt == t.ADCS11_PA_EPSAngleReqSt) && (ADCS11_PA_EPSAngleReqV == t.ADCS11_PA_EPSAngleReqV) && (fabs(static_cast<double>(ADCS11_PA_PAMaxSpd - t.ADCS11_PA_PAMaxSpd)) < DBL_EPSILON) && (ADCS11_PA_IDBControlReq == t.ADCS11_PA_IDBControlReq) && (ADCS11_PA_TorqReq == t.ADCS11_PA_TorqReq) && (ADCS11_PA_TorqReqValidity == t.ADCS11_PA_TorqReqValidity) && (ADCS11_PA_TarDecelreq == t.ADCS11_PA_TarDecelreq) && (ADCS11_PA_ParkingFnMd == t.ADCS11_PA_ParkingFnMd) && (ADCS11_PA_StopReq == t.ADCS11_PA_StopReq) && (ADCS11_PA_StopDistanceValid == t.ADCS11_PA_StopDistanceValid) && (fabs(static_cast<double>(ADCS11_PA_TorqReqValue - t.ADCS11_PA_TorqReqValue)) < DBL_EPSILON) && (ADCS11_PA_IDB_TorqReqValidity == t.ADCS11_PA_IDB_TorqReqValidity) && (fabs(static_cast<double>(ADCS11_PA_IDB_TorqReqValue - t.ADCS11_PA_IDB_TorqReqValue)) < DBL_EPSILON);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X8F_H
