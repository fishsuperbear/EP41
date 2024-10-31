/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ACTCOMPENSATION_IMPL_TYPE_STATUSINFO_H
#define ARA_ACTCOMPENSATION_IMPL_TYPE_STATUSINFO_H
#include <cfloat>
#include <cmath>
#include "ara/actcompensation/impl_type_doorstatus.h"
#include "ara/actcompensation/impl_type_uint8withvalid.h"

namespace ara {
namespace actcompensation {
struct StatusInfo {
    ::ara::actcompensation::DoorStatus doorStatus;
    ::ara::actcompensation::Uint8WithValid epbLockStatus;
    ::ara::actcompensation::Uint8WithValid absStatus;
    ::ara::actcompensation::Uint8WithValid tcsStatus;
    ::ara::actcompensation::Uint8WithValid vdcStatus;
    ::ara::actcompensation::Uint8WithValid standStill;
    ::ara::actcompensation::Uint8WithValid epsFunctionStyle;
    ::ara::actcompensation::Uint8WithValid sasCalibration;
    ::ara::actcompensation::Uint8WithValid steerMode;
    ::ara::actcompensation::Uint8WithValid steerWarnLamp;
    ::ara::actcompensation::Uint8WithValid keyStatus;
    ::ara::actcompensation::Uint8WithValid systemReadyStatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(doorStatus);
        fun(epbLockStatus);
        fun(absStatus);
        fun(tcsStatus);
        fun(vdcStatus);
        fun(standStill);
        fun(epsFunctionStyle);
        fun(sasCalibration);
        fun(steerMode);
        fun(steerWarnLamp);
        fun(keyStatus);
        fun(systemReadyStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(doorStatus);
        fun(epbLockStatus);
        fun(absStatus);
        fun(tcsStatus);
        fun(vdcStatus);
        fun(standStill);
        fun(epsFunctionStyle);
        fun(sasCalibration);
        fun(steerMode);
        fun(steerWarnLamp);
        fun(keyStatus);
        fun(systemReadyStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("doorStatus", doorStatus);
        fun("epbLockStatus", epbLockStatus);
        fun("absStatus", absStatus);
        fun("tcsStatus", tcsStatus);
        fun("vdcStatus", vdcStatus);
        fun("standStill", standStill);
        fun("epsFunctionStyle", epsFunctionStyle);
        fun("sasCalibration", sasCalibration);
        fun("steerMode", steerMode);
        fun("steerWarnLamp", steerWarnLamp);
        fun("keyStatus", keyStatus);
        fun("systemReadyStatus", systemReadyStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("doorStatus", doorStatus);
        fun("epbLockStatus", epbLockStatus);
        fun("absStatus", absStatus);
        fun("tcsStatus", tcsStatus);
        fun("vdcStatus", vdcStatus);
        fun("standStill", standStill);
        fun("epsFunctionStyle", epsFunctionStyle);
        fun("sasCalibration", sasCalibration);
        fun("steerMode", steerMode);
        fun("steerWarnLamp", steerWarnLamp);
        fun("keyStatus", keyStatus);
        fun("systemReadyStatus", systemReadyStatus);
    }

    bool operator==(const ::ara::actcompensation::StatusInfo& t) const
    {
        return (doorStatus == t.doorStatus) && (epbLockStatus == t.epbLockStatus) && (absStatus == t.absStatus) && (tcsStatus == t.tcsStatus) && (vdcStatus == t.vdcStatus) && (standStill == t.standStill) && (epsFunctionStyle == t.epsFunctionStyle) && (sasCalibration == t.sasCalibration) && (steerMode == t.steerMode) && (steerWarnLamp == t.steerWarnLamp) && (keyStatus == t.keyStatus) && (systemReadyStatus == t.systemReadyStatus);
    }
};
} // namespace actcompensation
} // namespace ara


#endif // ARA_ACTCOMPENSATION_IMPL_TYPE_STATUSINFO_H
