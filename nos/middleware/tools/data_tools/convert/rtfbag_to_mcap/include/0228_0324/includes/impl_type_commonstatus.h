/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_COMMONSTATUS_H
#define IMPL_TYPE_COMMONSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "impl_type_vindata.h"
#include "impl_type_bodyreportfloat32withvalid.h"

struct CommonStatus {
    ::UInt8 hornStatus;
    ::Float mileage;
    ::VinData vin;
    ::BodyReportFloat32WithValid interiorTemperature;
    ::BodyReportFloat32WithValid outsideTemperature;
    ::UInt8 keyStatus;
    ::UInt8 batterySocStatus;
    ::UInt8 powerSocStatus;
    ::UInt8 dcdcState;
    ::UInt8 gapStatus;
    ::UInt8 rearViewMirrorStatus;
    ::UInt8 rainLightSensorStatus;
    ::UInt8 powerMode;
    ::UInt8 engineRun;
    ::UInt8 chargingStatus;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(hornStatus);
        fun(mileage);
        fun(vin);
        fun(interiorTemperature);
        fun(outsideTemperature);
        fun(keyStatus);
        fun(batterySocStatus);
        fun(powerSocStatus);
        fun(dcdcState);
        fun(gapStatus);
        fun(rearViewMirrorStatus);
        fun(rainLightSensorStatus);
        fun(powerMode);
        fun(engineRun);
        fun(chargingStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(hornStatus);
        fun(mileage);
        fun(vin);
        fun(interiorTemperature);
        fun(outsideTemperature);
        fun(keyStatus);
        fun(batterySocStatus);
        fun(powerSocStatus);
        fun(dcdcState);
        fun(gapStatus);
        fun(rearViewMirrorStatus);
        fun(rainLightSensorStatus);
        fun(powerMode);
        fun(engineRun);
        fun(chargingStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("hornStatus", hornStatus);
        fun("mileage", mileage);
        fun("vin", vin);
        fun("interiorTemperature", interiorTemperature);
        fun("outsideTemperature", outsideTemperature);
        fun("keyStatus", keyStatus);
        fun("batterySocStatus", batterySocStatus);
        fun("powerSocStatus", powerSocStatus);
        fun("dcdcState", dcdcState);
        fun("gapStatus", gapStatus);
        fun("rearViewMirrorStatus", rearViewMirrorStatus);
        fun("rainLightSensorStatus", rainLightSensorStatus);
        fun("powerMode", powerMode);
        fun("engineRun", engineRun);
        fun("chargingStatus", chargingStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("hornStatus", hornStatus);
        fun("mileage", mileage);
        fun("vin", vin);
        fun("interiorTemperature", interiorTemperature);
        fun("outsideTemperature", outsideTemperature);
        fun("keyStatus", keyStatus);
        fun("batterySocStatus", batterySocStatus);
        fun("powerSocStatus", powerSocStatus);
        fun("dcdcState", dcdcState);
        fun("gapStatus", gapStatus);
        fun("rearViewMirrorStatus", rearViewMirrorStatus);
        fun("rainLightSensorStatus", rainLightSensorStatus);
        fun("powerMode", powerMode);
        fun("engineRun", engineRun);
        fun("chargingStatus", chargingStatus);
    }

    bool operator==(const ::CommonStatus& t) const
    {
        return (hornStatus == t.hornStatus) && (fabs(static_cast<double>(mileage - t.mileage)) < DBL_EPSILON) && (vin == t.vin) && (interiorTemperature == t.interiorTemperature) && (outsideTemperature == t.outsideTemperature) && (keyStatus == t.keyStatus) && (batterySocStatus == t.batterySocStatus) && (powerSocStatus == t.powerSocStatus) && (dcdcState == t.dcdcState) && (gapStatus == t.gapStatus) && (rearViewMirrorStatus == t.rearViewMirrorStatus) && (rainLightSensorStatus == t.rainLightSensorStatus) && (powerMode == t.powerMode) && (engineRun == t.engineRun) && (chargingStatus == t.chargingStatus);
    }
};


#endif // IMPL_TYPE_COMMONSTATUS_H
