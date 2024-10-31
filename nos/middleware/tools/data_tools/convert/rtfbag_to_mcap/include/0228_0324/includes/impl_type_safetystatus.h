/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_SAFETYSTATUS_H
#define IMPL_TYPE_SAFETYSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_bodyreportuint8withvalid.h"

struct SafetyStatus {
    ::UInt8 safeBeltDriverStatus;
    ::UInt8 safeBeltPassengerFrStatus;
    ::UInt8 safeBeltPassengerRlStatus;
    ::UInt8 safeBeltPassengerRmStatus;
    ::UInt8 safeBeltPassengerRrStatus;
    ::UInt8 driverSeatStatus;
    ::UInt8 passengerSeatrFrStatus;
    ::UInt8 passengerSeatrRlStatus;
    ::UInt8 passengerSeatrRmStatus;
    ::UInt8 passengerSeatrRrStatus;
    ::BodyReportUint8WithValid collisionEvent;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(safeBeltDriverStatus);
        fun(safeBeltPassengerFrStatus);
        fun(safeBeltPassengerRlStatus);
        fun(safeBeltPassengerRmStatus);
        fun(safeBeltPassengerRrStatus);
        fun(driverSeatStatus);
        fun(passengerSeatrFrStatus);
        fun(passengerSeatrRlStatus);
        fun(passengerSeatrRmStatus);
        fun(passengerSeatrRrStatus);
        fun(collisionEvent);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(safeBeltDriverStatus);
        fun(safeBeltPassengerFrStatus);
        fun(safeBeltPassengerRlStatus);
        fun(safeBeltPassengerRmStatus);
        fun(safeBeltPassengerRrStatus);
        fun(driverSeatStatus);
        fun(passengerSeatrFrStatus);
        fun(passengerSeatrRlStatus);
        fun(passengerSeatrRmStatus);
        fun(passengerSeatrRrStatus);
        fun(collisionEvent);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("safeBeltDriverStatus", safeBeltDriverStatus);
        fun("safeBeltPassengerFrStatus", safeBeltPassengerFrStatus);
        fun("safeBeltPassengerRlStatus", safeBeltPassengerRlStatus);
        fun("safeBeltPassengerRmStatus", safeBeltPassengerRmStatus);
        fun("safeBeltPassengerRrStatus", safeBeltPassengerRrStatus);
        fun("driverSeatStatus", driverSeatStatus);
        fun("passengerSeatrFrStatus", passengerSeatrFrStatus);
        fun("passengerSeatrRlStatus", passengerSeatrRlStatus);
        fun("passengerSeatrRmStatus", passengerSeatrRmStatus);
        fun("passengerSeatrRrStatus", passengerSeatrRrStatus);
        fun("collisionEvent", collisionEvent);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("safeBeltDriverStatus", safeBeltDriverStatus);
        fun("safeBeltPassengerFrStatus", safeBeltPassengerFrStatus);
        fun("safeBeltPassengerRlStatus", safeBeltPassengerRlStatus);
        fun("safeBeltPassengerRmStatus", safeBeltPassengerRmStatus);
        fun("safeBeltPassengerRrStatus", safeBeltPassengerRrStatus);
        fun("driverSeatStatus", driverSeatStatus);
        fun("passengerSeatrFrStatus", passengerSeatrFrStatus);
        fun("passengerSeatrRlStatus", passengerSeatrRlStatus);
        fun("passengerSeatrRmStatus", passengerSeatrRmStatus);
        fun("passengerSeatrRrStatus", passengerSeatrRrStatus);
        fun("collisionEvent", collisionEvent);
    }

    bool operator==(const ::SafetyStatus& t) const
    {
        return (safeBeltDriverStatus == t.safeBeltDriverStatus) && (safeBeltPassengerFrStatus == t.safeBeltPassengerFrStatus) && (safeBeltPassengerRlStatus == t.safeBeltPassengerRlStatus) && (safeBeltPassengerRmStatus == t.safeBeltPassengerRmStatus) && (safeBeltPassengerRrStatus == t.safeBeltPassengerRrStatus) && (driverSeatStatus == t.driverSeatStatus) && (passengerSeatrFrStatus == t.passengerSeatrFrStatus) && (passengerSeatrRlStatus == t.passengerSeatrRlStatus) && (passengerSeatrRmStatus == t.passengerSeatrRmStatus) && (passengerSeatrRrStatus == t.passengerSeatrRrStatus) && (collisionEvent == t.collisionEvent);
    }
};


#endif // IMPL_TYPE_SAFETYSTATUS_H
