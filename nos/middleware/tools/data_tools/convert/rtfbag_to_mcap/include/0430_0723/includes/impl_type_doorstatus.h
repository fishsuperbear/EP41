/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_DOORSTATUS_H
#define IMPL_TYPE_DOORSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct DoorStatus {
    ::UInt8 doorFlStatus;
    ::UInt8 doorFrStatus;
    ::UInt8 doorRlStatus;
    ::UInt8 doorRrStatus;
    ::UInt8 doorHoodStatus;
    ::UInt8 doorTrunkStatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(doorFlStatus);
        fun(doorFrStatus);
        fun(doorRlStatus);
        fun(doorRrStatus);
        fun(doorHoodStatus);
        fun(doorTrunkStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(doorFlStatus);
        fun(doorFrStatus);
        fun(doorRlStatus);
        fun(doorRrStatus);
        fun(doorHoodStatus);
        fun(doorTrunkStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("doorFlStatus", doorFlStatus);
        fun("doorFrStatus", doorFrStatus);
        fun("doorRlStatus", doorRlStatus);
        fun("doorRrStatus", doorRrStatus);
        fun("doorHoodStatus", doorHoodStatus);
        fun("doorTrunkStatus", doorTrunkStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("doorFlStatus", doorFlStatus);
        fun("doorFrStatus", doorFrStatus);
        fun("doorRlStatus", doorRlStatus);
        fun("doorRrStatus", doorRrStatus);
        fun("doorHoodStatus", doorHoodStatus);
        fun("doorTrunkStatus", doorTrunkStatus);
    }

    bool operator==(const ::DoorStatus& t) const
    {
        return (doorFlStatus == t.doorFlStatus) && (doorFrStatus == t.doorFrStatus) && (doorRlStatus == t.doorRlStatus) && (doorRrStatus == t.doorRrStatus) && (doorHoodStatus == t.doorHoodStatus) && (doorTrunkStatus == t.doorTrunkStatus);
    }
};


#endif // IMPL_TYPE_DOORSTATUS_H
