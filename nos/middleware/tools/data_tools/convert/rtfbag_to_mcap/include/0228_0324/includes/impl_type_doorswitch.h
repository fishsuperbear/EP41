/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_DOORSWITCH_H
#define IMPL_TYPE_DOORSWITCH_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct DoorSwitch {
    ::UInt8 doorFlSwitch;
    ::UInt8 doorFrSwitch;
    ::UInt8 doorRlSwitch;
    ::UInt8 doorRrSwitch;
    ::UInt8 doorHoodSwitch;
    ::UInt8 doorTrunkSwitch;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(doorFlSwitch);
        fun(doorFrSwitch);
        fun(doorRlSwitch);
        fun(doorRrSwitch);
        fun(doorHoodSwitch);
        fun(doorTrunkSwitch);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(doorFlSwitch);
        fun(doorFrSwitch);
        fun(doorRlSwitch);
        fun(doorRrSwitch);
        fun(doorHoodSwitch);
        fun(doorTrunkSwitch);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("doorFlSwitch", doorFlSwitch);
        fun("doorFrSwitch", doorFrSwitch);
        fun("doorRlSwitch", doorRlSwitch);
        fun("doorRrSwitch", doorRrSwitch);
        fun("doorHoodSwitch", doorHoodSwitch);
        fun("doorTrunkSwitch", doorTrunkSwitch);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("doorFlSwitch", doorFlSwitch);
        fun("doorFrSwitch", doorFrSwitch);
        fun("doorRlSwitch", doorRlSwitch);
        fun("doorRrSwitch", doorRrSwitch);
        fun("doorHoodSwitch", doorHoodSwitch);
        fun("doorTrunkSwitch", doorTrunkSwitch);
    }

    bool operator==(const ::DoorSwitch& t) const
    {
        return (doorFlSwitch == t.doorFlSwitch) && (doorFrSwitch == t.doorFrSwitch) && (doorRlSwitch == t.doorRlSwitch) && (doorRrSwitch == t.doorRrSwitch) && (doorHoodSwitch == t.doorHoodSwitch) && (doorTrunkSwitch == t.doorTrunkSwitch);
    }
};


#endif // IMPL_TYPE_DOORSWITCH_H
