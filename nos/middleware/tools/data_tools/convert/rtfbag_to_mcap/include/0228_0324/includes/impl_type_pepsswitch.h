/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_PEPSSWITCH_H
#define IMPL_TYPE_PEPSSWITCH_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct PepsSwitch {
    ::UInt8 pepsSwitch;
    ::UInt8 pepsSwitchUnLock;
    ::UInt8 pepsSwitchPackIn;
    ::UInt8 pepsSwitchPackOut;
    ::UInt8 pepsSwitchTrunkOpen;
    ::UInt8 pepsSwitchTrunkClose;
    ::UInt8 pepsSwitchRemoteStart;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(pepsSwitch);
        fun(pepsSwitchUnLock);
        fun(pepsSwitchPackIn);
        fun(pepsSwitchPackOut);
        fun(pepsSwitchTrunkOpen);
        fun(pepsSwitchTrunkClose);
        fun(pepsSwitchRemoteStart);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(pepsSwitch);
        fun(pepsSwitchUnLock);
        fun(pepsSwitchPackIn);
        fun(pepsSwitchPackOut);
        fun(pepsSwitchTrunkOpen);
        fun(pepsSwitchTrunkClose);
        fun(pepsSwitchRemoteStart);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("pepsSwitch", pepsSwitch);
        fun("pepsSwitchUnLock", pepsSwitchUnLock);
        fun("pepsSwitchPackIn", pepsSwitchPackIn);
        fun("pepsSwitchPackOut", pepsSwitchPackOut);
        fun("pepsSwitchTrunkOpen", pepsSwitchTrunkOpen);
        fun("pepsSwitchTrunkClose", pepsSwitchTrunkClose);
        fun("pepsSwitchRemoteStart", pepsSwitchRemoteStart);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("pepsSwitch", pepsSwitch);
        fun("pepsSwitchUnLock", pepsSwitchUnLock);
        fun("pepsSwitchPackIn", pepsSwitchPackIn);
        fun("pepsSwitchPackOut", pepsSwitchPackOut);
        fun("pepsSwitchTrunkOpen", pepsSwitchTrunkOpen);
        fun("pepsSwitchTrunkClose", pepsSwitchTrunkClose);
        fun("pepsSwitchRemoteStart", pepsSwitchRemoteStart);
    }

    bool operator==(const ::PepsSwitch& t) const
    {
        return (pepsSwitch == t.pepsSwitch) && (pepsSwitchUnLock == t.pepsSwitchUnLock) && (pepsSwitchPackIn == t.pepsSwitchPackIn) && (pepsSwitchPackOut == t.pepsSwitchPackOut) && (pepsSwitchTrunkOpen == t.pepsSwitchTrunkOpen) && (pepsSwitchTrunkClose == t.pepsSwitchTrunkClose) && (pepsSwitchRemoteStart == t.pepsSwitchRemoteStart);
    }
};


#endif // IMPL_TYPE_PEPSSWITCH_H
