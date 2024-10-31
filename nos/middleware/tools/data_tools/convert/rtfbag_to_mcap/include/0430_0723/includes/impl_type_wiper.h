/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_WIPER_H
#define IMPL_TYPE_WIPER_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct Wiper {
    ::UInt8 wiperValue;
    ::UInt8 wiperSpeedValue;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(wiperValue);
        fun(wiperSpeedValue);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(wiperValue);
        fun(wiperSpeedValue);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("wiperValue", wiperValue);
        fun("wiperSpeedValue", wiperSpeedValue);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("wiperValue", wiperValue);
        fun("wiperSpeedValue", wiperSpeedValue);
    }

    bool operator==(const ::Wiper& t) const
    {
        return (wiperValue == t.wiperValue) && (wiperSpeedValue == t.wiperSpeedValue);
    }
};


#endif // IMPL_TYPE_WIPER_H
