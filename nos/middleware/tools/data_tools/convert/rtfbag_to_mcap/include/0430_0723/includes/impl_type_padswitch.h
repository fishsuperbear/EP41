/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_PADSWITCH_H
#define IMPL_TYPE_PADSWITCH_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct PadSwitch {
    ::UInt8 up;
    ::UInt8 down;
    ::UInt8 left;
    ::UInt8 right;
    ::UInt8 ok;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(up);
        fun(down);
        fun(left);
        fun(right);
        fun(ok);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(up);
        fun(down);
        fun(left);
        fun(right);
        fun(ok);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("up", up);
        fun("down", down);
        fun("left", left);
        fun("right", right);
        fun("ok", ok);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("up", up);
        fun("down", down);
        fun("left", left);
        fun("right", right);
        fun("ok", ok);
    }

    bool operator==(const ::PadSwitch& t) const
    {
        return (up == t.up) && (down == t.down) && (left == t.left) && (right == t.right) && (ok == t.ok);
    }
};


#endif // IMPL_TYPE_PADSWITCH_H
