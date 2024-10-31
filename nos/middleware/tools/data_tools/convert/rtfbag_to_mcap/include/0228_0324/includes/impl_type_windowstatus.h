/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_WINDOWSTATUS_H
#define IMPL_TYPE_WINDOWSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct WindowStatus {
    ::UInt8 windowFlStatus;
    ::UInt8 windowFrStatus;
    ::UInt8 windowRlStatus;
    ::UInt8 windowRrStatus;
    ::UInt8 windowTopStatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(windowFlStatus);
        fun(windowFrStatus);
        fun(windowRlStatus);
        fun(windowRrStatus);
        fun(windowTopStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(windowFlStatus);
        fun(windowFrStatus);
        fun(windowRlStatus);
        fun(windowRrStatus);
        fun(windowTopStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("windowFlStatus", windowFlStatus);
        fun("windowFrStatus", windowFrStatus);
        fun("windowRlStatus", windowRlStatus);
        fun("windowRrStatus", windowRrStatus);
        fun("windowTopStatus", windowTopStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("windowFlStatus", windowFlStatus);
        fun("windowFrStatus", windowFrStatus);
        fun("windowRlStatus", windowRlStatus);
        fun("windowRrStatus", windowRrStatus);
        fun("windowTopStatus", windowTopStatus);
    }

    bool operator==(const ::WindowStatus& t) const
    {
        return (windowFlStatus == t.windowFlStatus) && (windowFrStatus == t.windowFrStatus) && (windowRlStatus == t.windowRlStatus) && (windowRrStatus == t.windowRrStatus) && (windowTopStatus == t.windowTopStatus);
    }
};


#endif // IMPL_TYPE_WINDOWSTATUS_H
