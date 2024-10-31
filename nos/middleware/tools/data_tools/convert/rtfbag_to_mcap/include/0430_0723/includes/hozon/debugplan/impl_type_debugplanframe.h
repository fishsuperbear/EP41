/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_DEBUGPLAN_IMPL_TYPE_DEBUGPLANFRAME_H
#define HOZON_DEBUGPLAN_IMPL_TYPE_DEBUGPLANFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_string.h"

namespace hozon {
namespace debugplan {
struct DebugPlanFrame {
    ::hozon::common::CommonHeader header;
    ::String msg_1;
    ::String msg_2;
    ::String msg_3;
    ::String msg_4;
    ::String msg_5;
    ::String msg_6;
    ::String msg_7;
    ::String msg_8;
    ::String msg_9;
    ::String msg_10;
    ::String msg_11;
    ::String msg_12;
    ::String msg_13;
    ::String msg_14;
    ::String msg_15;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(msg_1);
        fun(msg_2);
        fun(msg_3);
        fun(msg_4);
        fun(msg_5);
        fun(msg_6);
        fun(msg_7);
        fun(msg_8);
        fun(msg_9);
        fun(msg_10);
        fun(msg_11);
        fun(msg_12);
        fun(msg_13);
        fun(msg_14);
        fun(msg_15);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(msg_1);
        fun(msg_2);
        fun(msg_3);
        fun(msg_4);
        fun(msg_5);
        fun(msg_6);
        fun(msg_7);
        fun(msg_8);
        fun(msg_9);
        fun(msg_10);
        fun(msg_11);
        fun(msg_12);
        fun(msg_13);
        fun(msg_14);
        fun(msg_15);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("msg_1", msg_1);
        fun("msg_2", msg_2);
        fun("msg_3", msg_3);
        fun("msg_4", msg_4);
        fun("msg_5", msg_5);
        fun("msg_6", msg_6);
        fun("msg_7", msg_7);
        fun("msg_8", msg_8);
        fun("msg_9", msg_9);
        fun("msg_10", msg_10);
        fun("msg_11", msg_11);
        fun("msg_12", msg_12);
        fun("msg_13", msg_13);
        fun("msg_14", msg_14);
        fun("msg_15", msg_15);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("msg_1", msg_1);
        fun("msg_2", msg_2);
        fun("msg_3", msg_3);
        fun("msg_4", msg_4);
        fun("msg_5", msg_5);
        fun("msg_6", msg_6);
        fun("msg_7", msg_7);
        fun("msg_8", msg_8);
        fun("msg_9", msg_9);
        fun("msg_10", msg_10);
        fun("msg_11", msg_11);
        fun("msg_12", msg_12);
        fun("msg_13", msg_13);
        fun("msg_14", msg_14);
        fun("msg_15", msg_15);
    }

    bool operator==(const ::hozon::debugplan::DebugPlanFrame& t) const
    {
        return (header == t.header) && (msg_1 == t.msg_1) && (msg_2 == t.msg_2) && (msg_3 == t.msg_3) && (msg_4 == t.msg_4) && (msg_5 == t.msg_5) && (msg_6 == t.msg_6) && (msg_7 == t.msg_7) && (msg_8 == t.msg_8) && (msg_9 == t.msg_9) && (msg_10 == t.msg_10) && (msg_11 == t.msg_11) && (msg_12 == t.msg_12) && (msg_13 == t.msg_13) && (msg_14 == t.msg_14) && (msg_15 == t.msg_15);
    }
};
} // namespace debugplan
} // namespace hozon


#endif // HOZON_DEBUGPLAN_IMPL_TYPE_DEBUGPLANFRAME_H
