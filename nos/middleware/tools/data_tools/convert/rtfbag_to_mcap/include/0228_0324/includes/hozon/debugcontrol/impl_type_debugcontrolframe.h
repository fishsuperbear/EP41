/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_DEBUGCONTROL_IMPL_TYPE_DEBUGCONTROLFRAME_H
#define HOZON_DEBUGCONTROL_IMPL_TYPE_DEBUGCONTROLFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_string.h"

namespace hozon {
namespace debugcontrol {
struct DebugControlFrame {
    ::hozon::common::CommonHeader header;
    ::String msg_1;
    ::String msg_2;
    ::String msg_3;
    ::String msg_4;

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
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(msg_1);
        fun(msg_2);
        fun(msg_3);
        fun(msg_4);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("msg_1", msg_1);
        fun("msg_2", msg_2);
        fun("msg_3", msg_3);
        fun("msg_4", msg_4);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("msg_1", msg_1);
        fun("msg_2", msg_2);
        fun("msg_3", msg_3);
        fun("msg_4", msg_4);
    }

    bool operator==(const ::hozon::debugcontrol::DebugControlFrame& t) const
    {
        return (header == t.header) && (msg_1 == t.msg_1) && (msg_2 == t.msg_2) && (msg_3 == t.msg_3) && (msg_4 == t.msg_4);
    }
};
} // namespace debugcontrol
} // namespace hozon


#endif // HOZON_DEBUGCONTROL_IMPL_TYPE_DEBUGCONTROLFRAME_H
