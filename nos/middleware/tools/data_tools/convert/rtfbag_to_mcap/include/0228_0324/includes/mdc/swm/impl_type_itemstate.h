/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_ITEMSTATE_H
#define MDC_SWM_IMPL_TYPE_ITEMSTATE_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_uint8.h"
#include "impl_type_int32.h"

namespace mdc {
namespace swm {
struct itemState {
    ::String name;
    ::UInt8 process;
    ::Int32 errcode;
    ::String message;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(name);
        fun(process);
        fun(errcode);
        fun(message);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(name);
        fun(process);
        fun(errcode);
        fun(message);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("name", name);
        fun("process", process);
        fun("errcode", errcode);
        fun("message", message);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("name", name);
        fun("process", process);
        fun("errcode", errcode);
        fun("message", message);
    }

    bool operator==(const ::mdc::swm::itemState& t) const
    {
        return (name == t.name) && (process == t.process) && (errcode == t.errcode) && (message == t.message);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_ITEMSTATE_H
