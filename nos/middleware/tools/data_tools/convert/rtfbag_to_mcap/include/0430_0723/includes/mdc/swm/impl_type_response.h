/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_RESPONSE_H
#define MDC_SWM_IMPL_TYPE_RESPONSE_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"
#include "impl_type_string.h"

namespace mdc {
namespace swm {
struct Response {
    ::Int32 code;
    ::String message;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(code);
        fun(message);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(code);
        fun(message);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("code", code);
        fun("message", message);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("code", code);
        fun("message", message);
    }

    bool operator==(const ::mdc::swm::Response& t) const
    {
        return (code == t.code) && (message == t.message);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_RESPONSE_H
