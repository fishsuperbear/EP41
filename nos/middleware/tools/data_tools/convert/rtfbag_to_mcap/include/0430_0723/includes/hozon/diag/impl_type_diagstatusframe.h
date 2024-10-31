/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_DIAG_IMPL_TYPE_DIAGSTATUSFRAME_H
#define HOZON_DIAG_IMPL_TYPE_DIAGSTATUSFRAME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace diag {
struct DiagStatusFrame {
    ::UInt8 ecuType;
    ::UInt8 requestType;
    ::UInt8 requestStatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ecuType);
        fun(requestType);
        fun(requestStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ecuType);
        fun(requestType);
        fun(requestStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ecuType", ecuType);
        fun("requestType", requestType);
        fun("requestStatus", requestStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ecuType", ecuType);
        fun("requestType", requestType);
        fun("requestStatus", requestStatus);
    }

    bool operator==(const ::hozon::diag::DiagStatusFrame& t) const
    {
        return (ecuType == t.ecuType) && (requestType == t.requestType) && (requestStatus == t.requestStatus);
    }
};
} // namespace diag
} // namespace hozon


#endif // HOZON_DIAG_IMPL_TYPE_DIAGSTATUSFRAME_H
