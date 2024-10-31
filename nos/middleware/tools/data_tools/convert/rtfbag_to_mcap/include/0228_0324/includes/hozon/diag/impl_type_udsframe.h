/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_DIAG_IMPL_TYPE_UDSFRAME_H
#define HOZON_DIAG_IMPL_TYPE_UDSFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/diag/impl_type_uint8vector.h"

namespace hozon {
namespace diag {
struct UdsFrame {
    ::hozon::diag::Uint8Vector uds;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(uds);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(uds);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("uds", uds);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("uds", uds);
    }

    bool operator==(const ::hozon::diag::UdsFrame& t) const
    {
        return (uds == t.uds);
    }
};
} // namespace diag
} // namespace hozon


#endif // HOZON_DIAG_IMPL_TYPE_UDSFRAME_H
