/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_DIAG_IMPL_TYPE_MCUUDSRESFRAME_H
#define HOZON_DIAG_IMPL_TYPE_MCUUDSRESFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/diag/impl_type_uint8array_5.h"
#include "hozon/diag/impl_type_uint8array_18.h"
#include "hozon/diag/impl_type_uint8array_66.h"

namespace hozon {
namespace diag {
struct McuUdsResFrame {
    ::hozon::diag::Uint8Array_5 header;
    ::hozon::diag::Uint8Array_18 f1c1;
    ::hozon::diag::Uint8Array_66 f1c2;
    ::hozon::diag::Uint8Array_66 f1c3;
    ::hozon::diag::Uint8Array_66 f1c4;
    ::hozon::diag::Uint8Array_66 f1c5;
    ::hozon::diag::Uint8Array_66 f1c6;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(f1c1);
        fun(f1c2);
        fun(f1c3);
        fun(f1c4);
        fun(f1c5);
        fun(f1c6);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(f1c1);
        fun(f1c2);
        fun(f1c3);
        fun(f1c4);
        fun(f1c5);
        fun(f1c6);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("f1c1", f1c1);
        fun("f1c2", f1c2);
        fun("f1c3", f1c3);
        fun("f1c4", f1c4);
        fun("f1c5", f1c5);
        fun("f1c6", f1c6);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("f1c1", f1c1);
        fun("f1c2", f1c2);
        fun("f1c3", f1c3);
        fun("f1c4", f1c4);
        fun("f1c5", f1c5);
        fun("f1c6", f1c6);
    }

    bool operator==(const ::hozon::diag::McuUdsResFrame& t) const
    {
        return (header == t.header) && (f1c1 == t.f1c1) && (f1c2 == t.f1c2) && (f1c3 == t.f1c3) && (f1c4 == t.f1c4) && (f1c5 == t.f1c5) && (f1c6 == t.f1c6);
    }
};
} // namespace diag
} // namespace hozon


#endif // HOZON_DIAG_IMPL_TYPE_MCUUDSRESFRAME_H
