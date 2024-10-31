/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FREESPACE_IMPL_TYPE_FREESPACEFRAME_H
#define HOZON_FREESPACE_IMPL_TYPE_FREESPACEFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "hozon/freespace/impl_type_freespacevector.h"
#include "hozon/freespace/impl_type_freespace2dvector.h"

namespace hozon {
namespace freespace {
struct FreeSpaceFrame {
    ::hozon::common::CommonHeader header;
    ::UInt32 locSeq;
    ::hozon::freespace::FreeSpaceVector spaces;
    ::hozon::freespace::FreeSpace2DVector Spaces2D;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(locSeq);
        fun(spaces);
        fun(Spaces2D);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(locSeq);
        fun(spaces);
        fun(Spaces2D);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("spaces", spaces);
        fun("Spaces2D", Spaces2D);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("spaces", spaces);
        fun("Spaces2D", Spaces2D);
    }

    bool operator==(const ::hozon::freespace::FreeSpaceFrame& t) const
    {
        return (header == t.header) && (locSeq == t.locSeq) && (spaces == t.spaces) && (Spaces2D == t.Spaces2D);
    }
};
} // namespace freespace
} // namespace hozon


#endif // HOZON_FREESPACE_IMPL_TYPE_FREESPACEFRAME_H
