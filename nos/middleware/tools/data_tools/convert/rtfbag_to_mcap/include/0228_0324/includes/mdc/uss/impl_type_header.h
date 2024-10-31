/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_USS_IMPL_TYPE_HEADER_H
#define MDC_USS_IMPL_TYPE_HEADER_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "mdc/uss/impl_type_time.h"
#include "impl_type_string.h"

namespace mdc {
namespace uss {
struct Header {
    ::UInt32 seq;
    ::mdc::uss::Time stamp;
    ::String frame_id;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(seq);
        fun(stamp);
        fun(frame_id);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(seq);
        fun(stamp);
        fun(frame_id);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("seq", seq);
        fun("stamp", stamp);
        fun("frame_id", frame_id);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("seq", seq);
        fun("stamp", stamp);
        fun("frame_id", frame_id);
    }

    bool operator==(const ::mdc::uss::Header& t) const
    {
        return (seq == t.seq) && (stamp == t.stamp) && (frame_id == t.frame_id);
    }
};
} // namespace uss
} // namespace mdc


#endif // MDC_USS_IMPL_TYPE_HEADER_H
