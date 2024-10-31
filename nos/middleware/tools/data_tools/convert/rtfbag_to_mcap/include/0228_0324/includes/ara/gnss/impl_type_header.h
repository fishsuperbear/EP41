/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_GNSS_IMPL_TYPE_HEADER_H
#define ARA_GNSS_IMPL_TYPE_HEADER_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "ara/ins/impl_type_time.h"
#include "impl_type_string.h"

namespace ara {
namespace gnss {
struct Header {
    ::UInt32 seq;
    ::ara::ins::Time stamp;
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

    bool operator==(const ::ara::gnss::Header& t) const
    {
        return (seq == t.seq) && (stamp == t.stamp) && (frame_id == t.frame_id);
    }
};
} // namespace gnss
} // namespace ara


#endif // ARA_GNSS_IMPL_TYPE_HEADER_H
