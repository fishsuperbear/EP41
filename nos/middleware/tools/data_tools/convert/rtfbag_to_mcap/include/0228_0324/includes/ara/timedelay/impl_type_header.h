/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TIMEDELAY_IMPL_TYPE_HEADER_H
#define ARA_TIMEDELAY_IMPL_TYPE_HEADER_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "ara/timedelay/impl_type_time.h"
#include "impl_type_string.h"

namespace ara {
namespace timedelay {
struct Header {
    ::UInt32 seq;
    ::ara::timedelay::Time stamp;
    ::String frameId;

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
        fun(frameId);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(seq);
        fun(stamp);
        fun(frameId);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("seq", seq);
        fun("stamp", stamp);
        fun("frameId", frameId);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("seq", seq);
        fun("stamp", stamp);
        fun("frameId", frameId);
    }

    bool operator==(const ::ara::timedelay::Header& t) const
    {
        return (seq == t.seq) && (stamp == t.stamp) && (frameId == t.frameId);
    }
};
} // namespace timedelay
} // namespace ara


#endif // ARA_TIMEDELAY_IMPL_TYPE_HEADER_H