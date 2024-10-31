/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TRAFFICLIGHT_IMPL_TYPE_TRAFFICLIGHTHEADER_H
#define ARA_TRAFFICLIGHT_IMPL_TYPE_TRAFFICLIGHTHEADER_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "ara/adsfi/impl_type_time.h"
#include "impl_type_string.h"
#include "impl_type_int32_t.h"

namespace ara {
namespace trafficlight {
struct TrafficLightHeader {
    ::UInt32 seq;
    ::ara::adsfi::Time stamp;
    ::String frameId;
    ::int32_t count;

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
        fun(count);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(seq);
        fun(stamp);
        fun(frameId);
        fun(count);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("seq", seq);
        fun("stamp", stamp);
        fun("frameId", frameId);
        fun("count", count);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("seq", seq);
        fun("stamp", stamp);
        fun("frameId", frameId);
        fun("count", count);
    }

    bool operator==(const ::ara::trafficlight::TrafficLightHeader& t) const
    {
        return (seq == t.seq) && (stamp == t.stamp) && (frameId == t.frameId) && (count == t.count);
    }
};
} // namespace trafficlight
} // namespace ara


#endif // ARA_TRAFFICLIGHT_IMPL_TYPE_TRAFFICLIGHTHEADER_H
