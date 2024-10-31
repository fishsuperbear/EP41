/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMMON_IMPL_TYPE_COMMONHEADER_H
#define HOZON_COMMON_IMPL_TYPE_COMMONHEADER_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_string.h"
#include "hozon/common/impl_type_commontime.h"
#include "impl_type_latencyinfo.h"

namespace hozon {
namespace common {
struct CommonHeader {
    ::UInt32 seq;
    ::String frameId;
    ::hozon::common::CommonTime stamp;
    ::hozon::common::CommonTime gnssStamp;
    mdc::LatencyInfo latencyInfo;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(seq);
        fun(frameId);
        fun(stamp);
        fun(gnssStamp);
        fun(latencyInfo);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(seq);
        fun(frameId);
        fun(stamp);
        fun(gnssStamp);
        fun(latencyInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("seq", seq);
        fun("frameId", frameId);
        fun("stamp", stamp);
        fun("gnssStamp", gnssStamp);
        fun("latencyInfo", latencyInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("seq", seq);
        fun("frameId", frameId);
        fun("stamp", stamp);
        fun("gnssStamp", gnssStamp);
        fun("latencyInfo", latencyInfo);
    }

    bool operator==(const ::hozon::common::CommonHeader& t) const
    {
        return (seq == t.seq) && (frameId == t.frameId) && (stamp == t.stamp) && (gnssStamp == t.gnssStamp) && (latencyInfo == t.latencyInfo);
    }
};
} // namespace common
} // namespace hozon


#endif // HOZON_COMMON_IMPL_TYPE_COMMONHEADER_H
