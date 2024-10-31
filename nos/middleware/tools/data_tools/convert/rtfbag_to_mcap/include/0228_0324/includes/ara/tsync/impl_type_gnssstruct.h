/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TSYNC_IMPL_TYPE_GNSSSTRUCT_H
#define ARA_TSYNC_IMPL_TYPE_GNSSSTRUCT_H
#include <cfloat>
#include <cmath>
#include "ara/gnss/impl_type_header.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"

namespace ara {
namespace tsync {
struct GnssStruct {
    ::ara::gnss::Header header;
    ::UInt32 gnssTime;
    ::UInt8 gnssTimeStatus;
    ::UInt8 leapSecondType;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(gnssTime);
        fun(gnssTimeStatus);
        fun(leapSecondType);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(gnssTime);
        fun(gnssTimeStatus);
        fun(leapSecondType);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("gnssTime", gnssTime);
        fun("gnssTimeStatus", gnssTimeStatus);
        fun("leapSecondType", leapSecondType);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("gnssTime", gnssTime);
        fun("gnssTimeStatus", gnssTimeStatus);
        fun("leapSecondType", leapSecondType);
    }

    bool operator==(const ::ara::tsync::GnssStruct& t) const
    {
        return (header == t.header) && (gnssTime == t.gnssTime) && (gnssTimeStatus == t.gnssTimeStatus) && (leapSecondType == t.leapSecondType);
    }
};
} // namespace tsync
} // namespace ara


#endif // ARA_TSYNC_IMPL_TYPE_GNSSSTRUCT_H
