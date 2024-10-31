/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TIMEDELAY_IMPL_TYPE_TIMEDELAYDATATYPE_H
#define ARA_TIMEDELAY_IMPL_TYPE_TIMEDELAYDATATYPE_H
#include <cfloat>
#include <cmath>
#include "ara/timedelay/impl_type_header.h"
#include "ara/timedelay/impl_type_timepoint.h"
#include "impl_type_uint64.h"

namespace ara {
namespace timedelay {
struct TimeDelayDataType {
    ::ara::timedelay::Header header;
    ::ara::timedelay::TimePoint mdcRecvTimePoint;
    ::ara::timedelay::TimePoint sensorAccessRecvTimePoint;
    ::ara::timedelay::TimePoint publishTimePoint;
    ::UInt64 driverTimeDelay;
    ::UInt64 sensorAccessTimeDelay;
    ::UInt64 forwardTimeDelay;
    ::UInt64 frameTimeDelay;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(mdcRecvTimePoint);
        fun(sensorAccessRecvTimePoint);
        fun(publishTimePoint);
        fun(driverTimeDelay);
        fun(sensorAccessTimeDelay);
        fun(forwardTimeDelay);
        fun(frameTimeDelay);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(mdcRecvTimePoint);
        fun(sensorAccessRecvTimePoint);
        fun(publishTimePoint);
        fun(driverTimeDelay);
        fun(sensorAccessTimeDelay);
        fun(forwardTimeDelay);
        fun(frameTimeDelay);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("mdcRecvTimePoint", mdcRecvTimePoint);
        fun("sensorAccessRecvTimePoint", sensorAccessRecvTimePoint);
        fun("publishTimePoint", publishTimePoint);
        fun("driverTimeDelay", driverTimeDelay);
        fun("sensorAccessTimeDelay", sensorAccessTimeDelay);
        fun("forwardTimeDelay", forwardTimeDelay);
        fun("frameTimeDelay", frameTimeDelay);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("mdcRecvTimePoint", mdcRecvTimePoint);
        fun("sensorAccessRecvTimePoint", sensorAccessRecvTimePoint);
        fun("publishTimePoint", publishTimePoint);
        fun("driverTimeDelay", driverTimeDelay);
        fun("sensorAccessTimeDelay", sensorAccessTimeDelay);
        fun("forwardTimeDelay", forwardTimeDelay);
        fun("frameTimeDelay", frameTimeDelay);
    }

    bool operator==(const ::ara::timedelay::TimeDelayDataType& t) const
    {
        return (header == t.header) && (mdcRecvTimePoint == t.mdcRecvTimePoint) && (sensorAccessRecvTimePoint == t.sensorAccessRecvTimePoint) && (publishTimePoint == t.publishTimePoint) && (driverTimeDelay == t.driverTimeDelay) && (sensorAccessTimeDelay == t.sensorAccessTimeDelay) && (forwardTimeDelay == t.forwardTimeDelay) && (frameTimeDelay == t.frameTimeDelay);
    }
};
} // namespace timedelay
} // namespace ara


#endif // ARA_TIMEDELAY_IMPL_TYPE_TIMEDELAYDATATYPE_H
