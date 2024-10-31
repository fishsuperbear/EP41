/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_RADAR_IMPL_TYPE_TIMESTAMPINFO_H
#define ARA_RADAR_IMPL_TYPE_TIMESTAMPINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"
#include "impl_type_uint64.h"

namespace ara {
namespace radar {
struct TimeStampInfo {
    ::UInt16 timeStampStatus;
    ::UInt64 timeStamp;
    ::UInt64 autoSarTimeStamp;
    ::UInt64 reserved1;
    ::UInt64 reserved2;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(timeStampStatus);
        fun(timeStamp);
        fun(autoSarTimeStamp);
        fun(reserved1);
        fun(reserved2);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(timeStampStatus);
        fun(timeStamp);
        fun(autoSarTimeStamp);
        fun(reserved1);
        fun(reserved2);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("timeStampStatus", timeStampStatus);
        fun("timeStamp", timeStamp);
        fun("autoSarTimeStamp", autoSarTimeStamp);
        fun("reserved1", reserved1);
        fun("reserved2", reserved2);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("timeStampStatus", timeStampStatus);
        fun("timeStamp", timeStamp);
        fun("autoSarTimeStamp", autoSarTimeStamp);
        fun("reserved1", reserved1);
        fun("reserved2", reserved2);
    }

    bool operator==(const ::ara::radar::TimeStampInfo& t) const
    {
        return (timeStampStatus == t.timeStampStatus) && (timeStamp == t.timeStamp) && (autoSarTimeStamp == t.autoSarTimeStamp) && (reserved1 == t.reserved1) && (reserved2 == t.reserved2);
    }
};
} // namespace radar
} // namespace ara


#endif // ARA_RADAR_IMPL_TYPE_TIMESTAMPINFO_H
