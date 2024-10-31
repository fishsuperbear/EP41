/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_USSRAWDATA_APA_H
#define HOZON_SENSORS_IMPL_TYPE_USSRAWDATA_APA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"
#include "impl_type_uint8.h"
#include "impl_type_uint64.h"

namespace hozon {
namespace sensors {
struct UssRawData_APA {
    ::UInt16 distance;
    ::UInt16 distance_2nd;
    ::UInt16 width;
    ::UInt8 peak;
    ::UInt16 Rest_Time;
    ::UInt8 Diagnosis;
    ::UInt64 system_time;
    ::UInt8 counter;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(distance);
        fun(distance_2nd);
        fun(width);
        fun(peak);
        fun(Rest_Time);
        fun(Diagnosis);
        fun(system_time);
        fun(counter);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(distance);
        fun(distance_2nd);
        fun(width);
        fun(peak);
        fun(Rest_Time);
        fun(Diagnosis);
        fun(system_time);
        fun(counter);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("distance", distance);
        fun("distance_2nd", distance_2nd);
        fun("width", width);
        fun("peak", peak);
        fun("Rest_Time", Rest_Time);
        fun("Diagnosis", Diagnosis);
        fun("system_time", system_time);
        fun("counter", counter);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("distance", distance);
        fun("distance_2nd", distance_2nd);
        fun("width", width);
        fun("peak", peak);
        fun("Rest_Time", Rest_Time);
        fun("Diagnosis", Diagnosis);
        fun("system_time", system_time);
        fun("counter", counter);
    }

    bool operator==(const ::hozon::sensors::UssRawData_APA& t) const
    {
        return (distance == t.distance) && (distance_2nd == t.distance_2nd) && (width == t.width) && (peak == t.peak) && (Rest_Time == t.Rest_Time) && (Diagnosis == t.Diagnosis) && (system_time == t.system_time) && (counter == t.counter);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_USSRAWDATA_APA_H
