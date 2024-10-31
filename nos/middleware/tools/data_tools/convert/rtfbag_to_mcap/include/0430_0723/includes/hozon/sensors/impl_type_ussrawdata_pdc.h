/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_USSRAWDATA_PDC_H
#define HOZON_SENSORS_IMPL_TYPE_USSRAWDATA_PDC_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_uint16arry_10.h"
#include "hozon/composite/impl_type_uint8array_10.h"
#include "impl_type_uint64.h"
#include "impl_type_uint16.h"

namespace hozon {
namespace sensors {
struct UssRawData_PDC {
    ::UInt8 echo_num;
    ::hozon::composite::uint16Arry_10 distance;
    ::hozon::composite::uint16Arry_10 width;
    ::hozon::composite::uint8Array_10 peak;
    ::UInt8 status_error;
    ::UInt8 status_work;
    ::UInt8 counter;
    ::UInt64 system_time;
    ::UInt16 wTxSns_Ringtime;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(echo_num);
        fun(distance);
        fun(width);
        fun(peak);
        fun(status_error);
        fun(status_work);
        fun(counter);
        fun(system_time);
        fun(wTxSns_Ringtime);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(echo_num);
        fun(distance);
        fun(width);
        fun(peak);
        fun(status_error);
        fun(status_work);
        fun(counter);
        fun(system_time);
        fun(wTxSns_Ringtime);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("echo_num", echo_num);
        fun("distance", distance);
        fun("width", width);
        fun("peak", peak);
        fun("status_error", status_error);
        fun("status_work", status_work);
        fun("counter", counter);
        fun("system_time", system_time);
        fun("wTxSns_Ringtime", wTxSns_Ringtime);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("echo_num", echo_num);
        fun("distance", distance);
        fun("width", width);
        fun("peak", peak);
        fun("status_error", status_error);
        fun("status_work", status_work);
        fun("counter", counter);
        fun("system_time", system_time);
        fun("wTxSns_Ringtime", wTxSns_Ringtime);
    }

    bool operator==(const ::hozon::sensors::UssRawData_PDC& t) const
    {
        return (echo_num == t.echo_num) && (distance == t.distance) && (width == t.width) && (peak == t.peak) && (status_error == t.status_error) && (status_work == t.status_work) && (counter == t.counter) && (system_time == t.system_time) && (wTxSns_Ringtime == t.wTxSns_Ringtime);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_USSRAWDATA_PDC_H
