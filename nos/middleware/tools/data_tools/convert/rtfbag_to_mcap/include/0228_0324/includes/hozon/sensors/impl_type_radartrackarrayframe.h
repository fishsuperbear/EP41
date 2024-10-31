/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_RADARTRACKARRAYFRAME_H
#define HOZON_SENSORS_IMPL_TYPE_RADARTRACKARRAYFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint8.h"
#include "hozon/sensors/impl_type_radartrackdatalist.h"

namespace hozon {
namespace sensors {
struct RadarTrackArrayFrame {
    ::hozon::common::CommonHeader header;
    ::UInt8 sensorID;
    ::UInt8 radarState;
    ::hozon::sensors::RadarTrackDataList trackList;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(sensorID);
        fun(radarState);
        fun(trackList);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(sensorID);
        fun(radarState);
        fun(trackList);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("sensorID", sensorID);
        fun("radarState", radarState);
        fun("trackList", trackList);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("sensorID", sensorID);
        fun("radarState", radarState);
        fun("trackList", trackList);
    }

    bool operator==(const ::hozon::sensors::RadarTrackArrayFrame& t) const
    {
        return (header == t.header) && (sensorID == t.sensorID) && (radarState == t.radarState) && (trackList == t.trackList);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_RADARTRACKARRAYFRAME_H
