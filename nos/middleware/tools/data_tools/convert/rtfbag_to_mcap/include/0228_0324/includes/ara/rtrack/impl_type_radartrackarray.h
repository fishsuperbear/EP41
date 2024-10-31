/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_RTRACK_IMPL_TYPE_RADARTRACKARRAY_H
#define ARA_RTRACK_IMPL_TYPE_RADARTRACKARRAY_H
#include <cfloat>
#include <cmath>
#include "ara/radar/impl_type_header.h"
#include "ara/radar/impl_type_version.h"
#include "impl_type_uint8.h"
#include "ara/radar/impl_type_radarstate.h"
#include "impl_type_tracklist.h"
#include "ara/radar/impl_type_guardnailinfo.h"
#include "ara/radar/impl_type_timestampinfo.h"

namespace ara {
namespace rtrack {
struct RadarTrackArray {
    ::ara::radar::Header header;
    ::ara::radar::Version version;
    ::UInt8 sensorId;
    ::ara::radar::RadarState radarState;
    ::TrackList trackList;
    ::ara::radar::GuardnailInfo guardnailInfo;
    ::ara::radar::TimeStampInfo timeStampInfo;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(version);
        fun(sensorId);
        fun(radarState);
        fun(trackList);
        fun(guardnailInfo);
        fun(timeStampInfo);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(version);
        fun(sensorId);
        fun(radarState);
        fun(trackList);
        fun(guardnailInfo);
        fun(timeStampInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("version", version);
        fun("sensorId", sensorId);
        fun("radarState", radarState);
        fun("trackList", trackList);
        fun("guardnailInfo", guardnailInfo);
        fun("timeStampInfo", timeStampInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("version", version);
        fun("sensorId", sensorId);
        fun("radarState", radarState);
        fun("trackList", trackList);
        fun("guardnailInfo", guardnailInfo);
        fun("timeStampInfo", timeStampInfo);
    }

    bool operator==(const ::ara::rtrack::RadarTrackArray& t) const
    {
        return (header == t.header) && (version == t.version) && (sensorId == t.sensorId) && (radarState == t.radarState) && (trackList == t.trackList) && (guardnailInfo == t.guardnailInfo) && (timeStampInfo == t.timeStampInfo);
    }
};
} // namespace rtrack
} // namespace ara


#endif // ARA_RTRACK_IMPL_TYPE_RADARTRACKARRAY_H
