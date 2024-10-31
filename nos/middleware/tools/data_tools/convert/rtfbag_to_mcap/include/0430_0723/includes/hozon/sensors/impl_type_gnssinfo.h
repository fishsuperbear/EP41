/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_GNSSINFO_H
#define HOZON_SENSORS_IMPL_TYPE_GNSSINFO_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "impl_type_double.h"
#include "hozon/sensors/impl_type_gnssposinfoframe.h"
#include "hozon/sensors/impl_type_gnssvelinfoframe.h"
#include "hozon/sensors/impl_type_gnssheadinginfoframe.h"

namespace hozon {
namespace sensors {
struct GnssInfo {
    ::hozon::common::CommonHeader header;
    ::UInt32 gpsWeek;
    ::Double gpsSec;
    ::hozon::sensors::GnssPosInfoFrame gnss_pos;
    ::hozon::sensors::GnssVelInfoFrame gnss_vel;
    ::hozon::sensors::GnssHeadingInfoFrame gnss_heading;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(gpsWeek);
        fun(gpsSec);
        fun(gnss_pos);
        fun(gnss_vel);
        fun(gnss_heading);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(gpsWeek);
        fun(gpsSec);
        fun(gnss_pos);
        fun(gnss_vel);
        fun(gnss_heading);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("gnss_pos", gnss_pos);
        fun("gnss_vel", gnss_vel);
        fun("gnss_heading", gnss_heading);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("gnss_pos", gnss_pos);
        fun("gnss_vel", gnss_vel);
        fun("gnss_heading", gnss_heading);
    }

    bool operator==(const ::hozon::sensors::GnssInfo& t) const
    {
        return (header == t.header) && (gpsWeek == t.gpsWeek) && (fabs(static_cast<double>(gpsSec - t.gpsSec)) < DBL_EPSILON) && (gnss_pos == t.gnss_pos) && (gnss_vel == t.gnss_vel) && (gnss_heading == t.gnss_heading);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_GNSSINFO_H
