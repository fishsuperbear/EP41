/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI2LOCATION_IMPL_TYPE_ALGHMIAVPLOCFRAME_H
#define HOZON_HMI2LOCATION_IMPL_TYPE_ALGHMIAVPLOCFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32_t.h"
#include "hozon/composite/impl_type_point3d_double.h"
#include "impl_type_uint16_t.h"

namespace hozon {
namespace hmi2location {
struct AlgHmiAvpLocFrame {
    ::hozon::common::CommonHeader header;
    ::uint32_t gpsWeek;
    double gpsSec;
    double wgsLatitude;
    double wgsLongitude;
    double wgsAltitude;
    float wgsheading;
    double j02Latitude;
    double j02Longitude;
    double j02Altitude;
    float j02heading;
    ::hozon::composite::Point3D_double sdPosition;
    ::uint16_t sysStatus;
    ::uint16_t gpsStatus;

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
        fun(wgsLatitude);
        fun(wgsLongitude);
        fun(wgsAltitude);
        fun(wgsheading);
        fun(j02Latitude);
        fun(j02Longitude);
        fun(j02Altitude);
        fun(j02heading);
        fun(sdPosition);
        fun(sysStatus);
        fun(gpsStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(gpsWeek);
        fun(gpsSec);
        fun(wgsLatitude);
        fun(wgsLongitude);
        fun(wgsAltitude);
        fun(wgsheading);
        fun(j02Latitude);
        fun(j02Longitude);
        fun(j02Altitude);
        fun(j02heading);
        fun(sdPosition);
        fun(sysStatus);
        fun(gpsStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("wgsLatitude", wgsLatitude);
        fun("wgsLongitude", wgsLongitude);
        fun("wgsAltitude", wgsAltitude);
        fun("wgsheading", wgsheading);
        fun("j02Latitude", j02Latitude);
        fun("j02Longitude", j02Longitude);
        fun("j02Altitude", j02Altitude);
        fun("j02heading", j02heading);
        fun("sdPosition", sdPosition);
        fun("sysStatus", sysStatus);
        fun("gpsStatus", gpsStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("wgsLatitude", wgsLatitude);
        fun("wgsLongitude", wgsLongitude);
        fun("wgsAltitude", wgsAltitude);
        fun("wgsheading", wgsheading);
        fun("j02Latitude", j02Latitude);
        fun("j02Longitude", j02Longitude);
        fun("j02Altitude", j02Altitude);
        fun("j02heading", j02heading);
        fun("sdPosition", sdPosition);
        fun("sysStatus", sysStatus);
        fun("gpsStatus", gpsStatus);
    }

    bool operator==(const ::hozon::hmi2location::AlgHmiAvpLocFrame& t) const
    {
        return (header == t.header) && (gpsWeek == t.gpsWeek) && (fabs(static_cast<double>(gpsSec - t.gpsSec)) < DBL_EPSILON) && (fabs(static_cast<double>(wgsLatitude - t.wgsLatitude)) < DBL_EPSILON) && (fabs(static_cast<double>(wgsLongitude - t.wgsLongitude)) < DBL_EPSILON) && (fabs(static_cast<double>(wgsAltitude - t.wgsAltitude)) < DBL_EPSILON) && (fabs(static_cast<double>(wgsheading - t.wgsheading)) < DBL_EPSILON) && (fabs(static_cast<double>(j02Latitude - t.j02Latitude)) < DBL_EPSILON) && (fabs(static_cast<double>(j02Longitude - t.j02Longitude)) < DBL_EPSILON) && (fabs(static_cast<double>(j02Altitude - t.j02Altitude)) < DBL_EPSILON) && (fabs(static_cast<double>(j02heading - t.j02heading)) < DBL_EPSILON) && (sdPosition == t.sdPosition) && (sysStatus == t.sysStatus) && (gpsStatus == t.gpsStatus);
    }
};
} // namespace hmi2location
} // namespace hozon


#endif // HOZON_HMI2LOCATION_IMPL_TYPE_ALGHMIAVPLOCFRAME_H
