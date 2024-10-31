/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_INS_INFO_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_INS_INFO_STRUCT_H
#include <cfloat>
#include <cmath>
#include "hozon/hmi/impl_type_hafheader_struct.h"
#include "impl_type_uint8_t.h"
#include "impl_type_uint16_t.h"
#include "impl_type_uint32_t.h"
#include "hozon/hmi/impl_type_point3d_struct.h"

namespace hozon {
namespace hmi {
struct Ins_Info_Struct {
    ::hozon::hmi::HafHeader_Struct header;
    bool IsValid;
    ::uint8_t padding_u8_1;
    ::uint16_t sysStatus;
    ::uint16_t gpsStatus;
    ::uint16_t padding_u16_1;
    ::uint32_t gpsWeek;
    double gpsSec;
    double wgsLatitude;
    double wgsLongitude;
    double wgsAltitude;
    double wgsheading;
    double j02Latitude;
    double j02Longitude;
    double j02Altitude;
    double j02heading;
    ::hozon::hmi::Point3d_Struct sdPosition;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(IsValid);
        fun(padding_u8_1);
        fun(sysStatus);
        fun(gpsStatus);
        fun(padding_u16_1);
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
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(IsValid);
        fun(padding_u8_1);
        fun(sysStatus);
        fun(gpsStatus);
        fun(padding_u16_1);
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
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("IsValid", IsValid);
        fun("padding_u8_1", padding_u8_1);
        fun("sysStatus", sysStatus);
        fun("gpsStatus", gpsStatus);
        fun("padding_u16_1", padding_u16_1);
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
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("IsValid", IsValid);
        fun("padding_u8_1", padding_u8_1);
        fun("sysStatus", sysStatus);
        fun("gpsStatus", gpsStatus);
        fun("padding_u16_1", padding_u16_1);
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
    }

    bool operator==(const ::hozon::hmi::Ins_Info_Struct& t) const
    {
        return (header == t.header) && (IsValid == t.IsValid) && (padding_u8_1 == t.padding_u8_1) && (sysStatus == t.sysStatus) && (gpsStatus == t.gpsStatus) && (padding_u16_1 == t.padding_u16_1) && (gpsWeek == t.gpsWeek) && (fabs(static_cast<double>(gpsSec - t.gpsSec)) < DBL_EPSILON) && (fabs(static_cast<double>(wgsLatitude - t.wgsLatitude)) < DBL_EPSILON) && (fabs(static_cast<double>(wgsLongitude - t.wgsLongitude)) < DBL_EPSILON) && (fabs(static_cast<double>(wgsAltitude - t.wgsAltitude)) < DBL_EPSILON) && (fabs(static_cast<double>(wgsheading - t.wgsheading)) < DBL_EPSILON) && (fabs(static_cast<double>(j02Latitude - t.j02Latitude)) < DBL_EPSILON) && (fabs(static_cast<double>(j02Longitude - t.j02Longitude)) < DBL_EPSILON) && (fabs(static_cast<double>(j02Altitude - t.j02Altitude)) < DBL_EPSILON) && (fabs(static_cast<double>(j02heading - t.j02heading)) < DBL_EPSILON) && (sdPosition == t.sdPosition);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_INS_INFO_STRUCT_H
