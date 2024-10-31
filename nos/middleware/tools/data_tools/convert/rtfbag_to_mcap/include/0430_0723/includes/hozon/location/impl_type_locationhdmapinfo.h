/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_LOCATIONHDMAPINFO_H
#define HOZON_LOCATION_IMPL_TYPE_LOCATIONHDMAPINFO_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32_t.h"
#include "impl_type_uint32.h"
#include "hozon/location/impl_type_laneinfovec.h"
#include "hozon/location/impl_type_linepointvec.h"
#include "hozon/location/impl_type_trafficsignvec.h"
#include "hozon/location/impl_type_polevec.h"
#include "hozon/location/impl_type_roadmarkingvec.h"
#include "hozon/composite/impl_type_floatarray36.h"
#include "impl_type_float.h"

namespace hozon {
namespace location {
struct LocationHDMapInfo {
    ::hozon::common::CommonHeader header;
    ::uint32_t errorCode;
    ::UInt32 innerCode;
    ::hozon::location::LaneInfoVec laneCenter;
    ::hozon::location::LinePointVec lineBoundary;
    ::hozon::location::TrafficSignVec trafficSign;
    ::hozon::location::PoleVec pole;
    ::hozon::location::RoadMarkingVec roadMarking;
    ::hozon::composite::FloatArray36 covariance;
    ::uint32_t sysStatus;
    ::uint32_t gpsStatus;
    ::Float heading;
    ::uint32_t warn_info;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(errorCode);
        fun(innerCode);
        fun(laneCenter);
        fun(lineBoundary);
        fun(trafficSign);
        fun(pole);
        fun(roadMarking);
        fun(covariance);
        fun(sysStatus);
        fun(gpsStatus);
        fun(heading);
        fun(warn_info);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(errorCode);
        fun(innerCode);
        fun(laneCenter);
        fun(lineBoundary);
        fun(trafficSign);
        fun(pole);
        fun(roadMarking);
        fun(covariance);
        fun(sysStatus);
        fun(gpsStatus);
        fun(heading);
        fun(warn_info);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("errorCode", errorCode);
        fun("innerCode", innerCode);
        fun("laneCenter", laneCenter);
        fun("lineBoundary", lineBoundary);
        fun("trafficSign", trafficSign);
        fun("pole", pole);
        fun("roadMarking", roadMarking);
        fun("covariance", covariance);
        fun("sysStatus", sysStatus);
        fun("gpsStatus", gpsStatus);
        fun("heading", heading);
        fun("warn_info", warn_info);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("errorCode", errorCode);
        fun("innerCode", innerCode);
        fun("laneCenter", laneCenter);
        fun("lineBoundary", lineBoundary);
        fun("trafficSign", trafficSign);
        fun("pole", pole);
        fun("roadMarking", roadMarking);
        fun("covariance", covariance);
        fun("sysStatus", sysStatus);
        fun("gpsStatus", gpsStatus);
        fun("heading", heading);
        fun("warn_info", warn_info);
    }

    bool operator==(const ::hozon::location::LocationHDMapInfo& t) const
    {
        return (header == t.header) && (errorCode == t.errorCode) && (innerCode == t.innerCode) && (laneCenter == t.laneCenter) && (lineBoundary == t.lineBoundary) && (trafficSign == t.trafficSign) && (pole == t.pole) && (roadMarking == t.roadMarking) && (covariance == t.covariance) && (sysStatus == t.sysStatus) && (gpsStatus == t.gpsStatus) && (fabs(static_cast<double>(heading - t.heading)) < DBL_EPSILON) && (warn_info == t.warn_info);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_LOCATIONHDMAPINFO_H
