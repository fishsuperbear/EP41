/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_AVPMAPMSG_IMPL_TYPE_AVPMAPMSGFRAME_H
#define HOZON_AVPMAPMSG_IMPL_TYPE_AVPMAPMSGFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "hozon/parkinglot/impl_type_parkinglotvector.h"
#include "hozon/parkinglot/impl_type_pathpointvector.h"
#include "hozon/avpmapmsg/impl_type_mapmarkeroutvector.h"
#include "hozon/avpmapmsg/impl_type_maplaneoutvector.h"
#include "hozon/avpmapmsg/impl_type_maplaneobjectvector.h"
#include "impl_type_float.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace avpmapmsg {
struct AvpMapmsgFrame {
    ::hozon::common::CommonHeader header;
    ::UInt32 map_id;
    ::UInt32 optParkingSeq;
    ::hozon::parkinglot::ParkingLotVector parkingLots;
    ::hozon::parkinglot::PathPointVector tracedPath;
    ::UInt32 PathPointSize;
    ::hozon::avpmapmsg::MapMarkerOutVector Marker;
    ::hozon::avpmapmsg::MapLaneOutVector MapLane;
    ::hozon::avpmapmsg::MapLaneObjectvector MapLaneObject;
    ::Float hppLon;
    ::Float hppLat;
    ::Boolean isValid;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(map_id);
        fun(optParkingSeq);
        fun(parkingLots);
        fun(tracedPath);
        fun(PathPointSize);
        fun(Marker);
        fun(MapLane);
        fun(MapLaneObject);
        fun(hppLon);
        fun(hppLat);
        fun(isValid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(map_id);
        fun(optParkingSeq);
        fun(parkingLots);
        fun(tracedPath);
        fun(PathPointSize);
        fun(Marker);
        fun(MapLane);
        fun(MapLaneObject);
        fun(hppLon);
        fun(hppLat);
        fun(isValid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("map_id", map_id);
        fun("optParkingSeq", optParkingSeq);
        fun("parkingLots", parkingLots);
        fun("tracedPath", tracedPath);
        fun("PathPointSize", PathPointSize);
        fun("Marker", Marker);
        fun("MapLane", MapLane);
        fun("MapLaneObject", MapLaneObject);
        fun("hppLon", hppLon);
        fun("hppLat", hppLat);
        fun("isValid", isValid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("map_id", map_id);
        fun("optParkingSeq", optParkingSeq);
        fun("parkingLots", parkingLots);
        fun("tracedPath", tracedPath);
        fun("PathPointSize", PathPointSize);
        fun("Marker", Marker);
        fun("MapLane", MapLane);
        fun("MapLaneObject", MapLaneObject);
        fun("hppLon", hppLon);
        fun("hppLat", hppLat);
        fun("isValid", isValid);
    }

    bool operator==(const ::hozon::avpmapmsg::AvpMapmsgFrame& t) const
    {
        return (header == t.header) && (map_id == t.map_id) && (optParkingSeq == t.optParkingSeq) && (parkingLots == t.parkingLots) && (tracedPath == t.tracedPath) && (PathPointSize == t.PathPointSize) && (Marker == t.Marker) && (MapLane == t.MapLane) && (MapLaneObject == t.MapLaneObject) && (fabs(static_cast<double>(hppLon - t.hppLon)) < DBL_EPSILON) && (fabs(static_cast<double>(hppLat - t.hppLat)) < DBL_EPSILON) && (isValid == t.isValid);
    }
};
} // namespace avpmapmsg
} // namespace hozon


#endif // HOZON_AVPMAPMSG_IMPL_TYPE_AVPMAPMSGFRAME_H
