/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PARKINGLOT_IMPL_TYPE_PARKINGLOTFRAME_H
#define HOZON_PARKINGLOT_IMPL_TYPE_PARKINGLOTFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "hozon/parkinglot/impl_type_parkinglotvector.h"
#include "hozon/parkinglot/impl_type_pathpointvector.h"

namespace hozon {
namespace parkinglot {
struct ParkingLotFrame {
    ::hozon::common::CommonHeader header;
    ::UInt32 locSeq;
    ::UInt32 optSeq;
    ::hozon::parkinglot::ParkingLotVector parkingLots;
    ::hozon::parkinglot::PathPointVector tracedPath;
    ::UInt32 PathPointSize;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(locSeq);
        fun(optSeq);
        fun(parkingLots);
        fun(tracedPath);
        fun(PathPointSize);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(locSeq);
        fun(optSeq);
        fun(parkingLots);
        fun(tracedPath);
        fun(PathPointSize);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("optSeq", optSeq);
        fun("parkingLots", parkingLots);
        fun("tracedPath", tracedPath);
        fun("PathPointSize", PathPointSize);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("optSeq", optSeq);
        fun("parkingLots", parkingLots);
        fun("tracedPath", tracedPath);
        fun("PathPointSize", PathPointSize);
    }

    bool operator==(const ::hozon::parkinglot::ParkingLotFrame& t) const
    {
        return (header == t.header) && (locSeq == t.locSeq) && (optSeq == t.optSeq) && (parkingLots == t.parkingLots) && (tracedPath == t.tracedPath) && (PathPointSize == t.PathPointSize);
    }
};
} // namespace parkinglot
} // namespace hozon


#endif // HOZON_PARKINGLOT_IMPL_TYPE_PARKINGLOTFRAME_H
