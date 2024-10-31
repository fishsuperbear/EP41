/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_PARKINGSPOT_H
#define IMPL_TYPE_PARKINGSPOT_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "impl_type_pointarray.h"
#include "impl_type_uint8array.h"

struct ParkingSpot {
    ::ara::common::CommonHeader header;
    ::UInt32 id;
    ::UInt8 parking_class;
    ::Float class_confidence;
    ::PointArray parking_spot_ego;
    ::PointArray parking_spot_enu;
    ::Uint8Array entry;
    ::UInt8 parking_spot_type;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(id);
        fun(parking_class);
        fun(class_confidence);
        fun(parking_spot_ego);
        fun(parking_spot_enu);
        fun(entry);
        fun(parking_spot_type);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(id);
        fun(parking_class);
        fun(class_confidence);
        fun(parking_spot_ego);
        fun(parking_spot_enu);
        fun(entry);
        fun(parking_spot_type);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("id", id);
        fun("parking_class", parking_class);
        fun("class_confidence", class_confidence);
        fun("parking_spot_ego", parking_spot_ego);
        fun("parking_spot_enu", parking_spot_enu);
        fun("entry", entry);
        fun("parking_spot_type", parking_spot_type);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("id", id);
        fun("parking_class", parking_class);
        fun("class_confidence", class_confidence);
        fun("parking_spot_ego", parking_spot_ego);
        fun("parking_spot_enu", parking_spot_enu);
        fun("entry", entry);
        fun("parking_spot_type", parking_spot_type);
    }

    bool operator==(const ::ParkingSpot& t) const
    {
        return (header == t.header) && (id == t.id) && (parking_class == t.parking_class) && (fabs(static_cast<double>(class_confidence - t.class_confidence)) < DBL_EPSILON) && (parking_spot_ego == t.parking_spot_ego) && (parking_spot_enu == t.parking_spot_enu) && (entry == t.entry) && (parking_spot_type == t.parking_spot_type);
    }
};


#endif // IMPL_TYPE_PARKINGSPOT_H
