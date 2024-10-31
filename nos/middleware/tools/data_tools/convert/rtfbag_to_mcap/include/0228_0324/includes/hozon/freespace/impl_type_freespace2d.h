/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FREESPACE_IMPL_TYPE_FREESPACE2D_H
#define HOZON_FREESPACE_IMPL_TYPE_FREESPACE2D_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "hozon/composite/impl_type_point2darray.h"
#include "hozon/common/impl_type_commontime.h"

namespace hozon {
namespace freespace {
struct FreeSpace2D {
    ::Int32 spaceSeq;
    ::UInt8 type;
    ::String sensorName;
    ::hozon::composite::Point2DArray points;
    ::hozon::common::CommonTime timeCreation;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(spaceSeq);
        fun(type);
        fun(sensorName);
        fun(points);
        fun(timeCreation);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(spaceSeq);
        fun(type);
        fun(sensorName);
        fun(points);
        fun(timeCreation);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("spaceSeq", spaceSeq);
        fun("type", type);
        fun("sensorName", sensorName);
        fun("points", points);
        fun("timeCreation", timeCreation);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("spaceSeq", spaceSeq);
        fun("type", type);
        fun("sensorName", sensorName);
        fun("points", points);
        fun("timeCreation", timeCreation);
    }

    bool operator==(const ::hozon::freespace::FreeSpace2D& t) const
    {
        return (spaceSeq == t.spaceSeq) && (type == t.type) && (sensorName == t.sensorName) && (points == t.points) && (timeCreation == t.timeCreation);
    }
};
} // namespace freespace
} // namespace hozon


#endif // HOZON_FREESPACE_IMPL_TYPE_FREESPACE2D_H
