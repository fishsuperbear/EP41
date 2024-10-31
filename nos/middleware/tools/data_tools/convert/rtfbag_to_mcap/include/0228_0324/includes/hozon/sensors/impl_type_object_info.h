/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_OBJECT_INFO_H
#define HOZON_SENSORS_IMPL_TYPE_OBJECT_INFO_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_tpointinfo.h"
#include "hozon/composite/impl_type_uint16vector.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace sensors {
struct Object_Info {
    ::hozon::composite::tPointInfo ObstaclePoint;
    ::hozon::composite::Uint16Vector wDistance;
    ::UInt8 wTracker_age;
    ::UInt8 cTracker_status;
    ::UInt8 cTracker_ID;
    ::UInt8 cTracker_type;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ObstaclePoint);
        fun(wDistance);
        fun(wTracker_age);
        fun(cTracker_status);
        fun(cTracker_ID);
        fun(cTracker_type);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ObstaclePoint);
        fun(wDistance);
        fun(wTracker_age);
        fun(cTracker_status);
        fun(cTracker_ID);
        fun(cTracker_type);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ObstaclePoint", ObstaclePoint);
        fun("wDistance", wDistance);
        fun("wTracker_age", wTracker_age);
        fun("cTracker_status", cTracker_status);
        fun("cTracker_ID", cTracker_ID);
        fun("cTracker_type", cTracker_type);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ObstaclePoint", ObstaclePoint);
        fun("wDistance", wDistance);
        fun("wTracker_age", wTracker_age);
        fun("cTracker_status", cTracker_status);
        fun("cTracker_ID", cTracker_ID);
        fun("cTracker_type", cTracker_type);
    }

    bool operator==(const ::hozon::sensors::Object_Info& t) const
    {
        return (ObstaclePoint == t.ObstaclePoint) && (wDistance == t.wDistance) && (wTracker_age == t.wTracker_age) && (cTracker_status == t.cTracker_status) && (cTracker_ID == t.cTracker_ID) && (cTracker_type == t.cTracker_type);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_OBJECT_INFO_H
