/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_LANEINFO_H
#define HOZON_LOCATION_IMPL_TYPE_LANEINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/location/impl_type_lanepointvec.h"

namespace hozon {
namespace location {
struct LaneInfo {
    ::UInt32 idLane;
    ::hozon::location::LanePointVec points;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(idLane);
        fun(points);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(idLane);
        fun(points);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("idLane", idLane);
        fun("points", points);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("idLane", idLane);
        fun("points", points);
    }

    bool operator==(const ::hozon::location::LaneInfo& t) const
    {
        return (idLane == t.idLane) && (points == t.points);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_LANEINFO_H
