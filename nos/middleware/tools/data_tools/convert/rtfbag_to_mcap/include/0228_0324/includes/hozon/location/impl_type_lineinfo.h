/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_LINEINFO_H
#define HOZON_LOCATION_IMPL_TYPE_LINEINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/location/impl_type_linepointvec.h"

namespace hozon {
namespace location {
struct LineInfo {
    ::UInt32 idLane;
    ::hozon::location::LinePointVec points;
    ::UInt32 idLine;

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
        fun(idLine);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(idLane);
        fun(points);
        fun(idLine);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("idLane", idLane);
        fun("points", points);
        fun("idLine", idLine);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("idLane", idLane);
        fun("points", points);
        fun("idLine", idLine);
    }

    bool operator==(const ::hozon::location::LineInfo& t) const
    {
        return (idLane == t.idLane) && (points == t.points) && (idLine == t.idLine);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_LINEINFO_H
