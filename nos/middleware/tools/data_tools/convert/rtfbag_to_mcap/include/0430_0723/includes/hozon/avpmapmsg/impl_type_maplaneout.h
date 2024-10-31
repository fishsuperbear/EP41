/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_AVPMAPMSG_IMPL_TYPE_MAPLANEOUT_H
#define HOZON_AVPMAPMSG_IMPL_TYPE_MAPLANEOUT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_point3darray.h"

namespace hozon {
namespace avpmapmsg {
struct MapLaneOut {
    ::UInt32 ID;
    ::UInt8 type;
    ::hozon::composite::Point3DArray points;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ID);
        fun(type);
        fun(points);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ID);
        fun(type);
        fun(points);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ID", ID);
        fun("type", type);
        fun("points", points);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ID", ID);
        fun("type", type);
        fun("points", points);
    }

    bool operator==(const ::hozon::avpmapmsg::MapLaneOut& t) const
    {
        return (ID == t.ID) && (type == t.type) && (points == t.points);
    }
};
} // namespace avpmapmsg
} // namespace hozon


#endif // HOZON_AVPMAPMSG_IMPL_TYPE_MAPLANEOUT_H
