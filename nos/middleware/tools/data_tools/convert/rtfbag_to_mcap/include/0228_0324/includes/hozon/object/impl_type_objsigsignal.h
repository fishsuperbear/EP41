/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJSIGSIGNAL_H
#define HOZON_OBJECT_IMPL_TYPE_OBJSIGSIGNAL_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/composite/impl_type_polygon.h"
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_point2iarrayarray.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace object {
struct ObjSigSignal {
    ::UInt32 objectID;
    ::hozon::composite::Polygon polygon;
    ::UInt8 type;
    ::hozon::composite::Point2IArrayArray stopline;
    ::Boolean turn_allow;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(objectID);
        fun(polygon);
        fun(type);
        fun(stopline);
        fun(turn_allow);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(objectID);
        fun(polygon);
        fun(type);
        fun(stopline);
        fun(turn_allow);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("objectID", objectID);
        fun("polygon", polygon);
        fun("type", type);
        fun("stopline", stopline);
        fun("turn_allow", turn_allow);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("objectID", objectID);
        fun("polygon", polygon);
        fun("type", type);
        fun("stopline", stopline);
        fun("turn_allow", turn_allow);
    }

    bool operator==(const ::hozon::object::ObjSigSignal& t) const
    {
        return (objectID == t.objectID) && (polygon == t.polygon) && (type == t.type) && (stopline == t.stopline) && (turn_allow == t.turn_allow);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJSIGSIGNAL_H
