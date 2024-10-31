/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJSIGCROSSWALK_H
#define HOZON_OBJECT_IMPL_TYPE_OBJSIGCROSSWALK_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/composite/impl_type_polygon.h"

namespace hozon {
namespace object {
struct ObjSigCrosswalk {
    ::UInt32 objectID;
    ::hozon::composite::Polygon polygon;

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
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(objectID);
        fun(polygon);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("objectID", objectID);
        fun("polygon", polygon);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("objectID", objectID);
        fun("polygon", polygon);
    }

    bool operator==(const ::hozon::object::ObjSigCrosswalk& t) const
    {
        return (objectID == t.objectID) && (polygon == t.polygon);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJSIGCROSSWALK_H
