/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJSIGSTOPSIGN_H
#define HOZON_OBJECT_IMPL_TYPE_OBJSIGSTOPSIGN_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/composite/impl_type_point2iarrayarray.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace object {
struct ObjSigStopSign {
    ::UInt32 objectID;
    ::hozon::composite::Point2IArrayArray stopline;
    ::UInt8 stoptype;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(objectID);
        fun(stopline);
        fun(stoptype);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(objectID);
        fun(stopline);
        fun(stoptype);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("objectID", objectID);
        fun("stopline", stopline);
        fun("stoptype", stoptype);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("objectID", objectID);
        fun("stopline", stopline);
        fun("stoptype", stoptype);
    }

    bool operator==(const ::hozon::object::ObjSigStopSign& t) const
    {
        return (objectID == t.objectID) && (stopline == t.stopline) && (stoptype == t.stoptype);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJSIGSTOPSIGN_H
