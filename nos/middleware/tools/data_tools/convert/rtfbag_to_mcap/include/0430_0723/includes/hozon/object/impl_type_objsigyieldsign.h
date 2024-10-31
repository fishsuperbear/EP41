/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJSIGYIELDSIGN_H
#define HOZON_OBJECT_IMPL_TYPE_OBJSIGYIELDSIGN_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/composite/impl_type_point2iarrayarray.h"

namespace hozon {
namespace object {
struct ObjSigYieldSign {
    ::UInt32 objectID;
    ::hozon::composite::Point2IArrayArray stopline;

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
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(objectID);
        fun(stopline);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("objectID", objectID);
        fun("stopline", stopline);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("objectID", objectID);
        fun("stopline", stopline);
    }

    bool operator==(const ::hozon::object::ObjSigYieldSign& t) const
    {
        return (objectID == t.objectID) && (stopline == t.stopline);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJSIGYIELDSIGN_H
