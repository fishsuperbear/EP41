/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJECTFUSIONFRAME_H
#define HOZON_OBJECT_IMPL_TYPE_OBJECTFUSIONFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "hozon/object/impl_type_objectfusionvector.h"

namespace hozon {
namespace object {
struct ObjectFusionFrame {
    ::hozon::common::CommonHeader header;
    ::UInt32 locSeq;
    ::hozon::object::ObjectFusionVector object_fusion;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(locSeq);
        fun(object_fusion);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(locSeq);
        fun(object_fusion);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("object_fusion", object_fusion);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("object_fusion", object_fusion);
    }

    bool operator==(const ::hozon::object::ObjectFusionFrame& t) const
    {
        return (header == t.header) && (locSeq == t.locSeq) && (object_fusion == t.object_fusion);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJECTFUSIONFRAME_H
