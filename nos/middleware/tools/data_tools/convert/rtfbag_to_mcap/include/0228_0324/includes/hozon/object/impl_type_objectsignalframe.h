/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJECTSIGNALFRAME_H
#define HOZON_OBJECT_IMPL_TYPE_OBJECTSIGNALFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/object/impl_type_objsigroadmarkvector.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace object {
struct ObjectSignalFrame {
    ::hozon::common::CommonHeader header;
    ::hozon::object::ObjSigRoadMarkVector roadMarks;
    ::UInt8 fusionType;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(roadMarks);
        fun(fusionType);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(roadMarks);
        fun(fusionType);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("roadMarks", roadMarks);
        fun("fusionType", fusionType);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("roadMarks", roadMarks);
        fun("fusionType", fusionType);
    }

    bool operator==(const ::hozon::object::ObjectSignalFrame& t) const
    {
        return (header == t.header) && (roadMarks == t.roadMarks) && (fusionType == t.fusionType);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJECTSIGNALFRAME_H
