/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FREESPACE_IMPL_TYPE_LOCELEMENT_H
#define HOZON_FREESPACE_IMPL_TYPE_LOCELEMENT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "impl_type_float.h"
#include "hozon/object/impl_type_rect3d.h"
#include "impl_type_uint16.h"
#include "hozon/composite/impl_type_point4darray.h"
#include "hozon/composite/impl_type_point3darrayarray.h"

namespace hozon {
namespace freespace {
struct locElement {
    ::UInt8 type;
    ::UInt8 subType;
    ::UInt32 objectID;
    ::Float confidence;
    ::UInt8 color;
    ::Float colorConfidence;
    ::hozon::object::Rect3D rect;
    ::UInt16 detCamSensor;
    ::hozon::composite::Point4DArray keypoint;
    ::hozon::composite::Point3DArrayArray Polygon;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(type);
        fun(subType);
        fun(objectID);
        fun(confidence);
        fun(color);
        fun(colorConfidence);
        fun(rect);
        fun(detCamSensor);
        fun(keypoint);
        fun(Polygon);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(type);
        fun(subType);
        fun(objectID);
        fun(confidence);
        fun(color);
        fun(colorConfidence);
        fun(rect);
        fun(detCamSensor);
        fun(keypoint);
        fun(Polygon);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("type", type);
        fun("subType", subType);
        fun("objectID", objectID);
        fun("confidence", confidence);
        fun("color", color);
        fun("colorConfidence", colorConfidence);
        fun("rect", rect);
        fun("detCamSensor", detCamSensor);
        fun("keypoint", keypoint);
        fun("Polygon", Polygon);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("type", type);
        fun("subType", subType);
        fun("objectID", objectID);
        fun("confidence", confidence);
        fun("color", color);
        fun("colorConfidence", colorConfidence);
        fun("rect", rect);
        fun("detCamSensor", detCamSensor);
        fun("keypoint", keypoint);
        fun("Polygon", Polygon);
    }

    bool operator==(const ::hozon::freespace::locElement& t) const
    {
        return (type == t.type) && (subType == t.subType) && (objectID == t.objectID) && (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON) && (color == t.color) && (fabs(static_cast<double>(colorConfidence - t.colorConfidence)) < DBL_EPSILON) && (rect == t.rect) && (detCamSensor == t.detCamSensor) && (keypoint == t.keypoint) && (Polygon == t.Polygon);
    }
};
} // namespace freespace
} // namespace hozon


#endif // HOZON_FREESPACE_IMPL_TYPE_LOCELEMENT_H
