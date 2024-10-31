/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_AVPMAPMSG_IMPL_TYPE_MAPMARKEROUT_H
#define HOZON_AVPMAPMSG_IMPL_TYPE_MAPMARKEROUT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "hozon/object/impl_type_rect3d.h"
#include "hozon/composite/impl_type_point3darray.h"
#include "hozon/composite/impl_type_point3darrayarray.h"

namespace hozon {
namespace avpmapmsg {
struct MapMarkerOut {
    ::UInt32 objectID;
    ::UInt8 type;
    ::UInt8 subType;
    ::Float confidence;
    ::UInt8 color;
    ::Float colorConfidence;
    ::UInt32 detCamSensor;
    ::hozon::object::Rect3D rect;
    ::hozon::composite::Point3DArray keypoint;
    ::hozon::composite::Point3DArrayArray polygon;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(objectID);
        fun(type);
        fun(subType);
        fun(confidence);
        fun(color);
        fun(colorConfidence);
        fun(detCamSensor);
        fun(rect);
        fun(keypoint);
        fun(polygon);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(objectID);
        fun(type);
        fun(subType);
        fun(confidence);
        fun(color);
        fun(colorConfidence);
        fun(detCamSensor);
        fun(rect);
        fun(keypoint);
        fun(polygon);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("objectID", objectID);
        fun("type", type);
        fun("subType", subType);
        fun("confidence", confidence);
        fun("color", color);
        fun("colorConfidence", colorConfidence);
        fun("detCamSensor", detCamSensor);
        fun("rect", rect);
        fun("keypoint", keypoint);
        fun("polygon", polygon);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("objectID", objectID);
        fun("type", type);
        fun("subType", subType);
        fun("confidence", confidence);
        fun("color", color);
        fun("colorConfidence", colorConfidence);
        fun("detCamSensor", detCamSensor);
        fun("rect", rect);
        fun("keypoint", keypoint);
        fun("polygon", polygon);
    }

    bool operator==(const ::hozon::avpmapmsg::MapMarkerOut& t) const
    {
        return (objectID == t.objectID) && (type == t.type) && (subType == t.subType) && (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON) && (color == t.color) && (fabs(static_cast<double>(colorConfidence - t.colorConfidence)) < DBL_EPSILON) && (detCamSensor == t.detCamSensor) && (rect == t.rect) && (keypoint == t.keypoint) && (polygon == t.polygon);
    }
};
} // namespace avpmapmsg
} // namespace hozon


#endif // HOZON_AVPMAPMSG_IMPL_TYPE_MAPMARKEROUT_H
