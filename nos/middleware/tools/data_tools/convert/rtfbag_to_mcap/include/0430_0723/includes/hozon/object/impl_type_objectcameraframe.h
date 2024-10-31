/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJECTCAMERAFRAME_H
#define HOZON_OBJECT_IMPL_TYPE_OBJECTCAMERAFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint8.h"
#include "hozon/object/impl_type_objectcamvector3d.h"
#include "hozon/object/impl_type_objectcamvector2d.h"
#include "impl_type_float.h"

namespace hozon {
namespace object {
struct ObjectCameraFrame {
    ::hozon::common::CommonHeader header;
    ::UInt8 sensorStatus;
    ::hozon::object::ObjectCamVector3D object3d;
    ::hozon::object::ObjectCamVector2D object2d;
    ::Float lightIntensity;
    ::hozon::object::ObjectCamVector3D detectedOut3d;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(sensorStatus);
        fun(object3d);
        fun(object2d);
        fun(lightIntensity);
        fun(detectedOut3d);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(sensorStatus);
        fun(object3d);
        fun(object2d);
        fun(lightIntensity);
        fun(detectedOut3d);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("sensorStatus", sensorStatus);
        fun("object3d", object3d);
        fun("object2d", object2d);
        fun("lightIntensity", lightIntensity);
        fun("detectedOut3d", detectedOut3d);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("sensorStatus", sensorStatus);
        fun("object3d", object3d);
        fun("object2d", object2d);
        fun("lightIntensity", lightIntensity);
        fun("detectedOut3d", detectedOut3d);
    }

    bool operator==(const ::hozon::object::ObjectCameraFrame& t) const
    {
        return (header == t.header) && (sensorStatus == t.sensorStatus) && (object3d == t.object3d) && (object2d == t.object2d) && (fabs(static_cast<double>(lightIntensity - t.lightIntensity)) < DBL_EPSILON) && (detectedOut3d == t.detectedOut3d);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJECTCAMERAFRAME_H
