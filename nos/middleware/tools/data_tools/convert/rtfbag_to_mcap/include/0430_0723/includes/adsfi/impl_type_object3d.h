/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_OBJECT3D_H
#define ADSFI_IMPL_TYPE_OBJECT3D_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_double.h"
#include "impl_type_uint32.h"
#include "ara/common/impl_type_commontime.h"
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "impl_type_point.h"
#include "impl_type_pointarray.h"
#include "impl_type_rect3d.h"

namespace adsfi {
struct Object3d {
    ::ara::common::CommonHeader header;
    ::Double existProb;
    ::UInt32 id;
    ::ara::common::CommonTime timeCreation;
    ::UInt8 cls;
    ::String clsDescription;
    ::Double clsConfidence;
    ::Point velocity;
    ::Point absVelocity;
    ::PointArray pointsInObject;
    ::UInt8 cipvFlag;
    ::Point acceleration;
    ::UInt8 cameraStatus;
    ::String coordinate;
    ::Rect3d rect3d;
    ::UInt8 blinkerStatus;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(existProb);
        fun(id);
        fun(timeCreation);
        fun(cls);
        fun(clsDescription);
        fun(clsConfidence);
        fun(velocity);
        fun(absVelocity);
        fun(pointsInObject);
        fun(cipvFlag);
        fun(acceleration);
        fun(cameraStatus);
        fun(coordinate);
        fun(rect3d);
        fun(blinkerStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(existProb);
        fun(id);
        fun(timeCreation);
        fun(cls);
        fun(clsDescription);
        fun(clsConfidence);
        fun(velocity);
        fun(absVelocity);
        fun(pointsInObject);
        fun(cipvFlag);
        fun(acceleration);
        fun(cameraStatus);
        fun(coordinate);
        fun(rect3d);
        fun(blinkerStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("existProb", existProb);
        fun("id", id);
        fun("timeCreation", timeCreation);
        fun("cls", cls);
        fun("clsDescription", clsDescription);
        fun("clsConfidence", clsConfidence);
        fun("velocity", velocity);
        fun("absVelocity", absVelocity);
        fun("pointsInObject", pointsInObject);
        fun("cipvFlag", cipvFlag);
        fun("acceleration", acceleration);
        fun("cameraStatus", cameraStatus);
        fun("coordinate", coordinate);
        fun("rect3d", rect3d);
        fun("blinkerStatus", blinkerStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("existProb", existProb);
        fun("id", id);
        fun("timeCreation", timeCreation);
        fun("cls", cls);
        fun("clsDescription", clsDescription);
        fun("clsConfidence", clsConfidence);
        fun("velocity", velocity);
        fun("absVelocity", absVelocity);
        fun("pointsInObject", pointsInObject);
        fun("cipvFlag", cipvFlag);
        fun("acceleration", acceleration);
        fun("cameraStatus", cameraStatus);
        fun("coordinate", coordinate);
        fun("rect3d", rect3d);
        fun("blinkerStatus", blinkerStatus);
    }

    bool operator==(const ::adsfi::Object3d& t) const
    {
        return (header == t.header) && (fabs(static_cast<double>(existProb - t.existProb)) < DBL_EPSILON) && (id == t.id) && (timeCreation == t.timeCreation) && (cls == t.cls) && (clsDescription == t.clsDescription) && (fabs(static_cast<double>(clsConfidence - t.clsConfidence)) < DBL_EPSILON) && (velocity == t.velocity) && (absVelocity == t.absVelocity) && (pointsInObject == t.pointsInObject) && (cipvFlag == t.cipvFlag) && (acceleration == t.acceleration) && (cameraStatus == t.cameraStatus) && (coordinate == t.coordinate) && (rect3d == t.rect3d) && (blinkerStatus == t.blinkerStatus);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_OBJECT3D_H
