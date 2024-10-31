/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_OBJECT2D_H
#define ADSFI_IMPL_TYPE_OBJECT2D_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_double.h"
#include "impl_type_uint32.h"
#include "ara/common/impl_type_commontime.h"
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "impl_type_pointxydouble.h"
#include "impl_type_rect2d.h"

namespace adsfi {
struct Object2d {
    ::ara::common::CommonHeader header;
    ::Double existProb;
    ::UInt32 id;
    ::ara::common::CommonTime timeCreation;
    ::UInt8 cls;
    ::String clsDescription;
    ::Double clsConfidence;
    ::PointxyDouble velocity;
    ::PointxyDouble absVelocity;
    ::UInt8 cipvFlag;
    ::PointxyDouble acceleration;
    ::UInt8 cameraStatus;
    ::String coordinate;
    ::UInt8 blinkerStatus;
    ::Rect2d rect2d;

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
        fun(cipvFlag);
        fun(acceleration);
        fun(cameraStatus);
        fun(coordinate);
        fun(blinkerStatus);
        fun(rect2d);
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
        fun(cipvFlag);
        fun(acceleration);
        fun(cameraStatus);
        fun(coordinate);
        fun(blinkerStatus);
        fun(rect2d);
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
        fun("cipvFlag", cipvFlag);
        fun("acceleration", acceleration);
        fun("cameraStatus", cameraStatus);
        fun("coordinate", coordinate);
        fun("blinkerStatus", blinkerStatus);
        fun("rect2d", rect2d);
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
        fun("cipvFlag", cipvFlag);
        fun("acceleration", acceleration);
        fun("cameraStatus", cameraStatus);
        fun("coordinate", coordinate);
        fun("blinkerStatus", blinkerStatus);
        fun("rect2d", rect2d);
    }

    bool operator==(const ::adsfi::Object2d& t) const
    {
        return (header == t.header) && (fabs(static_cast<double>(existProb - t.existProb)) < DBL_EPSILON) && (id == t.id) && (timeCreation == t.timeCreation) && (cls == t.cls) && (clsDescription == t.clsDescription) && (fabs(static_cast<double>(clsConfidence - t.clsConfidence)) < DBL_EPSILON) && (velocity == t.velocity) && (absVelocity == t.absVelocity) && (cipvFlag == t.cipvFlag) && (acceleration == t.acceleration) && (cameraStatus == t.cameraStatus) && (coordinate == t.coordinate) && (blinkerStatus == t.blinkerStatus) && (rect2d == t.rect2d);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_OBJECT2D_H
