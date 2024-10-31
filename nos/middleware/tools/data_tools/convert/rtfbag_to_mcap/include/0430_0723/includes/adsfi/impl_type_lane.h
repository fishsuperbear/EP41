/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_LANE_H
#define ADSFI_IMPL_TYPE_LANE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "impl_type_float.h"
#include "adsfi/impl_type_lanefitparam.h"
#include "impl_type_pointxy.h"
#include "impl_type_pointxyarray.h"
#include "impl_type_matrix3d.h"
#include "ara/common/impl_type_commontime.h"

namespace adsfi {
struct Lane {
    ::UInt32 id;
    ::UInt8 cls;
    ::String clsDescription;
    ::Float clsConfidence;
    ::adsfi::LaneFitParam laneFit;
    ::Pointxy startPoint;
    ::Pointxy endPoint;
    ::PointxyArray keyPoints;
    ::Matrix3d homographyMat;
    ::Matrix3d homographyMatInv;
    ::ara::common::CommonTime timeCreation;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(cls);
        fun(clsDescription);
        fun(clsConfidence);
        fun(laneFit);
        fun(startPoint);
        fun(endPoint);
        fun(keyPoints);
        fun(homographyMat);
        fun(homographyMatInv);
        fun(timeCreation);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(cls);
        fun(clsDescription);
        fun(clsConfidence);
        fun(laneFit);
        fun(startPoint);
        fun(endPoint);
        fun(keyPoints);
        fun(homographyMat);
        fun(homographyMatInv);
        fun(timeCreation);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("cls", cls);
        fun("clsDescription", clsDescription);
        fun("clsConfidence", clsConfidence);
        fun("laneFit", laneFit);
        fun("startPoint", startPoint);
        fun("endPoint", endPoint);
        fun("keyPoints", keyPoints);
        fun("homographyMat", homographyMat);
        fun("homographyMatInv", homographyMatInv);
        fun("timeCreation", timeCreation);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("cls", cls);
        fun("clsDescription", clsDescription);
        fun("clsConfidence", clsConfidence);
        fun("laneFit", laneFit);
        fun("startPoint", startPoint);
        fun("endPoint", endPoint);
        fun("keyPoints", keyPoints);
        fun("homographyMat", homographyMat);
        fun("homographyMatInv", homographyMatInv);
        fun("timeCreation", timeCreation);
    }

    bool operator==(const ::adsfi::Lane& t) const
    {
        return (id == t.id) && (cls == t.cls) && (clsDescription == t.clsDescription) && (fabs(static_cast<double>(clsConfidence - t.clsConfidence)) < DBL_EPSILON) && (laneFit == t.laneFit) && (startPoint == t.startPoint) && (endPoint == t.endPoint) && (keyPoints == t.keyPoints) && (homographyMat == t.homographyMat) && (homographyMatInv == t.homographyMatInv) && (timeCreation == t.timeCreation);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_LANE_H
