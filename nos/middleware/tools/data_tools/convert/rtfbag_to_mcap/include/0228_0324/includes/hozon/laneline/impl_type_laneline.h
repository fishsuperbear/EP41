/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LANELINE_IMPL_TYPE_LANELINE_H
#define HOZON_LANELINE_IMPL_TYPE_LANELINE_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"
#include "impl_type_float.h"
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_point3darray.h"
#include "hozon/common/impl_type_commontime.h"
#include "hozon/laneline/impl_type_lanelinefit.h"
#include "hozon/composite/impl_type_point2darray.h"

namespace hozon {
namespace laneline {
struct LaneLine {
    ::Int32 lanSeq;
    ::Float geoConfidence;
    ::UInt8 cls;
    ::Float typeConfidence;
    ::UInt8 color;
    ::Float colorConfidence;
    ::Float laneLineWidth;
    ::hozon::composite::Point3DArray keyPointVRF;
    ::hozon::common::CommonTime timeCreation;
    ::hozon::laneline::LaneLineFit laneFits;
    ::hozon::composite::Point3DArray pointVehicleCoord;
    ::hozon::composite::Point2DArray pointImageCoord;
    ::hozon::composite::Point2DArray fitPointImageCoord;
    ::hozon::laneline::LaneLineFit imageLaneFit;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(lanSeq);
        fun(geoConfidence);
        fun(cls);
        fun(typeConfidence);
        fun(color);
        fun(colorConfidence);
        fun(laneLineWidth);
        fun(keyPointVRF);
        fun(timeCreation);
        fun(laneFits);
        fun(pointVehicleCoord);
        fun(pointImageCoord);
        fun(fitPointImageCoord);
        fun(imageLaneFit);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(lanSeq);
        fun(geoConfidence);
        fun(cls);
        fun(typeConfidence);
        fun(color);
        fun(colorConfidence);
        fun(laneLineWidth);
        fun(keyPointVRF);
        fun(timeCreation);
        fun(laneFits);
        fun(pointVehicleCoord);
        fun(pointImageCoord);
        fun(fitPointImageCoord);
        fun(imageLaneFit);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("lanSeq", lanSeq);
        fun("geoConfidence", geoConfidence);
        fun("cls", cls);
        fun("typeConfidence", typeConfidence);
        fun("color", color);
        fun("colorConfidence", colorConfidence);
        fun("laneLineWidth", laneLineWidth);
        fun("keyPointVRF", keyPointVRF);
        fun("timeCreation", timeCreation);
        fun("laneFits", laneFits);
        fun("pointVehicleCoord", pointVehicleCoord);
        fun("pointImageCoord", pointImageCoord);
        fun("fitPointImageCoord", fitPointImageCoord);
        fun("imageLaneFit", imageLaneFit);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("lanSeq", lanSeq);
        fun("geoConfidence", geoConfidence);
        fun("cls", cls);
        fun("typeConfidence", typeConfidence);
        fun("color", color);
        fun("colorConfidence", colorConfidence);
        fun("laneLineWidth", laneLineWidth);
        fun("keyPointVRF", keyPointVRF);
        fun("timeCreation", timeCreation);
        fun("laneFits", laneFits);
        fun("pointVehicleCoord", pointVehicleCoord);
        fun("pointImageCoord", pointImageCoord);
        fun("fitPointImageCoord", fitPointImageCoord);
        fun("imageLaneFit", imageLaneFit);
    }

    bool operator==(const ::hozon::laneline::LaneLine& t) const
    {
        return (lanSeq == t.lanSeq) && (fabs(static_cast<double>(geoConfidence - t.geoConfidence)) < DBL_EPSILON) && (cls == t.cls) && (fabs(static_cast<double>(typeConfidence - t.typeConfidence)) < DBL_EPSILON) && (color == t.color) && (fabs(static_cast<double>(colorConfidence - t.colorConfidence)) < DBL_EPSILON) && (fabs(static_cast<double>(laneLineWidth - t.laneLineWidth)) < DBL_EPSILON) && (keyPointVRF == t.keyPointVRF) && (timeCreation == t.timeCreation) && (laneFits == t.laneFits) && (pointVehicleCoord == t.pointVehicleCoord) && (pointImageCoord == t.pointImageCoord) && (fitPointImageCoord == t.fitPointImageCoord) && (imageLaneFit == t.imageLaneFit);
    }
};
} // namespace laneline
} // namespace hozon


#endif // HOZON_LANELINE_IMPL_TYPE_LANELINE_H
