/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_AVPMAPMSG_IMPL_TYPE_LANEDETECTIONOUT_H
#define HOZON_AVPMAPMSG_IMPL_TYPE_LANEDETECTIONOUT_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "hozon/composite/impl_type_point3darray.h"
#include "hozon/composite/impl_type_point2darray.h"
#include "hozon/common/impl_type_commontime.h"
#include "hozon/avpmapmsg/impl_type_imagelanefitinfo.h"

namespace hozon {
namespace avpmapmsg {
struct LaneDetectionOut {
    ::Int32 lanelineSeq;
    ::UInt8 cls;
    ::Float typeConfidence;
    ::UInt8 color;
    ::Float colorConfidence;
    ::Float laneLineWidth;
    ::hozon::composite::Point3DArray keyPointVRF;
    ::hozon::composite::Point3DArray pointVehicleCoord;
    ::hozon::composite::Point2DArray pointImageCoord;
    ::hozon::composite::Point2DArray fitPointImageCoord;
    ::hozon::common::CommonTime timeCreation;
    ::hozon::avpmapmsg::imageLaneFitInfo imageLaneFit;
    ::hozon::avpmapmsg::imageLaneFitInfo laneFit;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(lanelineSeq);
        fun(cls);
        fun(typeConfidence);
        fun(color);
        fun(colorConfidence);
        fun(laneLineWidth);
        fun(keyPointVRF);
        fun(pointVehicleCoord);
        fun(pointImageCoord);
        fun(fitPointImageCoord);
        fun(timeCreation);
        fun(imageLaneFit);
        fun(laneFit);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(lanelineSeq);
        fun(cls);
        fun(typeConfidence);
        fun(color);
        fun(colorConfidence);
        fun(laneLineWidth);
        fun(keyPointVRF);
        fun(pointVehicleCoord);
        fun(pointImageCoord);
        fun(fitPointImageCoord);
        fun(timeCreation);
        fun(imageLaneFit);
        fun(laneFit);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("lanelineSeq", lanelineSeq);
        fun("cls", cls);
        fun("typeConfidence", typeConfidence);
        fun("color", color);
        fun("colorConfidence", colorConfidence);
        fun("laneLineWidth", laneLineWidth);
        fun("keyPointVRF", keyPointVRF);
        fun("pointVehicleCoord", pointVehicleCoord);
        fun("pointImageCoord", pointImageCoord);
        fun("fitPointImageCoord", fitPointImageCoord);
        fun("timeCreation", timeCreation);
        fun("imageLaneFit", imageLaneFit);
        fun("laneFit", laneFit);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("lanelineSeq", lanelineSeq);
        fun("cls", cls);
        fun("typeConfidence", typeConfidence);
        fun("color", color);
        fun("colorConfidence", colorConfidence);
        fun("laneLineWidth", laneLineWidth);
        fun("keyPointVRF", keyPointVRF);
        fun("pointVehicleCoord", pointVehicleCoord);
        fun("pointImageCoord", pointImageCoord);
        fun("fitPointImageCoord", fitPointImageCoord);
        fun("timeCreation", timeCreation);
        fun("imageLaneFit", imageLaneFit);
        fun("laneFit", laneFit);
    }

    bool operator==(const ::hozon::avpmapmsg::LaneDetectionOut& t) const
    {
        return (lanelineSeq == t.lanelineSeq) && (cls == t.cls) && (fabs(static_cast<double>(typeConfidence - t.typeConfidence)) < DBL_EPSILON) && (color == t.color) && (fabs(static_cast<double>(colorConfidence - t.colorConfidence)) < DBL_EPSILON) && (fabs(static_cast<double>(laneLineWidth - t.laneLineWidth)) < DBL_EPSILON) && (keyPointVRF == t.keyPointVRF) && (pointVehicleCoord == t.pointVehicleCoord) && (pointImageCoord == t.pointImageCoord) && (fitPointImageCoord == t.fitPointImageCoord) && (timeCreation == t.timeCreation) && (imageLaneFit == t.imageLaneFit) && (laneFit == t.laneFit);
    }
};
} // namespace avpmapmsg
} // namespace hozon


#endif // HOZON_AVPMAPMSG_IMPL_TYPE_LANEDETECTIONOUT_H
