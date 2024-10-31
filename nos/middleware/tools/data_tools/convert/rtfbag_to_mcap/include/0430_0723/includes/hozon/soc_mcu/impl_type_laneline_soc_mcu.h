/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_LANELINE_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_LANELINE_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"
#include "impl_type_float.h"
#include "impl_type_uint8.h"
#include "hozon/soc_mcu/impl_type_point3f_soc_mcu.h"
#include "hozon/common/impl_type_commontime.h"
#include "hozon/laneline/impl_type_lanelinefit.h"

namespace hozon {
namespace soc_mcu {
struct LaneLine_soc_mcu {
    ::Int32 lanSeq;
    ::Float geoConfidence;
    ::UInt8 cls;
    ::Float typeConfidence;
    ::UInt8 color;
    ::Float colorConfidence;
    ::Float laneLineWidth;
    ::hozon::soc_mcu::Point3f_soc_mcu keyPointVRF;
    ::hozon::common::CommonTime timeCreation;
    ::hozon::laneline::LaneLineFit laneFits;

    static bool IsPlane()
    {
        return true;
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
    }

    bool operator==(const ::hozon::soc_mcu::LaneLine_soc_mcu& t) const
    {
        return (lanSeq == t.lanSeq) && (fabs(static_cast<double>(geoConfidence - t.geoConfidence)) < DBL_EPSILON) && (cls == t.cls) && (fabs(static_cast<double>(typeConfidence - t.typeConfidence)) < DBL_EPSILON) && (color == t.color) && (fabs(static_cast<double>(colorConfidence - t.colorConfidence)) < DBL_EPSILON) && (fabs(static_cast<double>(laneLineWidth - t.laneLineWidth)) < DBL_EPSILON) && (keyPointVRF == t.keyPointVRF) && (timeCreation == t.timeCreation) && (laneFits == t.laneFits);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_LANELINE_SOC_MCU_H
