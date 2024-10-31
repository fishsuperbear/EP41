/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_RDETECT_IMPL_TYPE_RADARDETECT_H
#define ARA_RDETECT_IMPL_TYPE_RADARDETECT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace ara {
namespace rdetect {
struct RadarDetect {
    ::UInt8 id;
    ::UInt8 idPair;
    ::UInt8 coordinate;
    ::Float x;
    ::Float y;
    ::Float z;
    ::Float vx;
    ::Float vy;
    ::Float rcs;
    ::Float snr;
    ::Float xRms;
    ::Float yRms;
    ::Float zRms;
    ::Float vxRms;
    ::Float vyRms;
    ::Float xQuality;
    ::Float yQuality;
    ::Float zQuality;
    ::Float vxQuality;
    ::Float vyQuality;
    ::UInt8 existProbability;
    ::UInt8 falseProbability;
    ::UInt8 movProperty;
    ::UInt8 invalidState;
    ::UInt8 ambiguity;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(idPair);
        fun(coordinate);
        fun(x);
        fun(y);
        fun(z);
        fun(vx);
        fun(vy);
        fun(rcs);
        fun(snr);
        fun(xRms);
        fun(yRms);
        fun(zRms);
        fun(vxRms);
        fun(vyRms);
        fun(xQuality);
        fun(yQuality);
        fun(zQuality);
        fun(vxQuality);
        fun(vyQuality);
        fun(existProbability);
        fun(falseProbability);
        fun(movProperty);
        fun(invalidState);
        fun(ambiguity);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(idPair);
        fun(coordinate);
        fun(x);
        fun(y);
        fun(z);
        fun(vx);
        fun(vy);
        fun(rcs);
        fun(snr);
        fun(xRms);
        fun(yRms);
        fun(zRms);
        fun(vxRms);
        fun(vyRms);
        fun(xQuality);
        fun(yQuality);
        fun(zQuality);
        fun(vxQuality);
        fun(vyQuality);
        fun(existProbability);
        fun(falseProbability);
        fun(movProperty);
        fun(invalidState);
        fun(ambiguity);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("idPair", idPair);
        fun("coordinate", coordinate);
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("vx", vx);
        fun("vy", vy);
        fun("rcs", rcs);
        fun("snr", snr);
        fun("xRms", xRms);
        fun("yRms", yRms);
        fun("zRms", zRms);
        fun("vxRms", vxRms);
        fun("vyRms", vyRms);
        fun("xQuality", xQuality);
        fun("yQuality", yQuality);
        fun("zQuality", zQuality);
        fun("vxQuality", vxQuality);
        fun("vyQuality", vyQuality);
        fun("existProbability", existProbability);
        fun("falseProbability", falseProbability);
        fun("movProperty", movProperty);
        fun("invalidState", invalidState);
        fun("ambiguity", ambiguity);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("idPair", idPair);
        fun("coordinate", coordinate);
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("vx", vx);
        fun("vy", vy);
        fun("rcs", rcs);
        fun("snr", snr);
        fun("xRms", xRms);
        fun("yRms", yRms);
        fun("zRms", zRms);
        fun("vxRms", vxRms);
        fun("vyRms", vyRms);
        fun("xQuality", xQuality);
        fun("yQuality", yQuality);
        fun("zQuality", zQuality);
        fun("vxQuality", vxQuality);
        fun("vyQuality", vyQuality);
        fun("existProbability", existProbability);
        fun("falseProbability", falseProbability);
        fun("movProperty", movProperty);
        fun("invalidState", invalidState);
        fun("ambiguity", ambiguity);
    }

    bool operator==(const ::ara::rdetect::RadarDetect& t) const
    {
        return (id == t.id) && (idPair == t.idPair) && (coordinate == t.coordinate) && (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(vx - t.vx)) < DBL_EPSILON) && (fabs(static_cast<double>(vy - t.vy)) < DBL_EPSILON) && (fabs(static_cast<double>(rcs - t.rcs)) < DBL_EPSILON) && (fabs(static_cast<double>(snr - t.snr)) < DBL_EPSILON) && (fabs(static_cast<double>(xRms - t.xRms)) < DBL_EPSILON) && (fabs(static_cast<double>(yRms - t.yRms)) < DBL_EPSILON) && (fabs(static_cast<double>(zRms - t.zRms)) < DBL_EPSILON) && (fabs(static_cast<double>(vxRms - t.vxRms)) < DBL_EPSILON) && (fabs(static_cast<double>(vyRms - t.vyRms)) < DBL_EPSILON) && (fabs(static_cast<double>(xQuality - t.xQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(yQuality - t.yQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(zQuality - t.zQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(vxQuality - t.vxQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(vyQuality - t.vyQuality)) < DBL_EPSILON) && (existProbability == t.existProbability) && (falseProbability == t.falseProbability) && (movProperty == t.movProperty) && (invalidState == t.invalidState) && (ambiguity == t.ambiguity);
    }
};
} // namespace rdetect
} // namespace ara


#endif // ARA_RDETECT_IMPL_TYPE_RADARDETECT_H
