/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_GNSSHEADINGINFOFRAME_H
#define HOZON_SENSORS_IMPL_TYPE_GNSSHEADINGINFOFRAME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace hozon {
namespace sensors {
struct GnssHeadingInfoFrame {
    ::UInt8 svs;
    ::UInt8 solnSVs;
    ::UInt8 posType;
    ::Float length;
    ::Float heading;
    ::Float pitch;
    ::Float hdgStd;
    ::Float pitchStd;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(svs);
        fun(solnSVs);
        fun(posType);
        fun(length);
        fun(heading);
        fun(pitch);
        fun(hdgStd);
        fun(pitchStd);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(svs);
        fun(solnSVs);
        fun(posType);
        fun(length);
        fun(heading);
        fun(pitch);
        fun(hdgStd);
        fun(pitchStd);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("svs", svs);
        fun("solnSVs", solnSVs);
        fun("posType", posType);
        fun("length", length);
        fun("heading", heading);
        fun("pitch", pitch);
        fun("hdgStd", hdgStd);
        fun("pitchStd", pitchStd);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("svs", svs);
        fun("solnSVs", solnSVs);
        fun("posType", posType);
        fun("length", length);
        fun("heading", heading);
        fun("pitch", pitch);
        fun("hdgStd", hdgStd);
        fun("pitchStd", pitchStd);
    }

    bool operator==(const ::hozon::sensors::GnssHeadingInfoFrame& t) const
    {
        return (svs == t.svs) && (solnSVs == t.solnSVs) && (posType == t.posType) && (fabs(static_cast<double>(length - t.length)) < DBL_EPSILON) && (fabs(static_cast<double>(heading - t.heading)) < DBL_EPSILON) && (fabs(static_cast<double>(pitch - t.pitch)) < DBL_EPSILON) && (fabs(static_cast<double>(hdgStd - t.hdgStd)) < DBL_EPSILON) && (fabs(static_cast<double>(pitchStd - t.pitchStd)) < DBL_EPSILON);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_GNSSHEADINGINFOFRAME_H
