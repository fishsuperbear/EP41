/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_RADARTRACKDATA_H
#define HOZON_SENSORS_IMPL_TYPE_RADARTRACKDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/sensors/impl_type_radarmodeinfo.h"
#include "impl_type_float.h"
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_point.h"

namespace hozon {
namespace sensors {
struct RadarTrackData {
    ::UInt32 id;
    ::hozon::sensors::RadarModeInfo position;
    ::hozon::sensors::RadarModeInfo velocity;
    ::hozon::sensors::RadarModeInfo acceleration;
    ::Float rcs;
    ::Float snr;
    ::Float existProbability;
    ::UInt8 movProperty;
    ::UInt8 trackType;
    ::UInt8 objObstacleProb;
    ::UInt8 measState;
    ::hozon::composite::Point sizeLWH;
    ::Float orientAgl;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(position);
        fun(velocity);
        fun(acceleration);
        fun(rcs);
        fun(snr);
        fun(existProbability);
        fun(movProperty);
        fun(trackType);
        fun(objObstacleProb);
        fun(measState);
        fun(sizeLWH);
        fun(orientAgl);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(position);
        fun(velocity);
        fun(acceleration);
        fun(rcs);
        fun(snr);
        fun(existProbability);
        fun(movProperty);
        fun(trackType);
        fun(objObstacleProb);
        fun(measState);
        fun(sizeLWH);
        fun(orientAgl);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("position", position);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
        fun("rcs", rcs);
        fun("snr", snr);
        fun("existProbability", existProbability);
        fun("movProperty", movProperty);
        fun("trackType", trackType);
        fun("objObstacleProb", objObstacleProb);
        fun("measState", measState);
        fun("sizeLWH", sizeLWH);
        fun("orientAgl", orientAgl);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("position", position);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
        fun("rcs", rcs);
        fun("snr", snr);
        fun("existProbability", existProbability);
        fun("movProperty", movProperty);
        fun("trackType", trackType);
        fun("objObstacleProb", objObstacleProb);
        fun("measState", measState);
        fun("sizeLWH", sizeLWH);
        fun("orientAgl", orientAgl);
    }

    bool operator==(const ::hozon::sensors::RadarTrackData& t) const
    {
        return (id == t.id) && (position == t.position) && (velocity == t.velocity) && (acceleration == t.acceleration) && (fabs(static_cast<double>(rcs - t.rcs)) < DBL_EPSILON) && (fabs(static_cast<double>(snr - t.snr)) < DBL_EPSILON) && (fabs(static_cast<double>(existProbability - t.existProbability)) < DBL_EPSILON) && (movProperty == t.movProperty) && (trackType == t.trackType) && (objObstacleProb == t.objObstacleProb) && (measState == t.measState) && (sizeLWH == t.sizeLWH) && (fabs(static_cast<double>(orientAgl - t.orientAgl)) < DBL_EPSILON);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_RADARTRACKDATA_H
