/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_RDETECT_IMPL_TYPE_RADARDETECTARRAY_H
#define ARA_RDETECT_IMPL_TYPE_RADARDETECTARRAY_H
#include <cfloat>
#include <cmath>
#include "ara/radar/impl_type_header.h"
#include "impl_type_uint8.h"
#include "ara/radar/impl_type_radarstate.h"
#include "impl_type_detectlist.h"
#include "impl_type_floatvector.h"
#include "impl_type_float.h"

namespace ara {
namespace rdetect {
struct RadarDetectArray {
    ::ara::radar::Header header;
    ::UInt8 sensorId;
    ::ara::radar::RadarState radarState;
    ::DetectList detectList;
    ::FloatVector maxDistanceOverAzimuthList;
    ::FloatVector azimuthForMaxDistanceList;
    ::FloatVector factorDistanceOverElevationList;
    ::FloatVector elevationForFactorMaxDistanceList;
    ::Float maxDistanceDueToProgram;
    ::Float minDistanceDueToProgram;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(sensorId);
        fun(radarState);
        fun(detectList);
        fun(maxDistanceOverAzimuthList);
        fun(azimuthForMaxDistanceList);
        fun(factorDistanceOverElevationList);
        fun(elevationForFactorMaxDistanceList);
        fun(maxDistanceDueToProgram);
        fun(minDistanceDueToProgram);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(sensorId);
        fun(radarState);
        fun(detectList);
        fun(maxDistanceOverAzimuthList);
        fun(azimuthForMaxDistanceList);
        fun(factorDistanceOverElevationList);
        fun(elevationForFactorMaxDistanceList);
        fun(maxDistanceDueToProgram);
        fun(minDistanceDueToProgram);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("sensorId", sensorId);
        fun("radarState", radarState);
        fun("detectList", detectList);
        fun("maxDistanceOverAzimuthList", maxDistanceOverAzimuthList);
        fun("azimuthForMaxDistanceList", azimuthForMaxDistanceList);
        fun("factorDistanceOverElevationList", factorDistanceOverElevationList);
        fun("elevationForFactorMaxDistanceList", elevationForFactorMaxDistanceList);
        fun("maxDistanceDueToProgram", maxDistanceDueToProgram);
        fun("minDistanceDueToProgram", minDistanceDueToProgram);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("sensorId", sensorId);
        fun("radarState", radarState);
        fun("detectList", detectList);
        fun("maxDistanceOverAzimuthList", maxDistanceOverAzimuthList);
        fun("azimuthForMaxDistanceList", azimuthForMaxDistanceList);
        fun("factorDistanceOverElevationList", factorDistanceOverElevationList);
        fun("elevationForFactorMaxDistanceList", elevationForFactorMaxDistanceList);
        fun("maxDistanceDueToProgram", maxDistanceDueToProgram);
        fun("minDistanceDueToProgram", minDistanceDueToProgram);
    }

    bool operator==(const ::ara::rdetect::RadarDetectArray& t) const
    {
        return (header == t.header) && (sensorId == t.sensorId) && (radarState == t.radarState) && (detectList == t.detectList) && (maxDistanceOverAzimuthList == t.maxDistanceOverAzimuthList) && (azimuthForMaxDistanceList == t.azimuthForMaxDistanceList) && (factorDistanceOverElevationList == t.factorDistanceOverElevationList) && (elevationForFactorMaxDistanceList == t.elevationForFactorMaxDistanceList) && (fabs(static_cast<double>(maxDistanceDueToProgram - t.maxDistanceDueToProgram)) < DBL_EPSILON) && (fabs(static_cast<double>(minDistanceDueToProgram - t.minDistanceDueToProgram)) < DBL_EPSILON);
    }
};
} // namespace rdetect
} // namespace ara


#endif // ARA_RDETECT_IMPL_TYPE_RADARDETECTARRAY_H
