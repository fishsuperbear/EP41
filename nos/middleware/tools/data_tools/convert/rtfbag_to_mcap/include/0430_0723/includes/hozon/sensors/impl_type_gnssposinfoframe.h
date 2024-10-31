/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_GNSSPOSINFOFRAME_H
#define HOZON_SENSORS_IMPL_TYPE_GNSSPOSINFOFRAME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_double.h"
#include "impl_type_float.h"

namespace hozon {
namespace sensors {
struct GnssPosInfoFrame {
    ::UInt8 posType;
    ::Double latitude;
    ::Double longitude;
    ::Float undulation;
    ::Float altitude;
    ::Float latStd;
    ::Float lonStd;
    ::Float hgtStd;
    ::UInt8 svs;
    ::UInt8 solnSVs;
    ::Float diffAge;
    ::Float hdop;
    ::Float vdop;
    ::Float pdop;
    ::Float gdop;
    ::Float tdop;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(posType);
        fun(latitude);
        fun(longitude);
        fun(undulation);
        fun(altitude);
        fun(latStd);
        fun(lonStd);
        fun(hgtStd);
        fun(svs);
        fun(solnSVs);
        fun(diffAge);
        fun(hdop);
        fun(vdop);
        fun(pdop);
        fun(gdop);
        fun(tdop);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(posType);
        fun(latitude);
        fun(longitude);
        fun(undulation);
        fun(altitude);
        fun(latStd);
        fun(lonStd);
        fun(hgtStd);
        fun(svs);
        fun(solnSVs);
        fun(diffAge);
        fun(hdop);
        fun(vdop);
        fun(pdop);
        fun(gdop);
        fun(tdop);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("posType", posType);
        fun("latitude", latitude);
        fun("longitude", longitude);
        fun("undulation", undulation);
        fun("altitude", altitude);
        fun("latStd", latStd);
        fun("lonStd", lonStd);
        fun("hgtStd", hgtStd);
        fun("svs", svs);
        fun("solnSVs", solnSVs);
        fun("diffAge", diffAge);
        fun("hdop", hdop);
        fun("vdop", vdop);
        fun("pdop", pdop);
        fun("gdop", gdop);
        fun("tdop", tdop);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("posType", posType);
        fun("latitude", latitude);
        fun("longitude", longitude);
        fun("undulation", undulation);
        fun("altitude", altitude);
        fun("latStd", latStd);
        fun("lonStd", lonStd);
        fun("hgtStd", hgtStd);
        fun("svs", svs);
        fun("solnSVs", solnSVs);
        fun("diffAge", diffAge);
        fun("hdop", hdop);
        fun("vdop", vdop);
        fun("pdop", pdop);
        fun("gdop", gdop);
        fun("tdop", tdop);
    }

    bool operator==(const ::hozon::sensors::GnssPosInfoFrame& t) const
    {
        return (posType == t.posType) && (fabs(static_cast<double>(latitude - t.latitude)) < DBL_EPSILON) && (fabs(static_cast<double>(longitude - t.longitude)) < DBL_EPSILON) && (fabs(static_cast<double>(undulation - t.undulation)) < DBL_EPSILON) && (fabs(static_cast<double>(altitude - t.altitude)) < DBL_EPSILON) && (fabs(static_cast<double>(latStd - t.latStd)) < DBL_EPSILON) && (fabs(static_cast<double>(lonStd - t.lonStd)) < DBL_EPSILON) && (fabs(static_cast<double>(hgtStd - t.hgtStd)) < DBL_EPSILON) && (svs == t.svs) && (solnSVs == t.solnSVs) && (fabs(static_cast<double>(diffAge - t.diffAge)) < DBL_EPSILON) && (fabs(static_cast<double>(hdop - t.hdop)) < DBL_EPSILON) && (fabs(static_cast<double>(vdop - t.vdop)) < DBL_EPSILON) && (fabs(static_cast<double>(pdop - t.pdop)) < DBL_EPSILON) && (fabs(static_cast<double>(gdop - t.gdop)) < DBL_EPSILON) && (fabs(static_cast<double>(tdop - t.tdop)) < DBL_EPSILON);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_GNSSPOSINFOFRAME_H
