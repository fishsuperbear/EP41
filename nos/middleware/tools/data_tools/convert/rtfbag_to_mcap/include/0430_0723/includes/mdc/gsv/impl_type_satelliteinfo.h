/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_GSV_IMPL_TYPE_SATELLITEINFO_H
#define MDC_GSV_IMPL_TYPE_SATELLITEINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_int32.h"
#include "impl_type_double.h"
#include "impl_type_uint8.h"

namespace mdc {
namespace gsv {
struct SatelliteInfo {
    ::String gsvName;
    ::Int32 prn;
    ::Double elevation;
    ::Double azimuth;
    ::Double snr;
    ::UInt8 signalId;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(gsvName);
        fun(prn);
        fun(elevation);
        fun(azimuth);
        fun(snr);
        fun(signalId);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(gsvName);
        fun(prn);
        fun(elevation);
        fun(azimuth);
        fun(snr);
        fun(signalId);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("gsvName", gsvName);
        fun("prn", prn);
        fun("elevation", elevation);
        fun("azimuth", azimuth);
        fun("snr", snr);
        fun("signalId", signalId);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("gsvName", gsvName);
        fun("prn", prn);
        fun("elevation", elevation);
        fun("azimuth", azimuth);
        fun("snr", snr);
        fun("signalId", signalId);
    }

    bool operator==(const ::mdc::gsv::SatelliteInfo& t) const
    {
        return (gsvName == t.gsvName) && (prn == t.prn) && (fabs(static_cast<double>(elevation - t.elevation)) < DBL_EPSILON) && (fabs(static_cast<double>(azimuth - t.azimuth)) < DBL_EPSILON) && (fabs(static_cast<double>(snr - t.snr)) < DBL_EPSILON) && (signalId == t.signalId);
    }
};
} // namespace gsv
} // namespace mdc


#endif // MDC_GSV_IMPL_TYPE_SATELLITEINFO_H
