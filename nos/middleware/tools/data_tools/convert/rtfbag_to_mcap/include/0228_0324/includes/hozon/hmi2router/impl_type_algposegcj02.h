/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI2ROUTER_IMPL_TYPE_ALGPOSEGCJ02_H
#define HOZON_HMI2ROUTER_IMPL_TYPE_ALGPOSEGCJ02_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace hmi2router {
struct AlgPoseGCJ02 {
    float nns_Lon;
    float nns_Lat;
    float nns_High;
    float nns_Heading;
    ::uint8_t ns;
    ::uint8_t ew;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(nns_Lon);
        fun(nns_Lat);
        fun(nns_High);
        fun(nns_Heading);
        fun(ns);
        fun(ew);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(nns_Lon);
        fun(nns_Lat);
        fun(nns_High);
        fun(nns_Heading);
        fun(ns);
        fun(ew);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("nns_Lon", nns_Lon);
        fun("nns_Lat", nns_Lat);
        fun("nns_High", nns_High);
        fun("nns_Heading", nns_Heading);
        fun("ns", ns);
        fun("ew", ew);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("nns_Lon", nns_Lon);
        fun("nns_Lat", nns_Lat);
        fun("nns_High", nns_High);
        fun("nns_Heading", nns_Heading);
        fun("ns", ns);
        fun("ew", ew);
    }

    bool operator==(const ::hozon::hmi2router::AlgPoseGCJ02& t) const
    {
        return (fabs(static_cast<double>(nns_Lon - t.nns_Lon)) < DBL_EPSILON) && (fabs(static_cast<double>(nns_Lat - t.nns_Lat)) < DBL_EPSILON) && (fabs(static_cast<double>(nns_High - t.nns_High)) < DBL_EPSILON) && (fabs(static_cast<double>(nns_Heading - t.nns_Heading)) < DBL_EPSILON) && (ns == t.ns) && (ew == t.ew);
    }
};
} // namespace hmi2router
} // namespace hozon


#endif // HOZON_HMI2ROUTER_IMPL_TYPE_ALGPOSEGCJ02_H
