/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_NNSPATH_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_NNSPATH_STRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"
#include "impl_type_uint16_t.h"

namespace hozon {
namespace hmi {
struct NNSPath_Struct {
    float nns_Lon;
    float nns_Lat;
    float nns_High;
    float nns_Heading;
    ::uint8_t ns;
    ::uint8_t ew;
    ::uint16_t padding_u16_1;

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
        fun(padding_u16_1);
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
        fun(padding_u16_1);
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
        fun("padding_u16_1", padding_u16_1);
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
        fun("padding_u16_1", padding_u16_1);
    }

    bool operator==(const ::hozon::hmi::NNSPath_Struct& t) const
    {
        return (fabs(static_cast<double>(nns_Lon - t.nns_Lon)) < DBL_EPSILON) && (fabs(static_cast<double>(nns_Lat - t.nns_Lat)) < DBL_EPSILON) && (fabs(static_cast<double>(nns_High - t.nns_High)) < DBL_EPSILON) && (fabs(static_cast<double>(nns_Heading - t.nns_Heading)) < DBL_EPSILON) && (ns == t.ns) && (ew == t.ew) && (padding_u16_1 == t.padding_u16_1);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_NNSPATH_STRUCT_H
