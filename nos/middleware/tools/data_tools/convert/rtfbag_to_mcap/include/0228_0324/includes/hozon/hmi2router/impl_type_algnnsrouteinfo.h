/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI2ROUTER_IMPL_TYPE_ALGNNSROUTEINFO_H
#define HOZON_HMI2ROUTER_IMPL_TYPE_ALGNNSROUTEINFO_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32_t.h"
#include "hozon/hmi2router/impl_type_algnnsroute.h"

namespace hozon {
namespace hmi2router {
struct AlgNNSRouteInfo {
    ::hozon::common::CommonHeader header;
    ::uint32_t id;
    ::uint32_t locSeq;
    bool isPublicRoad;
    float nextRouteDis;
    ::uint32_t nextManeuverId;
    bool isReplan;
    ::uint32_t routePointSize;
    ::hozon::hmi2router::AlgNNSRoute nnsRoute;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(id);
        fun(locSeq);
        fun(isPublicRoad);
        fun(nextRouteDis);
        fun(nextManeuverId);
        fun(isReplan);
        fun(routePointSize);
        fun(nnsRoute);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(id);
        fun(locSeq);
        fun(isPublicRoad);
        fun(nextRouteDis);
        fun(nextManeuverId);
        fun(isReplan);
        fun(routePointSize);
        fun(nnsRoute);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("id", id);
        fun("locSeq", locSeq);
        fun("isPublicRoad", isPublicRoad);
        fun("nextRouteDis", nextRouteDis);
        fun("nextManeuverId", nextManeuverId);
        fun("isReplan", isReplan);
        fun("routePointSize", routePointSize);
        fun("nnsRoute", nnsRoute);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("id", id);
        fun("locSeq", locSeq);
        fun("isPublicRoad", isPublicRoad);
        fun("nextRouteDis", nextRouteDis);
        fun("nextManeuverId", nextManeuverId);
        fun("isReplan", isReplan);
        fun("routePointSize", routePointSize);
        fun("nnsRoute", nnsRoute);
    }

    bool operator==(const ::hozon::hmi2router::AlgNNSRouteInfo& t) const
    {
        return (header == t.header) && (id == t.id) && (locSeq == t.locSeq) && (isPublicRoad == t.isPublicRoad) && (fabs(static_cast<double>(nextRouteDis - t.nextRouteDis)) < DBL_EPSILON) && (nextManeuverId == t.nextManeuverId) && (isReplan == t.isReplan) && (routePointSize == t.routePointSize) && (nnsRoute == t.nnsRoute);
    }
};
} // namespace hmi2router
} // namespace hozon


#endif // HOZON_HMI2ROUTER_IMPL_TYPE_ALGNNSROUTEINFO_H
