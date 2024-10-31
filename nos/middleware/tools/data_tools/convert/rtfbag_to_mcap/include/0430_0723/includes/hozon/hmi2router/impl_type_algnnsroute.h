/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI2ROUTER_IMPL_TYPE_ALGNNSROUTE_H
#define HOZON_HMI2ROUTER_IMPL_TYPE_ALGNNSROUTE_H
#include <cfloat>
#include <cmath>
#include "hozon/hmi2router/impl_type_algposegcj02_vector.h"
#include "hozon/hmi2router/impl_type_algposelocal_vector.h"

namespace hozon {
namespace hmi2router {
struct AlgNNSRoute {
    ::hozon::hmi2router::AlgPoseGCJ02_vector routeGCJ02;
    ::hozon::hmi2router::AlgPoseLocal_vector routeLocal;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(routeGCJ02);
        fun(routeLocal);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(routeGCJ02);
        fun(routeLocal);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("routeGCJ02", routeGCJ02);
        fun("routeLocal", routeLocal);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("routeGCJ02", routeGCJ02);
        fun("routeLocal", routeLocal);
    }

    bool operator==(const ::hozon::hmi2router::AlgNNSRoute& t) const
    {
        return (routeGCJ02 == t.routeGCJ02) && (routeLocal == t.routeLocal);
    }
};
} // namespace hmi2router
} // namespace hozon


#endif // HOZON_HMI2ROUTER_IMPL_TYPE_ALGNNSROUTE_H
