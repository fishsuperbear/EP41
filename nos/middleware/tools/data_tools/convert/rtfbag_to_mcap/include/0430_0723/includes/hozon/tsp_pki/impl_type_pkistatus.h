/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_TSP_PKI_IMPL_TYPE_PKISTATUS_H
#define HOZON_TSP_PKI_IMPL_TYPE_PKISTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32_t.h"

namespace hozon {
namespace tsp_pki {
struct Pkistatus {
    ::int32_t Pkistatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Pkistatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Pkistatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Pkistatus", Pkistatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Pkistatus", Pkistatus);
    }

    bool operator==(const ::hozon::tsp_pki::Pkistatus& t) const
    {
        return (Pkistatus == t.Pkistatus);
    }
};
} // namespace tsp_pki
} // namespace hozon


#endif // HOZON_TSP_PKI_IMPL_TYPE_PKISTATUS_H
