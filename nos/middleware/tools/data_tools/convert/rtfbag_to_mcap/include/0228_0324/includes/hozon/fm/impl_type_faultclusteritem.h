/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_IMPL_TYPE_FAULTCLUSTERITEM_H
#define HOZON_FM_IMPL_TYPE_FAULTCLUSTERITEM_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"

namespace hozon {
namespace fm {
struct FaultClusterItem {
    ::String cluster_name;
    ::String cluster_level;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(cluster_name);
        fun(cluster_level);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(cluster_name);
        fun(cluster_level);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("cluster_name", cluster_name);
        fun("cluster_level", cluster_level);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("cluster_name", cluster_name);
        fun("cluster_level", cluster_level);
    }

    bool operator==(const ::hozon::fm::FaultClusterItem& t) const
    {
        return (cluster_name == t.cluster_name) && (cluster_level == t.cluster_level);
    }
};
} // namespace fm
} // namespace hozon


#endif // HOZON_FM_IMPL_TYPE_FAULTCLUSTERITEM_H
