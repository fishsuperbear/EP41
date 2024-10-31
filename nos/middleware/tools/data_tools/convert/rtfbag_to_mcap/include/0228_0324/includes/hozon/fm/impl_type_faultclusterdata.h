/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_IMPL_TYPE_FAULTCLUSTERDATA_H
#define HOZON_FM_IMPL_TYPE_FAULTCLUSTERDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"
#include "impl_type_uint8_t.h"
#include "hozon/fm/impl_type_faultclusteritemvector.h"

namespace hozon {
namespace fm {
struct FaultClusterData {
    ::uint32_t fault_key;
    ::uint8_t fault_status;
    ::hozon::fm::FaultClusterItemVector cluster_vec;
    ::uint8_t fault_report_type;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(fault_key);
        fun(fault_status);
        fun(cluster_vec);
        fun(fault_report_type);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(fault_key);
        fun(fault_status);
        fun(cluster_vec);
        fun(fault_report_type);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("fault_key", fault_key);
        fun("fault_status", fault_status);
        fun("cluster_vec", cluster_vec);
        fun("fault_report_type", fault_report_type);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("fault_key", fault_key);
        fun("fault_status", fault_status);
        fun("cluster_vec", cluster_vec);
        fun("fault_report_type", fault_report_type);
    }

    bool operator==(const ::hozon::fm::FaultClusterData& t) const
    {
        return (fault_key == t.fault_key) && (fault_status == t.fault_status) && (cluster_vec == t.cluster_vec) && (fault_report_type == t.fault_report_type);
    }
};
} // namespace fm
} // namespace hozon


#endif // HOZON_FM_IMPL_TYPE_FAULTCLUSTERDATA_H
