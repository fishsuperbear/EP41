/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_TSP_PKI_IMPL_TYPE_REMOTECONFIGRESULT_H
#define HOZON_TSP_PKI_IMPL_TYPE_REMOTECONFIGRESULT_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32_t.h"
#include "impl_type_string.h"

namespace hozon {
namespace tsp_pki {
struct RemoteConfigResult {
    ::int32_t result_code;
    ::String remote_config;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result_code);
        fun(remote_config);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result_code);
        fun(remote_config);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("result_code", result_code);
        fun("remote_config", remote_config);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("result_code", result_code);
        fun("remote_config", remote_config);
    }

    bool operator==(const ::hozon::tsp_pki::RemoteConfigResult& t) const
    {
        return (result_code == t.result_code) && (remote_config == t.remote_config);
    }
};
} // namespace tsp_pki
} // namespace hozon


#endif // HOZON_TSP_PKI_IMPL_TYPE_REMOTECONFIGRESULT_H
