/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_TSP_PKI_IMPL_TYPE_RESPONSEHTTPS_H
#define HOZON_TSP_PKI_IMPL_TYPE_RESPONSEHTTPS_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32_t.h"
#include "impl_type_string.h"

namespace hozon {
namespace tsp_pki {
struct ResponseHttps {
    ::int32_t result_code;
    ::String response;
    ::String content_type;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result_code);
        fun(response);
        fun(content_type);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result_code);
        fun(response);
        fun(content_type);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("result_code", result_code);
        fun("response", response);
        fun("content_type", content_type);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("result_code", result_code);
        fun("response", response);
        fun("content_type", content_type);
    }

    bool operator==(const ::hozon::tsp_pki::ResponseHttps& t) const
    {
        return (result_code == t.result_code) && (response == t.response) && (content_type == t.content_type);
    }
};
} // namespace tsp_pki
} // namespace hozon


#endif // HOZON_TSP_PKI_IMPL_TYPE_RESPONSEHTTPS_H
