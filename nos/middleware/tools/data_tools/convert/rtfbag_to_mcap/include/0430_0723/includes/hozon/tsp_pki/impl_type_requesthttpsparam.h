/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_TSP_PKI_IMPL_TYPE_REQUESTHTTPSPARAM_H
#define HOZON_TSP_PKI_IMPL_TYPE_REQUESTHTTPSPARAM_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32_t.h"
#include "impl_type_string.h"

namespace hozon {
namespace tsp_pki {
struct RequestHttpsParam {
    ::int32_t method;
    ::String url;
    ::String request_body;
    ::String content_type;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(method);
        fun(url);
        fun(request_body);
        fun(content_type);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(method);
        fun(url);
        fun(request_body);
        fun(content_type);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("method", method);
        fun("url", url);
        fun("request_body", request_body);
        fun("content_type", content_type);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("method", method);
        fun("url", url);
        fun("request_body", request_body);
        fun("content_type", content_type);
    }

    bool operator==(const ::hozon::tsp_pki::RequestHttpsParam& t) const
    {
        return (method == t.method) && (url == t.url) && (request_body == t.request_body) && (content_type == t.content_type);
    }
};
} // namespace tsp_pki
} // namespace hozon


#endif // HOZON_TSP_PKI_IMPL_TYPE_REQUESTHTTPSPARAM_H
