/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_TSP_PKI_IMPL_TYPE_UPLOADTOKENRESULT_H
#define HOZON_TSP_PKI_IMPL_TYPE_UPLOADTOKENRESULT_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32_t.h"
#include "impl_type_string.h"

namespace hozon {
namespace tsp_pki {
struct UploadTokenResult {
    ::int32_t result_code;
    ::String upload_token;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result_code);
        fun(upload_token);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result_code);
        fun(upload_token);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("result_code", result_code);
        fun("upload_token", upload_token);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("result_code", result_code);
        fun("upload_token", upload_token);
    }

    bool operator==(const ::hozon::tsp_pki::UploadTokenResult& t) const
    {
        return (result_code == t.result_code) && (upload_token == t.upload_token);
    }
};
} // namespace tsp_pki
} // namespace hozon


#endif // HOZON_TSP_PKI_IMPL_TYPE_UPLOADTOKENRESULT_H
