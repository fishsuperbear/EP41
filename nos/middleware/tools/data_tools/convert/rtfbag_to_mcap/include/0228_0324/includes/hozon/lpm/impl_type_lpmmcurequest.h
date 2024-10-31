/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LPM_IMPL_TYPE_LPMMCUREQUEST_H
#define HOZON_LPM_IMPL_TYPE_LPMMCUREQUEST_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace lpm {
struct LpmMcuRequest {
    ::UInt8 lpm_req_state;
    ::UInt8 lpm_req_reason;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(lpm_req_state);
        fun(lpm_req_reason);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(lpm_req_state);
        fun(lpm_req_reason);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("lpm_req_state", lpm_req_state);
        fun("lpm_req_reason", lpm_req_reason);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("lpm_req_state", lpm_req_state);
        fun("lpm_req_reason", lpm_req_reason);
    }

    bool operator==(const ::hozon::lpm::LpmMcuRequest& t) const
    {
        return (lpm_req_state == t.lpm_req_state) && (lpm_req_reason == t.lpm_req_reason);
    }
};
} // namespace lpm
} // namespace hozon


#endif // HOZON_LPM_IMPL_TYPE_LPMMCUREQUEST_H
