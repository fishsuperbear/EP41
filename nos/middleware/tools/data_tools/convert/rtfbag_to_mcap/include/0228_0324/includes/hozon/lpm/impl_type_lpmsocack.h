/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LPM_IMPL_TYPE_LPMSOCACK_H
#define HOZON_LPM_IMPL_TYPE_LPMSOCACK_H
#include <cfloat>
#include <cmath>
#include "impl_type_int8.h"

namespace hozon {
namespace lpm {
struct LpmSocAck {
    ::Int8 lpm_soc_result;
    ::Int8 lpm_soc_error;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(lpm_soc_result);
        fun(lpm_soc_error);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(lpm_soc_result);
        fun(lpm_soc_error);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("lpm_soc_result", lpm_soc_result);
        fun("lpm_soc_error", lpm_soc_error);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("lpm_soc_result", lpm_soc_result);
        fun("lpm_soc_error", lpm_soc_error);
    }

    bool operator==(const ::hozon::lpm::LpmSocAck& t) const
    {
        return (lpm_soc_result == t.lpm_soc_result) && (lpm_soc_error == t.lpm_soc_error);
    }
};
} // namespace lpm
} // namespace hozon


#endif // HOZON_LPM_IMPL_TYPE_LPMSOCACK_H
