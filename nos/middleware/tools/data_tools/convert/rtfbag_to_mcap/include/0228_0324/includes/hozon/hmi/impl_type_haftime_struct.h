/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_HAFTIME_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_HAFTIME_STRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"

namespace hozon {
namespace hmi {
struct HafTime_Struct {
    ::uint32_t sec;
    ::uint32_t nsec;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(sec);
        fun(nsec);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(sec);
        fun(nsec);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("sec", sec);
        fun("nsec", nsec);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("sec", sec);
        fun("nsec", nsec);
    }

    bool operator==(const ::hozon::hmi::HafTime_Struct& t) const
    {
        return (sec == t.sec) && (nsec == t.nsec);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_HAFTIME_STRUCT_H
