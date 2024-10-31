/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_IMPL_TYPE_HZFAULTADVICES_H
#define HOZON_FM_IMPL_TYPE_HZFAULTADVICES_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_uint16_t.h"

namespace hozon {
namespace fm {
struct HzFaultAdvices {
    ::String name;
    ::uint16_t advices;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(name);
        fun(advices);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(name);
        fun(advices);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("name", name);
        fun("advices", advices);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("name", name);
        fun("advices", advices);
    }

    bool operator==(const ::hozon::fm::HzFaultAdvices& t) const
    {
        return (name == t.name) && (advices == t.advices);
    }
};
} // namespace fm
} // namespace hozon


#endif // HOZON_FM_IMPL_TYPE_HZFAULTADVICES_H
