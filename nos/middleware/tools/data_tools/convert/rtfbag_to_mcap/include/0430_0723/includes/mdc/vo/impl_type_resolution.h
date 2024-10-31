/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_VO_IMPL_TYPE_RESOLUTION_H
#define MDC_VO_IMPL_TYPE_RESOLUTION_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"

namespace mdc {
namespace vo {
struct Resolution {
    ::uint32_t width;
    ::uint32_t height;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(width);
        fun(height);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(width);
        fun(height);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("width", width);
        fun("height", height);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("width", width);
        fun("height", height);
    }

    bool operator==(const ::mdc::vo::Resolution& t) const
    {
        return (width == t.width) && (height == t.height);
    }
};
} // namespace vo
} // namespace mdc


#endif // MDC_VO_IMPL_TYPE_RESOLUTION_H
