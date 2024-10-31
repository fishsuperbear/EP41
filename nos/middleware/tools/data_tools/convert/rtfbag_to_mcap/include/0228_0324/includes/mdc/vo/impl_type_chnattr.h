/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_VO_IMPL_TYPE_CHNATTR_H
#define MDC_VO_IMPL_TYPE_CHNATTR_H
#include <cfloat>
#include <cmath>

namespace mdc {
namespace vo {
struct ChnAttr {
    bool isEnabled;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(isEnabled);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(isEnabled);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("isEnabled", isEnabled);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("isEnabled", isEnabled);
    }

    bool operator==(const ::mdc::vo::ChnAttr& t) const
    {
        return (isEnabled == t.isEnabled);
    }
};
} // namespace vo
} // namespace mdc


#endif // MDC_VO_IMPL_TYPE_CHNATTR_H
