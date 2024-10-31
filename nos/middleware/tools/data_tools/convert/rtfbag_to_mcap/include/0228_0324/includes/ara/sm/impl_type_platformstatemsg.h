/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SM_IMPL_TYPE_PLATFORMSTATEMSG_H
#define ARA_SM_IMPL_TYPE_PLATFORMSTATEMSG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace ara {
namespace sm {
struct PlatformStateMsg {
    ::uint8_t platformState;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(platformState);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(platformState);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("platformState", platformState);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("platformState", platformState);
    }

    bool operator==(const ::ara::sm::PlatformStateMsg& t) const
    {
        return (platformState == t.platformState);
    }
};
} // namespace sm
} // namespace ara


#endif // ARA_SM_IMPL_TYPE_PLATFORMSTATEMSG_H
