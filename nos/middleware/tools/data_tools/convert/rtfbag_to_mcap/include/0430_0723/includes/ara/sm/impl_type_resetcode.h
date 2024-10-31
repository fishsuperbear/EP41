/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SM_IMPL_TYPE_RESETCODE_H
#define ARA_SM_IMPL_TYPE_RESETCODE_H
#include <cfloat>
#include <cmath>
#include "ara/sm/impl_type_resettype.h"
#include "impl_type_uint32.h"

namespace ara {
namespace sm {
struct ResetCode {
    ::ara::sm::ResetType resetType;
    ::UInt32 resetTime;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(resetType);
        fun(resetTime);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(resetType);
        fun(resetTime);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("resetType", resetType);
        fun("resetTime", resetTime);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("resetType", resetType);
        fun("resetTime", resetTime);
    }

    bool operator==(const ::ara::sm::ResetCode& t) const
    {
        return (resetType == t.resetType) && (resetTime == t.resetTime);
    }
};
} // namespace sm
} // namespace ara


#endif // ARA_SM_IMPL_TYPE_RESETCODE_H
