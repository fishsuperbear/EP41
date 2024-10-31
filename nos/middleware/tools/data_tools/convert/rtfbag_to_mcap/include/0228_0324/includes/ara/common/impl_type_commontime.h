/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_COMMON_IMPL_TYPE_COMMONTIME_H
#define ARA_COMMON_IMPL_TYPE_COMMONTIME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"

namespace ara {
namespace common {
struct CommonTime {
    ::UInt32 sec;
    ::UInt32 nsec;

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

    bool operator==(const ::ara::common::CommonTime& t) const
    {
        return (sec == t.sec) && (nsec == t.nsec);
    }
};
} // namespace common
} // namespace ara


#endif // ARA_COMMON_IMPL_TYPE_COMMONTIME_H
