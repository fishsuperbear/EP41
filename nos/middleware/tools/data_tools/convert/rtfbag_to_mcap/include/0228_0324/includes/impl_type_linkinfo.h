/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_LINKINFO_H
#define IMPL_TYPE_LINKINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "hozon/common/impl_type_commontime.h"

namespace mdc {
struct LinkInfo {
    ::String linkName;
    ::hozon::common::CommonTime timestamp;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(linkName);
        fun(timestamp);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(linkName);
        fun(timestamp);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("linkName", linkName);
        fun("timestamp", timestamp);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("linkName", linkName);
        fun("timestamp", timestamp);
    }

    bool operator==(const mdc::LinkInfo& t) const
    {
        return (linkName == t.linkName) && (timestamp == t.timestamp);
    }
};
}

#endif // IMPL_TYPE_LINKINFO_H
