/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_LATENCYINFO_H
#define IMPL_TYPE_LATENCYINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_linkinfovec.h"

namespace mdc{
struct LatencyInfo {
    ::LinkInfoVec linkInfos;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(linkInfos);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(linkInfos);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("linkInfos", linkInfos);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("linkInfos", linkInfos);
    }

    bool operator==(const mdc::LatencyInfo& t) const
    {
        return (linkInfos == t.linkInfos);
    }
};
}

#endif // IMPL_TYPE_LATENCYINFO_H
