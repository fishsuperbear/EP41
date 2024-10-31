/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SM_IMPL_TYPE_FGSTATECHANGE_H
#define HOZON_SM_IMPL_TYPE_FGSTATECHANGE_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"

namespace hozon {
namespace sm {
struct FGStateChange {
    ::String functionGroupName;
    ::String stateName;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(functionGroupName);
        fun(stateName);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(functionGroupName);
        fun(stateName);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("functionGroupName", functionGroupName);
        fun("stateName", stateName);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("functionGroupName", functionGroupName);
        fun("stateName", stateName);
    }

    bool operator==(const ::hozon::sm::FGStateChange& t) const
    {
        return (functionGroupName == t.functionGroupName) && (stateName == t.stateName);
    }
};
} // namespace sm
} // namespace hozon


#endif // HOZON_SM_IMPL_TYPE_FGSTATECHANGE_H
