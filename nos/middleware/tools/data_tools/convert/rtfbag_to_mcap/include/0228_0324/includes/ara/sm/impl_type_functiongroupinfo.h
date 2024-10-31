/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SM_IMPL_TYPE_FUNCTIONGROUPINFO_H
#define ARA_SM_IMPL_TYPE_FUNCTIONGROUPINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "ara/sm/impl_type_statenames.h"

namespace ara {
namespace sm {
struct FunctionGroupInfo {
    ::String functionGroupName;
    ::ara::sm::StateNames stateNames;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(functionGroupName);
        fun(stateNames);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(functionGroupName);
        fun(stateNames);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("functionGroupName", functionGroupName);
        fun("stateNames", stateNames);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("functionGroupName", functionGroupName);
        fun("stateNames", stateNames);
    }

    bool operator==(const ::ara::sm::FunctionGroupInfo& t) const
    {
        return (functionGroupName == t.functionGroupName) && (stateNames == t.stateNames);
    }
};
} // namespace sm
} // namespace ara


#endif // ARA_SM_IMPL_TYPE_FUNCTIONGROUPINFO_H
