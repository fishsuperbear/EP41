/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SM_IMPL_TYPE_STATETRANSITION_H
#define ARA_SM_IMPL_TYPE_STATETRANSITION_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"

namespace ara {
namespace sm {
struct StateTransition {
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

    bool operator==(const ::ara::sm::StateTransition& t) const
    {
        return (functionGroupName == t.functionGroupName) && (stateName == t.stateName);
    }
};
} // namespace sm
} // namespace ara


#endif // ARA_SM_IMPL_TYPE_STATETRANSITION_H
