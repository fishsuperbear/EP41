/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_FAULTPHMDATA_H
#define MDC_DEVM_IMPL_TYPE_FAULTPHMDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "impl_type_uint32.h"

namespace mdc {
namespace devm {
struct FaultPhmData {
    ::UInt8 faultSate;
    ::UInt8 systemDamageLevel;
    ::String faultDesc;
    ::UInt32 faultId;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(faultSate);
        fun(systemDamageLevel);
        fun(faultDesc);
        fun(faultId);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(faultSate);
        fun(systemDamageLevel);
        fun(faultDesc);
        fun(faultId);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("faultSate", faultSate);
        fun("systemDamageLevel", systemDamageLevel);
        fun("faultDesc", faultDesc);
        fun("faultId", faultId);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("faultSate", faultSate);
        fun("systemDamageLevel", systemDamageLevel);
        fun("faultDesc", faultDesc);
        fun("faultId", faultId);
    }

    bool operator==(const ::mdc::devm::FaultPhmData& t) const
    {
        return (faultSate == t.faultSate) && (systemDamageLevel == t.systemDamageLevel) && (faultDesc == t.faultDesc) && (faultId == t.faultId);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_FAULTPHMDATA_H
