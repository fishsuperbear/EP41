/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_MODULERESULT_H
#define HOZON_LOCATION_IMPL_TYPE_MODULERESULT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"

namespace hozon {
namespace location {
struct ModuleResult {
    ::UInt8 module;
    ::UInt32 errorCode;
    ::UInt32 innerCode;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(module);
        fun(errorCode);
        fun(innerCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(module);
        fun(errorCode);
        fun(innerCode);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("module", module);
        fun("errorCode", errorCode);
        fun("innerCode", innerCode);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("module", module);
        fun("errorCode", errorCode);
        fun("innerCode", innerCode);
    }

    bool operator==(const ::hozon::location::ModuleResult& t) const
    {
        return (module == t.module) && (errorCode == t.errorCode) && (innerCode == t.innerCode);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_MODULERESULT_H
