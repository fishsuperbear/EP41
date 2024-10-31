/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_CANIDINFO_H
#define MDC_DEVM_IMPL_TYPE_CANIDINFO_H
#include <cfloat>
#include <cmath>
#include "mdc/devm/impl_type_uint32array2.h"
#include "impl_type_uint32.h"

namespace mdc {
namespace devm {
struct CanIdInfo {
    ::mdc::devm::UInt32Array2 canIdRange;
    ::UInt32 canIdType;
    ::UInt32 canIdRangeType;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(canIdRange);
        fun(canIdType);
        fun(canIdRangeType);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(canIdRange);
        fun(canIdType);
        fun(canIdRangeType);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("canIdRange", canIdRange);
        fun("canIdType", canIdType);
        fun("canIdRangeType", canIdRangeType);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("canIdRange", canIdRange);
        fun("canIdType", canIdType);
        fun("canIdRangeType", canIdRangeType);
    }

    bool operator==(const ::mdc::devm::CanIdInfo& t) const
    {
        return (canIdRange == t.canIdRange) && (canIdType == t.canIdType) && (canIdRangeType == t.canIdRangeType);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_CANIDINFO_H
