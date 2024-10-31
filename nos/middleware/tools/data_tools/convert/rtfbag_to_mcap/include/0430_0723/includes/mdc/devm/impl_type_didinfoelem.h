/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_DIDINFOELEM_H
#define MDC_DEVM_IMPL_TYPE_DIDINFOELEM_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"
#include "mdc/devm/impl_type_uint8list.h"

namespace mdc {
namespace devm {
struct DidInfoElem {
    ::UInt16 did;
    ::mdc::devm::Uint8List value;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(did);
        fun(value);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(did);
        fun(value);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("did", did);
        fun("value", value);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("did", did);
        fun("value", value);
    }

    bool operator==(const ::mdc::devm::DidInfoElem& t) const
    {
        return (did == t.did) && (value == t.value);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_DIDINFOELEM_H
