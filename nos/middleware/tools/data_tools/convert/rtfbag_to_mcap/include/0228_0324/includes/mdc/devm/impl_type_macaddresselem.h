/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_MACADDRESSELEM_H
#define MDC_DEVM_IMPL_TYPE_MACADDRESSELEM_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"
#include "impl_type_uint8.h"
#include "mdc/devm/impl_type_bufarray12.h"

namespace mdc {
namespace devm {
struct MacAddressElem {
    ::UInt16 type;
    ::UInt8 subType;
    ::UInt8 format;
    ::mdc::devm::BufArray12 macBuf;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(type);
        fun(subType);
        fun(format);
        fun(macBuf);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(type);
        fun(subType);
        fun(format);
        fun(macBuf);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("type", type);
        fun("subType", subType);
        fun("format", format);
        fun("macBuf", macBuf);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("type", type);
        fun("subType", subType);
        fun("format", format);
        fun("macBuf", macBuf);
    }

    bool operator==(const ::mdc::devm::MacAddressElem& t) const
    {
        return (type == t.type) && (subType == t.subType) && (format == t.format) && (macBuf == t.macBuf);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_MACADDRESSELEM_H
