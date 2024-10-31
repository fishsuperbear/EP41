/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_MACADDRESS_H
#define MDC_DEVM_IMPL_TYPE_MACADDRESS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "mdc/devm/impl_type_macaddrarray.h"
#include "impl_type_uint8.h"

namespace mdc {
namespace devm {
struct MacAddress {
    ::UInt32 macNum;
    ::mdc::devm::MacAddrArray macArray;
    ::UInt8 result;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(macNum);
        fun(macArray);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(macNum);
        fun(macArray);
        fun(result);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("macNum", macNum);
        fun("macArray", macArray);
        fun("result", result);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("macNum", macNum);
        fun("macArray", macArray);
        fun("result", result);
    }

    bool operator==(const ::mdc::devm::MacAddress& t) const
    {
        return (macNum == t.macNum) && (macArray == t.macArray) && (result == t.result);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_MACADDRESS_H
