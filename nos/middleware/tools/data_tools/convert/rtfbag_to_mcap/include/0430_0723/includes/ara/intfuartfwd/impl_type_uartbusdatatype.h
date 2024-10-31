/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_INTFUARTFWD_IMPL_TYPE_UARTBUSDATATYPE_H
#define ARA_INTFUARTFWD_IMPL_TYPE_UARTBUSDATATYPE_H
#include <cfloat>
#include <cmath>
#include "ara/intfuartfwd/impl_type_header.h"
#include "ara/intfuartfwd/impl_type_uint8vector.h"

namespace ara {
namespace intfuartfwd {
struct UartBusDataType {
    ::ara::intfuartfwd::Header header;
    ::ara::intfuartfwd::Uint8Vector data;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(data);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(data);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("data", data);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("data", data);
    }

    bool operator==(const ::ara::intfuartfwd::UartBusDataType& t) const
    {
        return (header == t.header) && (data == t.data);
    }
};
} // namespace intfuartfwd
} // namespace ara


#endif // ARA_INTFUARTFWD_IMPL_TYPE_UARTBUSDATATYPE_H
