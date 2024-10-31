/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SENSORINFO_IMPL_TYPE_SENSORINFO_H
#define MDC_SENSORINFO_IMPL_TYPE_SENSORINFO_H
#include <cfloat>
#include <cmath>
#include "ara/gnss/impl_type_header.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8vector.h"

namespace mdc {
namespace sensorinfo {
struct SensorInfo {
    ::ara::gnss::Header header;
    ::UInt32 length;
    ::Uint8Vector data;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(length);
        fun(data);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(length);
        fun(data);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("length", length);
        fun("data", data);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("length", length);
        fun("data", data);
    }

    bool operator==(const ::mdc::sensorinfo::SensorInfo& t) const
    {
        return (header == t.header) && (length == t.length) && (data == t.data);
    }
};
} // namespace sensorinfo
} // namespace mdc


#endif // MDC_SENSORINFO_IMPL_TYPE_SENSORINFO_H
