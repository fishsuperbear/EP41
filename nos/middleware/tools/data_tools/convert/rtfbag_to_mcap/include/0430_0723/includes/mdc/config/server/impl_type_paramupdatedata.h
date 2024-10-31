/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_CONFIG_SERVER_IMPL_TYPE_PARAMUPDATEDATA_H
#define MDC_CONFIG_SERVER_IMPL_TYPE_PARAMUPDATEDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_uint8.h"

namespace mdc {
namespace config {
namespace server {
struct ParamUpdateData {
    ::String paramName;
    ::String paramValue;
    ::UInt8 paramType;
    ::String clientName;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(paramName);
        fun(paramValue);
        fun(paramType);
        fun(clientName);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(paramName);
        fun(paramValue);
        fun(paramType);
        fun(clientName);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("paramName", paramName);
        fun("paramValue", paramValue);
        fun("paramType", paramType);
        fun("clientName", clientName);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("paramName", paramName);
        fun("paramValue", paramValue);
        fun("paramType", paramType);
        fun("clientName", clientName);
    }

    bool operator==(const ::mdc::config::server::ParamUpdateData& t) const
    {
        return (paramName == t.paramName) && (paramValue == t.paramValue) && (paramType == t.paramType) && (clientName == t.clientName);
    }
};
} // namespace server
} // namespace config
} // namespace mdc


#endif // MDC_CONFIG_SERVER_IMPL_TYPE_PARAMUPDATEDATA_H
