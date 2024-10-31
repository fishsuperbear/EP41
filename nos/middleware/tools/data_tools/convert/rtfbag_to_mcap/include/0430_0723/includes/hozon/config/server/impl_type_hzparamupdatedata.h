/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CONFIG_SERVER_IMPL_TYPE_HZPARAMUPDATEDATA_H
#define HOZON_CONFIG_SERVER_IMPL_TYPE_HZPARAMUPDATEDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace config {
namespace server {
struct HzParamUpdateData {
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

    bool operator==(const ::hozon::config::server::HzParamUpdateData& t) const
    {
        return (paramName == t.paramName) && (paramValue == t.paramValue) && (paramType == t.paramType) && (clientName == t.clientName);
    }
};
} // namespace server
} // namespace config
} // namespace hozon


#endif // HOZON_CONFIG_SERVER_IMPL_TYPE_HZPARAMUPDATEDATA_H
