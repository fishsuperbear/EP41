/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CONFIG_SERVER_IMPL_TYPE_STRUCT_CONFIG_ARRAY_H
#define HOZON_CONFIG_SERVER_IMPL_TYPE_STRUCT_CONFIG_ARRAY_H
#include <cfloat>
#include <cmath>
#include "hozon/config/server/impl_type_uint8array_58.h"

namespace hozon {
namespace config {
namespace server {
struct struct_config_array {
    ::hozon::config::server::uint8Array_58 config_array;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(config_array);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(config_array);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("config_array", config_array);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("config_array", config_array);
    }

    bool operator==(const ::hozon::config::server::struct_config_array& t) const
    {
        return (config_array == t.config_array);
    }
};
} // namespace server
} // namespace config
} // namespace hozon


#endif // HOZON_CONFIG_SERVER_IMPL_TYPE_STRUCT_CONFIG_ARRAY_H
