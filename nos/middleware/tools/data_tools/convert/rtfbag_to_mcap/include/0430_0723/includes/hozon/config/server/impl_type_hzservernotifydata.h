/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CONFIG_SERVER_IMPL_TYPE_HZSERVERNOTIFYDATA_H
#define HOZON_CONFIG_SERVER_IMPL_TYPE_HZSERVERNOTIFYDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32_t.h"
#include "impl_type_string.h"

namespace hozon {
namespace config {
namespace server {
struct HzServerNotifyData {
    ::int32_t notifyType;
    ::String clientName;
    ::String extraData;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(notifyType);
        fun(clientName);
        fun(extraData);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(notifyType);
        fun(clientName);
        fun(extraData);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("notifyType", notifyType);
        fun("clientName", clientName);
        fun("extraData", extraData);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("notifyType", notifyType);
        fun("clientName", clientName);
        fun("extraData", extraData);
    }

    bool operator==(const ::hozon::config::server::HzServerNotifyData& t) const
    {
        return (notifyType == t.notifyType) && (clientName == t.clientName) && (extraData == t.extraData);
    }
};
} // namespace server
} // namespace config
} // namespace hozon


#endif // HOZON_CONFIG_SERVER_IMPL_TYPE_HZSERVERNOTIFYDATA_H
