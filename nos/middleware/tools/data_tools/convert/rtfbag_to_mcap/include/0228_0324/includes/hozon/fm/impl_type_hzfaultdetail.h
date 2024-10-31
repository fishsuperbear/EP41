/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_IMPL_TYPE_HZFAULTDETAIL_H
#define HOZON_FM_IMPL_TYPE_HZFAULTDETAIL_H
#include <cfloat>
#include <cmath>
#include "hozon/fm/impl_type_hzfaultdata.h"
#include "impl_type_string.h"
#include "impl_type_uint64_t.h"

namespace hozon {
namespace fm {
struct HzFaultDetail {
    ::hozon::fm::HzFaultData faultData;
    ::String actionName;
    ::uint64_t firstTime;
    ::uint64_t lastTime;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(faultData);
        fun(actionName);
        fun(firstTime);
        fun(lastTime);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(faultData);
        fun(actionName);
        fun(firstTime);
        fun(lastTime);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("faultData", faultData);
        fun("actionName", actionName);
        fun("firstTime", firstTime);
        fun("lastTime", lastTime);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("faultData", faultData);
        fun("actionName", actionName);
        fun("firstTime", firstTime);
        fun("lastTime", lastTime);
    }

    bool operator==(const ::hozon::fm::HzFaultDetail& t) const
    {
        return (faultData == t.faultData) && (actionName == t.actionName) && (firstTime == t.firstTime) && (lastTime == t.lastTime);
    }
};
} // namespace fm
} // namespace hozon


#endif // HOZON_FM_IMPL_TYPE_HZFAULTDETAIL_H
