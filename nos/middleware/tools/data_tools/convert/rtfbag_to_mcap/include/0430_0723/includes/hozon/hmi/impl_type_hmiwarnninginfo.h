/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_HMIWARNNINGINFO_H
#define HOZON_HMI_IMPL_TYPE_HMIWARNNINGINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"

namespace hozon {
namespace hmi {
struct HmiWarnningInfo {
    ::uint32_t obsID;
    ::uint32_t IsHighLight;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(obsID);
        fun(IsHighLight);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(obsID);
        fun(IsHighLight);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("obsID", obsID);
        fun("IsHighLight", IsHighLight);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("obsID", obsID);
        fun("IsHighLight", IsHighLight);
    }

    bool operator==(const ::hozon::hmi::HmiWarnningInfo& t) const
    {
        return (obsID == t.obsID) && (IsHighLight == t.IsHighLight);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_HMIWARNNINGINFO_H
