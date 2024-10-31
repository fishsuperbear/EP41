/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_HMILANECHANGEINFO_H
#define HOZON_HMI_IMPL_TYPE_HMILANECHANGEINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace hmi {
struct HmiLaneChangeInfo {
    ::uint8_t laneChangeStatus;
    ::uint8_t laneChangeType;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(laneChangeStatus);
        fun(laneChangeType);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(laneChangeStatus);
        fun(laneChangeType);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("laneChangeStatus", laneChangeStatus);
        fun("laneChangeType", laneChangeType);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("laneChangeStatus", laneChangeStatus);
        fun("laneChangeType", laneChangeType);
    }

    bool operator==(const ::hozon::hmi::HmiLaneChangeInfo& t) const
    {
        return (laneChangeStatus == t.laneChangeStatus) && (laneChangeType == t.laneChangeType);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_HMILANECHANGEINFO_H
