/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_IMPL_TYPE_HZFAULTANALYSISEVENT_H
#define HOZON_FM_IMPL_TYPE_HZFAULTANALYSISEVENT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16_t.h"
#include "impl_type_uint8_t.h"
#include "impl_type_uint64_t.h"

namespace hozon {
namespace fm {
struct HzFaultAnalysisEvent {
    ::uint16_t faultId;
    ::uint16_t faultObj;
    ::uint8_t faultStatus;
    ::uint64_t silentCount;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(faultId);
        fun(faultObj);
        fun(faultStatus);
        fun(silentCount);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(faultId);
        fun(faultObj);
        fun(faultStatus);
        fun(silentCount);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("faultStatus", faultStatus);
        fun("silentCount", silentCount);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("faultStatus", faultStatus);
        fun("silentCount", silentCount);
    }

    bool operator==(const ::hozon::fm::HzFaultAnalysisEvent& t) const
    {
        return (faultId == t.faultId) && (faultObj == t.faultObj) && (faultStatus == t.faultStatus) && (silentCount == t.silentCount);
    }
};
} // namespace fm
} // namespace hozon


#endif // HOZON_FM_IMPL_TYPE_HZFAULTANALYSISEVENT_H
