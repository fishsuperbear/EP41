/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_IMPL_TYPE_HZFAULTDATA_H
#define HOZON_FM_IMPL_TYPE_HZFAULTDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16_t.h"
#include "impl_type_uint8_t.h"
#include "impl_type_uint64_t.h"
#include "impl_type_string.h"

namespace hozon {
namespace fm {
struct HzFaultData {
    ::uint16_t faultId;
    ::uint16_t faultObj;
    ::uint8_t faultStatus;
    ::uint64_t faultOccurTime_sec;
    ::uint64_t faultOccurTime_nsec;
    ::String faultDes;
    ::uint16_t faultClusterId;
    ::uint8_t faultChannel;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(faultId);
        fun(faultObj);
        fun(faultStatus);
        fun(faultOccurTime_sec);
        fun(faultOccurTime_nsec);
        fun(faultDes);
        fun(faultClusterId);
        fun(faultChannel);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(faultId);
        fun(faultObj);
        fun(faultStatus);
        fun(faultOccurTime_sec);
        fun(faultOccurTime_nsec);
        fun(faultDes);
        fun(faultClusterId);
        fun(faultChannel);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("faultStatus", faultStatus);
        fun("faultOccurTime_sec", faultOccurTime_sec);
        fun("faultOccurTime_nsec", faultOccurTime_nsec);
        fun("faultDes", faultDes);
        fun("faultClusterId", faultClusterId);
        fun("faultChannel", faultChannel);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("faultStatus", faultStatus);
        fun("faultOccurTime_sec", faultOccurTime_sec);
        fun("faultOccurTime_nsec", faultOccurTime_nsec);
        fun("faultDes", faultDes);
        fun("faultClusterId", faultClusterId);
        fun("faultChannel", faultChannel);
    }

    bool operator==(const ::hozon::fm::HzFaultData& t) const
    {
        return (faultId == t.faultId) && (faultObj == t.faultObj) && (faultStatus == t.faultStatus) && (faultOccurTime_sec == t.faultOccurTime_sec) && (faultOccurTime_nsec == t.faultOccurTime_nsec) && (faultDes == t.faultDes) && (faultClusterId == t.faultClusterId) && (faultChannel == t.faultChannel);
    }
};
} // namespace fm
} // namespace hozon


#endif // HOZON_FM_IMPL_TYPE_HZFAULTDATA_H
