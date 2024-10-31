/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_IMPL_TYPE_HZFAULTEVENT_H
#define HOZON_FM_IMPL_TYPE_HZFAULTEVENT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16_t.h"
#include "impl_type_uint8_t.h"
#include "impl_type_uint64_t.h"

namespace hozon {
namespace fm {
struct HzFaultEvent {
    ::uint16_t faultId;
    ::uint8_t faultObj;
    ::uint8_t faultStatus;
    ::uint64_t faultOccurTime;
    ::uint8_t faultUtil;

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
        fun(faultOccurTime);
        fun(faultUtil);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(faultId);
        fun(faultObj);
        fun(faultStatus);
        fun(faultOccurTime);
        fun(faultUtil);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("faultStatus", faultStatus);
        fun("faultOccurTime", faultOccurTime);
        fun("faultUtil", faultUtil);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("faultStatus", faultStatus);
        fun("faultOccurTime", faultOccurTime);
        fun("faultUtil", faultUtil);
    }

    bool operator==(const ::hozon::fm::HzFaultEvent& t) const
    {
        return (faultId == t.faultId) && (faultObj == t.faultObj) && (faultStatus == t.faultStatus) && (faultOccurTime == t.faultOccurTime) && (faultUtil == t.faultUtil);
    }
};
} // namespace fm
} // namespace hozon


#endif // HOZON_FM_IMPL_TYPE_HZFAULTEVENT_H
