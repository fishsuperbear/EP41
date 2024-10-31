/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_IMPL_TYPE_FMFAULTSTATDATA_H
#define MDC_FM_IMPL_TYPE_FMFAULTSTATDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint64.h"
#include "impl_type_double.h"

namespace mdc {
namespace fm {
struct FmFaultStatData {
    ::UInt32 faultCnt;
    ::UInt32 faultClearCnt;
    ::UInt64 minTimeInvl;
    ::UInt64 maxTimeInvl;
    ::Double avgTimeInvl;
    ::UInt32 alarmCnt;
    ::UInt32 alarmClearCnt;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(faultCnt);
        fun(faultClearCnt);
        fun(minTimeInvl);
        fun(maxTimeInvl);
        fun(avgTimeInvl);
        fun(alarmCnt);
        fun(alarmClearCnt);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(faultCnt);
        fun(faultClearCnt);
        fun(minTimeInvl);
        fun(maxTimeInvl);
        fun(avgTimeInvl);
        fun(alarmCnt);
        fun(alarmClearCnt);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("faultCnt", faultCnt);
        fun("faultClearCnt", faultClearCnt);
        fun("minTimeInvl", minTimeInvl);
        fun("maxTimeInvl", maxTimeInvl);
        fun("avgTimeInvl", avgTimeInvl);
        fun("alarmCnt", alarmCnt);
        fun("alarmClearCnt", alarmClearCnt);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("faultCnt", faultCnt);
        fun("faultClearCnt", faultClearCnt);
        fun("minTimeInvl", minTimeInvl);
        fun("maxTimeInvl", maxTimeInvl);
        fun("avgTimeInvl", avgTimeInvl);
        fun("alarmCnt", alarmCnt);
        fun("alarmClearCnt", alarmClearCnt);
    }

    bool operator==(const ::mdc::fm::FmFaultStatData& t) const
    {
        return (faultCnt == t.faultCnt) && (faultClearCnt == t.faultClearCnt) && (minTimeInvl == t.minTimeInvl) && (maxTimeInvl == t.maxTimeInvl) && (fabs(static_cast<double>(avgTimeInvl - t.avgTimeInvl)) < DBL_EPSILON) && (alarmCnt == t.alarmCnt) && (alarmClearCnt == t.alarmClearCnt);
    }
};
} // namespace fm
} // namespace mdc


#endif // MDC_FM_IMPL_TYPE_FMFAULTSTATDATA_H
