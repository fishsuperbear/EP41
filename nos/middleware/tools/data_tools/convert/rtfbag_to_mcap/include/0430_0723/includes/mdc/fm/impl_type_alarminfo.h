/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_IMPL_TYPE_ALARMINFO_H
#define MDC_FM_IMPL_TYPE_ALARMINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"
#include "impl_type_uint8.h"
#include "impl_type_uint64.h"
#include "impl_type_string.h"

namespace mdc {
namespace fm {
struct AlarmInfo {
    ::UInt16 alarmId;
    ::UInt16 alarmObj;
    ::UInt8 clss;
    ::UInt8 level;
    ::UInt8 status;
    ::UInt64 time;
    ::String desc;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(alarmId);
        fun(alarmObj);
        fun(clss);
        fun(level);
        fun(status);
        fun(time);
        fun(desc);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(alarmId);
        fun(alarmObj);
        fun(clss);
        fun(level);
        fun(status);
        fun(time);
        fun(desc);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("alarmId", alarmId);
        fun("alarmObj", alarmObj);
        fun("clss", clss);
        fun("level", level);
        fun("status", status);
        fun("time", time);
        fun("desc", desc);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("alarmId", alarmId);
        fun("alarmObj", alarmObj);
        fun("clss", clss);
        fun("level", level);
        fun("status", status);
        fun("time", time);
        fun("desc", desc);
    }

    bool operator==(const ::mdc::fm::AlarmInfo& t) const
    {
        return (alarmId == t.alarmId) && (alarmObj == t.alarmObj) && (clss == t.clss) && (level == t.level) && (status == t.status) && (time == t.time) && (desc == t.desc);
    }
};
} // namespace fm
} // namespace mdc


#endif // MDC_FM_IMPL_TYPE_ALARMINFO_H
