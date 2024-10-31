/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGCHASSISTIME_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGCHASSISTIME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint64_t.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace chassis {
struct AlgChassisTime {
    ::uint64_t CalendarYear;
    ::uint8_t CalendarMonth;
    ::uint8_t CalendarDay;
    ::uint8_t HourOfDay;
    ::uint8_t MinuteOfHour;
    ::uint8_t SecsOfMinute;
    ::uint8_t TimeDspFmt;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(CalendarYear);
        fun(CalendarMonth);
        fun(CalendarDay);
        fun(HourOfDay);
        fun(MinuteOfHour);
        fun(SecsOfMinute);
        fun(TimeDspFmt);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(CalendarYear);
        fun(CalendarMonth);
        fun(CalendarDay);
        fun(HourOfDay);
        fun(MinuteOfHour);
        fun(SecsOfMinute);
        fun(TimeDspFmt);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("CalendarYear", CalendarYear);
        fun("CalendarMonth", CalendarMonth);
        fun("CalendarDay", CalendarDay);
        fun("HourOfDay", HourOfDay);
        fun("MinuteOfHour", MinuteOfHour);
        fun("SecsOfMinute", SecsOfMinute);
        fun("TimeDspFmt", TimeDspFmt);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("CalendarYear", CalendarYear);
        fun("CalendarMonth", CalendarMonth);
        fun("CalendarDay", CalendarDay);
        fun("HourOfDay", HourOfDay);
        fun("MinuteOfHour", MinuteOfHour);
        fun("SecsOfMinute", SecsOfMinute);
        fun("TimeDspFmt", TimeDspFmt);
    }

    bool operator==(const ::hozon::chassis::AlgChassisTime& t) const
    {
        return (CalendarYear == t.CalendarYear) && (CalendarMonth == t.CalendarMonth) && (CalendarDay == t.CalendarDay) && (HourOfDay == t.HourOfDay) && (MinuteOfHour == t.MinuteOfHour) && (SecsOfMinute == t.SecsOfMinute) && (TimeDspFmt == t.TimeDspFmt);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGCHASSISTIME_H
