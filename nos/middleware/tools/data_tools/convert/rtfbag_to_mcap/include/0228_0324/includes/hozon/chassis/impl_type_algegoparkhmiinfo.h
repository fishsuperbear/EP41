/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGEGOPARKHMIINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGEGOPARKHMIINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "impl_type_uint16.h"

namespace hozon {
namespace chassis {
struct AlgEgoParkHmiInfo {
    ::UInt8 PA_ParkBarPercent;
    ::Float PA_GuideLineE_a;
    ::Float PA_GuideLineE_b;
    ::Float PA_GuideLineE_c;
    ::Float PA_GuideLineE_d;
    ::Float PA_GuideLineE_Xmin;
    ::Float PA_GuideLineE_Xmax;
    ::UInt8 HourOfDay;
    ::UInt8 MinuteOfHour;
    ::UInt8 SecondOfMinute;
    ::UInt16 NNS_distance;
    ::UInt16 HPA_distance;
    ::UInt16 Parkingtimeremaining;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(PA_ParkBarPercent);
        fun(PA_GuideLineE_a);
        fun(PA_GuideLineE_b);
        fun(PA_GuideLineE_c);
        fun(PA_GuideLineE_d);
        fun(PA_GuideLineE_Xmin);
        fun(PA_GuideLineE_Xmax);
        fun(HourOfDay);
        fun(MinuteOfHour);
        fun(SecondOfMinute);
        fun(NNS_distance);
        fun(HPA_distance);
        fun(Parkingtimeremaining);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(PA_ParkBarPercent);
        fun(PA_GuideLineE_a);
        fun(PA_GuideLineE_b);
        fun(PA_GuideLineE_c);
        fun(PA_GuideLineE_d);
        fun(PA_GuideLineE_Xmin);
        fun(PA_GuideLineE_Xmax);
        fun(HourOfDay);
        fun(MinuteOfHour);
        fun(SecondOfMinute);
        fun(NNS_distance);
        fun(HPA_distance);
        fun(Parkingtimeremaining);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("PA_ParkBarPercent", PA_ParkBarPercent);
        fun("PA_GuideLineE_a", PA_GuideLineE_a);
        fun("PA_GuideLineE_b", PA_GuideLineE_b);
        fun("PA_GuideLineE_c", PA_GuideLineE_c);
        fun("PA_GuideLineE_d", PA_GuideLineE_d);
        fun("PA_GuideLineE_Xmin", PA_GuideLineE_Xmin);
        fun("PA_GuideLineE_Xmax", PA_GuideLineE_Xmax);
        fun("HourOfDay", HourOfDay);
        fun("MinuteOfHour", MinuteOfHour);
        fun("SecondOfMinute", SecondOfMinute);
        fun("NNS_distance", NNS_distance);
        fun("HPA_distance", HPA_distance);
        fun("Parkingtimeremaining", Parkingtimeremaining);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("PA_ParkBarPercent", PA_ParkBarPercent);
        fun("PA_GuideLineE_a", PA_GuideLineE_a);
        fun("PA_GuideLineE_b", PA_GuideLineE_b);
        fun("PA_GuideLineE_c", PA_GuideLineE_c);
        fun("PA_GuideLineE_d", PA_GuideLineE_d);
        fun("PA_GuideLineE_Xmin", PA_GuideLineE_Xmin);
        fun("PA_GuideLineE_Xmax", PA_GuideLineE_Xmax);
        fun("HourOfDay", HourOfDay);
        fun("MinuteOfHour", MinuteOfHour);
        fun("SecondOfMinute", SecondOfMinute);
        fun("NNS_distance", NNS_distance);
        fun("HPA_distance", HPA_distance);
        fun("Parkingtimeremaining", Parkingtimeremaining);
    }

    bool operator==(const ::hozon::chassis::AlgEgoParkHmiInfo& t) const
    {
        return (PA_ParkBarPercent == t.PA_ParkBarPercent) && (fabs(static_cast<double>(PA_GuideLineE_a - t.PA_GuideLineE_a)) < DBL_EPSILON) && (fabs(static_cast<double>(PA_GuideLineE_b - t.PA_GuideLineE_b)) < DBL_EPSILON) && (fabs(static_cast<double>(PA_GuideLineE_c - t.PA_GuideLineE_c)) < DBL_EPSILON) && (fabs(static_cast<double>(PA_GuideLineE_d - t.PA_GuideLineE_d)) < DBL_EPSILON) && (fabs(static_cast<double>(PA_GuideLineE_Xmin - t.PA_GuideLineE_Xmin)) < DBL_EPSILON) && (fabs(static_cast<double>(PA_GuideLineE_Xmax - t.PA_GuideLineE_Xmax)) < DBL_EPSILON) && (HourOfDay == t.HourOfDay) && (MinuteOfHour == t.MinuteOfHour) && (SecondOfMinute == t.SecondOfMinute) && (NNS_distance == t.NNS_distance) && (HPA_distance == t.HPA_distance) && (Parkingtimeremaining == t.Parkingtimeremaining);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGEGOPARKHMIINFO_H
