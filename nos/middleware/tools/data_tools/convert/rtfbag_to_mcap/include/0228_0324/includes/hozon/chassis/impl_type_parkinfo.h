/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_PARKINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_PARKINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace chassis {
struct ParkInfo {
    ::Boolean TCSActive;
    ::Boolean ABSActive;
    ::Boolean ARPActive;
    ::Boolean ESCActive;
    ::UInt8 EPBStatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(TCSActive);
        fun(ABSActive);
        fun(ARPActive);
        fun(ESCActive);
        fun(EPBStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(TCSActive);
        fun(ABSActive);
        fun(ARPActive);
        fun(ESCActive);
        fun(EPBStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("TCSActive", TCSActive);
        fun("ABSActive", ABSActive);
        fun("ARPActive", ARPActive);
        fun("ESCActive", ESCActive);
        fun("EPBStatus", EPBStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("TCSActive", TCSActive);
        fun("ABSActive", ABSActive);
        fun("ARPActive", ARPActive);
        fun("ESCActive", ESCActive);
        fun("EPBStatus", EPBStatus);
    }

    bool operator==(const ::hozon::chassis::ParkInfo& t) const
    {
        return (TCSActive == t.TCSActive) && (ABSActive == t.ABSActive) && (ARPActive == t.ARPActive) && (ESCActive == t.ESCActive) && (EPBStatus == t.EPBStatus);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_PARKINFO_H
