/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_FLCINFO_H
#define ARA_VEHICLE_IMPL_TYPE_FLCINFO_H
#include <cfloat>
#include <cmath>
#include "ara/vehicle/impl_type_time.h"
#include "ara/vehicle/impl_type_flcfr01info.h"
#include "ara/vehicle/impl_type_flcfr02info.h"

namespace ara {
namespace vehicle {
struct FLCInfo {
    ::ara::vehicle::Time time;
    ::ara::vehicle::FLCFr01Info flc_fr01_info;
    ::ara::vehicle::FLCFr02Info flc_fr02_info;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(time);
        fun(flc_fr01_info);
        fun(flc_fr02_info);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(time);
        fun(flc_fr01_info);
        fun(flc_fr02_info);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("time", time);
        fun("flc_fr01_info", flc_fr01_info);
        fun("flc_fr02_info", flc_fr02_info);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("time", time);
        fun("flc_fr01_info", flc_fr01_info);
        fun("flc_fr02_info", flc_fr02_info);
    }

    bool operator==(const ::ara::vehicle::FLCInfo& t) const
    {
        return (time == t.time) && (flc_fr01_info == t.flc_fr01_info) && (flc_fr02_info == t.flc_fr02_info);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_FLCINFO_H
