/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_FLRINFO_H
#define ARA_VEHICLE_IMPL_TYPE_FLRINFO_H
#include <cfloat>
#include <cmath>
#include "ara/vehicle/impl_type_time.h"
#include "ara/vehicle/impl_type_flrfr01info.h"
#include "ara/vehicle/impl_type_flrfr02info.h"
#include "ara/vehicle/impl_type_flrfr03info.h"

namespace ara {
namespace vehicle {
struct FLRInfo {
    ::ara::vehicle::Time time;
    ::ara::vehicle::FLRFr01Info flr_fr01_info;
    ::ara::vehicle::FLRFr02Info flr_fr02_info;
    ::ara::vehicle::FLRFr03Info flr_fr03_info;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(time);
        fun(flr_fr01_info);
        fun(flr_fr02_info);
        fun(flr_fr03_info);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(time);
        fun(flr_fr01_info);
        fun(flr_fr02_info);
        fun(flr_fr03_info);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("time", time);
        fun("flr_fr01_info", flr_fr01_info);
        fun("flr_fr02_info", flr_fr02_info);
        fun("flr_fr03_info", flr_fr03_info);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("time", time);
        fun("flr_fr01_info", flr_fr01_info);
        fun("flr_fr02_info", flr_fr02_info);
        fun("flr_fr03_info", flr_fr03_info);
    }

    bool operator==(const ::ara::vehicle::FLRInfo& t) const
    {
        return (time == t.time) && (flr_fr01_info == t.flr_fr01_info) && (flr_fr02_info == t.flr_fr02_info) && (flr_fr03_info == t.flr_fr03_info);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_FLRINFO_H
