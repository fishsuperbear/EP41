/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_APAINFO_H
#define ARA_VEHICLE_IMPL_TYPE_APAINFO_H
#include <cfloat>
#include <cmath>
#include "ara/vehicle/impl_type_time.h"
#include "ara/vehicle/impl_type_apafr01info.h"
#include "ara/vehicle/impl_type_apafr02info.h"
#include "ara/vehicle/impl_type_apafr03info.h"
#include "ara/vehicle/impl_type_apafr04info.h"

namespace ara {
namespace vehicle {
struct APAInfo {
    ::ara::vehicle::Time time;
    ::ara::vehicle::APAFr01Info apa_fr01_info;
    ::ara::vehicle::APAFr02Info apa_fr02_info;
    ::ara::vehicle::APAFr03Info apa_fr03_info;
    ::ara::vehicle::APAFr04Info apa_fr04_info;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(time);
        fun(apa_fr01_info);
        fun(apa_fr02_info);
        fun(apa_fr03_info);
        fun(apa_fr04_info);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(time);
        fun(apa_fr01_info);
        fun(apa_fr02_info);
        fun(apa_fr03_info);
        fun(apa_fr04_info);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("time", time);
        fun("apa_fr01_info", apa_fr01_info);
        fun("apa_fr02_info", apa_fr02_info);
        fun("apa_fr03_info", apa_fr03_info);
        fun("apa_fr04_info", apa_fr04_info);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("time", time);
        fun("apa_fr01_info", apa_fr01_info);
        fun("apa_fr02_info", apa_fr02_info);
        fun("apa_fr03_info", apa_fr03_info);
        fun("apa_fr04_info", apa_fr04_info);
    }

    bool operator==(const ::ara::vehicle::APAInfo& t) const
    {
        return (time == t.time) && (apa_fr01_info == t.apa_fr01_info) && (apa_fr02_info == t.apa_fr02_info) && (apa_fr03_info == t.apa_fr03_info) && (apa_fr04_info == t.apa_fr04_info);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_APAINFO_H
