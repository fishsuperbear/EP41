/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_USSINFO_H
#define HOZON_SENSORS_IMPL_TYPE_USSINFO_H
#include <cfloat>
#include <cmath>
#include "hozon/sensors/impl_type_apa_info_t.h"
#include "hozon/sensors/impl_type_upa_info_arry2.h"
#include "hozon/sensors/impl_type_tracker_data_arry50.h"
#include "hozon/sensors/impl_type_tpointinfo_arry_5.h"
#include "hozon/composite/impl_type_uint16arry_12.h"

namespace hozon {
namespace sensors {
struct UssInfo {
    ::hozon::sensors::APA_Info_T APA_Inf;
    ::hozon::sensors::UPA_Info_Arry2 UPA_Info;
    ::hozon::sensors::Tracker_Data_Arry50 Tracker_Data;
    ::hozon::sensors::tPointInfo_Arry_5 APA_Virtual_Point;
    ::hozon::composite::uint16Arry_12 reserved1;
    ::hozon::composite::uint16Arry_12 reserved2;
    ::hozon::composite::uint16Arry_12 reserved3;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(APA_Inf);
        fun(UPA_Info);
        fun(Tracker_Data);
        fun(APA_Virtual_Point);
        fun(reserved1);
        fun(reserved2);
        fun(reserved3);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(APA_Inf);
        fun(UPA_Info);
        fun(Tracker_Data);
        fun(APA_Virtual_Point);
        fun(reserved1);
        fun(reserved2);
        fun(reserved3);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("APA_Inf", APA_Inf);
        fun("UPA_Info", UPA_Info);
        fun("Tracker_Data", Tracker_Data);
        fun("APA_Virtual_Point", APA_Virtual_Point);
        fun("reserved1", reserved1);
        fun("reserved2", reserved2);
        fun("reserved3", reserved3);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("APA_Inf", APA_Inf);
        fun("UPA_Info", UPA_Info);
        fun("Tracker_Data", Tracker_Data);
        fun("APA_Virtual_Point", APA_Virtual_Point);
        fun("reserved1", reserved1);
        fun("reserved2", reserved2);
        fun("reserved3", reserved3);
    }

    bool operator==(const ::hozon::sensors::UssInfo& t) const
    {
        return (APA_Inf == t.APA_Inf) && (UPA_Info == t.UPA_Info) && (Tracker_Data == t.Tracker_Data) && (APA_Virtual_Point == t.APA_Virtual_Point) && (reserved1 == t.reserved1) && (reserved2 == t.reserved2) && (reserved3 == t.reserved3);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_USSINFO_H
