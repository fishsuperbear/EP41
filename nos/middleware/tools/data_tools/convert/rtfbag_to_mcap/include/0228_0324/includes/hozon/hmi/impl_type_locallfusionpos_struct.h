/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_LOCALLFUSIONPOS_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_LOCALLFUSIONPOS_STRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint64_t.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace hmi {
struct LocAllFusionPos_Struct {
    ::uint64_t TickTime;
    ::uint8_t Status;
    ::uint8_t Ns;
    ::uint8_t Ew;
    ::uint8_t FusionType;
    float PosEnu_Longitude;
    float PosEnu_Latitude;
    float Speed;
    float Course;
    float Alt;
    float PosAcc;
    float CourseAcc;
    float AltAcc;
    float SpeedAcc;
    ::uint64_t DataTime;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(TickTime);
        fun(Status);
        fun(Ns);
        fun(Ew);
        fun(FusionType);
        fun(PosEnu_Longitude);
        fun(PosEnu_Latitude);
        fun(Speed);
        fun(Course);
        fun(Alt);
        fun(PosAcc);
        fun(CourseAcc);
        fun(AltAcc);
        fun(SpeedAcc);
        fun(DataTime);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(TickTime);
        fun(Status);
        fun(Ns);
        fun(Ew);
        fun(FusionType);
        fun(PosEnu_Longitude);
        fun(PosEnu_Latitude);
        fun(Speed);
        fun(Course);
        fun(Alt);
        fun(PosAcc);
        fun(CourseAcc);
        fun(AltAcc);
        fun(SpeedAcc);
        fun(DataTime);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("TickTime", TickTime);
        fun("Status", Status);
        fun("Ns", Ns);
        fun("Ew", Ew);
        fun("FusionType", FusionType);
        fun("PosEnu_Longitude", PosEnu_Longitude);
        fun("PosEnu_Latitude", PosEnu_Latitude);
        fun("Speed", Speed);
        fun("Course", Course);
        fun("Alt", Alt);
        fun("PosAcc", PosAcc);
        fun("CourseAcc", CourseAcc);
        fun("AltAcc", AltAcc);
        fun("SpeedAcc", SpeedAcc);
        fun("DataTime", DataTime);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("TickTime", TickTime);
        fun("Status", Status);
        fun("Ns", Ns);
        fun("Ew", Ew);
        fun("FusionType", FusionType);
        fun("PosEnu_Longitude", PosEnu_Longitude);
        fun("PosEnu_Latitude", PosEnu_Latitude);
        fun("Speed", Speed);
        fun("Course", Course);
        fun("Alt", Alt);
        fun("PosAcc", PosAcc);
        fun("CourseAcc", CourseAcc);
        fun("AltAcc", AltAcc);
        fun("SpeedAcc", SpeedAcc);
        fun("DataTime", DataTime);
    }

    bool operator==(const ::hozon::hmi::LocAllFusionPos_Struct& t) const
    {
        return (TickTime == t.TickTime) && (Status == t.Status) && (Ns == t.Ns) && (Ew == t.Ew) && (FusionType == t.FusionType) && (fabs(static_cast<double>(PosEnu_Longitude - t.PosEnu_Longitude)) < DBL_EPSILON) && (fabs(static_cast<double>(PosEnu_Latitude - t.PosEnu_Latitude)) < DBL_EPSILON) && (fabs(static_cast<double>(Speed - t.Speed)) < DBL_EPSILON) && (fabs(static_cast<double>(Course - t.Course)) < DBL_EPSILON) && (fabs(static_cast<double>(Alt - t.Alt)) < DBL_EPSILON) && (fabs(static_cast<double>(PosAcc - t.PosAcc)) < DBL_EPSILON) && (fabs(static_cast<double>(CourseAcc - t.CourseAcc)) < DBL_EPSILON) && (fabs(static_cast<double>(AltAcc - t.AltAcc)) < DBL_EPSILON) && (fabs(static_cast<double>(SpeedAcc - t.SpeedAcc)) < DBL_EPSILON) && (DataTime == t.DataTime);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_LOCALLFUSIONPOS_STRUCT_H
