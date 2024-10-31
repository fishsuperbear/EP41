/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGFAULTDIDINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGFAULTDIDINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace chassis {
struct AlgFaultDidInfo {
    bool BDCS10_AC_OutsideTempValid;
    ::Float BDCS10_AC_OutsideTemp;
    ::uint8_t Power_Supply_Voltage;
    bool ICU1_VehicleSpdDisplayValid;
    ::Float ICU1_VehicleSpdDisplay;
    ::Float ICU2_Odometer;
    ::uint8_t BDCS1_PowerManageMode;
    ::uint8_t Ignition_status;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(BDCS10_AC_OutsideTempValid);
        fun(BDCS10_AC_OutsideTemp);
        fun(Power_Supply_Voltage);
        fun(ICU1_VehicleSpdDisplayValid);
        fun(ICU1_VehicleSpdDisplay);
        fun(ICU2_Odometer);
        fun(BDCS1_PowerManageMode);
        fun(Ignition_status);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(BDCS10_AC_OutsideTempValid);
        fun(BDCS10_AC_OutsideTemp);
        fun(Power_Supply_Voltage);
        fun(ICU1_VehicleSpdDisplayValid);
        fun(ICU1_VehicleSpdDisplay);
        fun(ICU2_Odometer);
        fun(BDCS1_PowerManageMode);
        fun(Ignition_status);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("BDCS10_AC_OutsideTempValid", BDCS10_AC_OutsideTempValid);
        fun("BDCS10_AC_OutsideTemp", BDCS10_AC_OutsideTemp);
        fun("Power_Supply_Voltage", Power_Supply_Voltage);
        fun("ICU1_VehicleSpdDisplayValid", ICU1_VehicleSpdDisplayValid);
        fun("ICU1_VehicleSpdDisplay", ICU1_VehicleSpdDisplay);
        fun("ICU2_Odometer", ICU2_Odometer);
        fun("BDCS1_PowerManageMode", BDCS1_PowerManageMode);
        fun("Ignition_status", Ignition_status);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("BDCS10_AC_OutsideTempValid", BDCS10_AC_OutsideTempValid);
        fun("BDCS10_AC_OutsideTemp", BDCS10_AC_OutsideTemp);
        fun("Power_Supply_Voltage", Power_Supply_Voltage);
        fun("ICU1_VehicleSpdDisplayValid", ICU1_VehicleSpdDisplayValid);
        fun("ICU1_VehicleSpdDisplay", ICU1_VehicleSpdDisplay);
        fun("ICU2_Odometer", ICU2_Odometer);
        fun("BDCS1_PowerManageMode", BDCS1_PowerManageMode);
        fun("Ignition_status", Ignition_status);
    }

    bool operator==(const ::hozon::chassis::AlgFaultDidInfo& t) const
    {
        return (BDCS10_AC_OutsideTempValid == t.BDCS10_AC_OutsideTempValid) && (fabs(static_cast<double>(BDCS10_AC_OutsideTemp - t.BDCS10_AC_OutsideTemp)) < DBL_EPSILON) && (Power_Supply_Voltage == t.Power_Supply_Voltage) && (ICU1_VehicleSpdDisplayValid == t.ICU1_VehicleSpdDisplayValid) && (fabs(static_cast<double>(ICU1_VehicleSpdDisplay - t.ICU1_VehicleSpdDisplay)) < DBL_EPSILON) && (fabs(static_cast<double>(ICU2_Odometer - t.ICU2_Odometer)) < DBL_EPSILON) && (BDCS1_PowerManageMode == t.BDCS1_PowerManageMode) && (Ignition_status == t.Ignition_status);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGFAULTDIDINFO_H
