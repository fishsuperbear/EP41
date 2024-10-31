/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ESCDRIVINGINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_ESCDRIVINGINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace chassis {
struct EscDrivingInfo {
    ::Double ESC_VehicleSpeed;
    ::Boolean ESC_VehicleSpeedValid;
    ::Boolean ESC_BrakePedalSwitchStatus;
    ::Boolean ESC_BrakePedalSwitchValid;
    float BrkPedVal;
    ::Double VehicleSpdDisplay;
    ::Boolean VehicleSpdDisplayValid;
    ::Boolean ESC_ApaStandStill;
    ::Double ESC_LongAccValue;
    ::Boolean ESC_LongAccValue_Valid;
    ::Double ESC_LatAccValue;
    ::Boolean ESC_LatAccValue_Valid;
    ::Double ESC_YawRate;
    ::Boolean ESC_YawRate_Valid;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ESC_VehicleSpeed);
        fun(ESC_VehicleSpeedValid);
        fun(ESC_BrakePedalSwitchStatus);
        fun(ESC_BrakePedalSwitchValid);
        fun(BrkPedVal);
        fun(VehicleSpdDisplay);
        fun(VehicleSpdDisplayValid);
        fun(ESC_ApaStandStill);
        fun(ESC_LongAccValue);
        fun(ESC_LongAccValue_Valid);
        fun(ESC_LatAccValue);
        fun(ESC_LatAccValue_Valid);
        fun(ESC_YawRate);
        fun(ESC_YawRate_Valid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ESC_VehicleSpeed);
        fun(ESC_VehicleSpeedValid);
        fun(ESC_BrakePedalSwitchStatus);
        fun(ESC_BrakePedalSwitchValid);
        fun(BrkPedVal);
        fun(VehicleSpdDisplay);
        fun(VehicleSpdDisplayValid);
        fun(ESC_ApaStandStill);
        fun(ESC_LongAccValue);
        fun(ESC_LongAccValue_Valid);
        fun(ESC_LatAccValue);
        fun(ESC_LatAccValue_Valid);
        fun(ESC_YawRate);
        fun(ESC_YawRate_Valid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ESC_VehicleSpeed", ESC_VehicleSpeed);
        fun("ESC_VehicleSpeedValid", ESC_VehicleSpeedValid);
        fun("ESC_BrakePedalSwitchStatus", ESC_BrakePedalSwitchStatus);
        fun("ESC_BrakePedalSwitchValid", ESC_BrakePedalSwitchValid);
        fun("BrkPedVal", BrkPedVal);
        fun("VehicleSpdDisplay", VehicleSpdDisplay);
        fun("VehicleSpdDisplayValid", VehicleSpdDisplayValid);
        fun("ESC_ApaStandStill", ESC_ApaStandStill);
        fun("ESC_LongAccValue", ESC_LongAccValue);
        fun("ESC_LongAccValue_Valid", ESC_LongAccValue_Valid);
        fun("ESC_LatAccValue", ESC_LatAccValue);
        fun("ESC_LatAccValue_Valid", ESC_LatAccValue_Valid);
        fun("ESC_YawRate", ESC_YawRate);
        fun("ESC_YawRate_Valid", ESC_YawRate_Valid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ESC_VehicleSpeed", ESC_VehicleSpeed);
        fun("ESC_VehicleSpeedValid", ESC_VehicleSpeedValid);
        fun("ESC_BrakePedalSwitchStatus", ESC_BrakePedalSwitchStatus);
        fun("ESC_BrakePedalSwitchValid", ESC_BrakePedalSwitchValid);
        fun("BrkPedVal", BrkPedVal);
        fun("VehicleSpdDisplay", VehicleSpdDisplay);
        fun("VehicleSpdDisplayValid", VehicleSpdDisplayValid);
        fun("ESC_ApaStandStill", ESC_ApaStandStill);
        fun("ESC_LongAccValue", ESC_LongAccValue);
        fun("ESC_LongAccValue_Valid", ESC_LongAccValue_Valid);
        fun("ESC_LatAccValue", ESC_LatAccValue);
        fun("ESC_LatAccValue_Valid", ESC_LatAccValue_Valid);
        fun("ESC_YawRate", ESC_YawRate);
        fun("ESC_YawRate_Valid", ESC_YawRate_Valid);
    }

    bool operator==(const ::hozon::chassis::EscDrivingInfo& t) const
    {
        return (fabs(static_cast<double>(ESC_VehicleSpeed - t.ESC_VehicleSpeed)) < DBL_EPSILON) && (ESC_VehicleSpeedValid == t.ESC_VehicleSpeedValid) && (ESC_BrakePedalSwitchStatus == t.ESC_BrakePedalSwitchStatus) && (ESC_BrakePedalSwitchValid == t.ESC_BrakePedalSwitchValid) && (fabs(static_cast<double>(BrkPedVal - t.BrkPedVal)) < DBL_EPSILON) && (fabs(static_cast<double>(VehicleSpdDisplay - t.VehicleSpdDisplay)) < DBL_EPSILON) && (VehicleSpdDisplayValid == t.VehicleSpdDisplayValid) && (ESC_ApaStandStill == t.ESC_ApaStandStill) && (fabs(static_cast<double>(ESC_LongAccValue - t.ESC_LongAccValue)) < DBL_EPSILON) && (ESC_LongAccValue_Valid == t.ESC_LongAccValue_Valid) && (fabs(static_cast<double>(ESC_LatAccValue - t.ESC_LatAccValue)) < DBL_EPSILON) && (ESC_LatAccValue_Valid == t.ESC_LatAccValue_Valid) && (fabs(static_cast<double>(ESC_YawRate - t.ESC_YawRate)) < DBL_EPSILON) && (ESC_YawRate_Valid == t.ESC_YawRate_Valid);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ESCDRIVINGINFO_H
