/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_FUNCTIONSTATE_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_FUNCTIONSTATE_STRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace hmi {
struct FunctionState_Struct {
    ::uint8_t LaneChangeStatus;
    ::uint8_t LaneChangedType;
    ::uint8_t DriveMode;
    ::uint8_t Padding_u8_1;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(LaneChangeStatus);
        fun(LaneChangedType);
        fun(DriveMode);
        fun(Padding_u8_1);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(LaneChangeStatus);
        fun(LaneChangedType);
        fun(DriveMode);
        fun(Padding_u8_1);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("LaneChangeStatus", LaneChangeStatus);
        fun("LaneChangedType", LaneChangedType);
        fun("DriveMode", DriveMode);
        fun("Padding_u8_1", Padding_u8_1);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("LaneChangeStatus", LaneChangeStatus);
        fun("LaneChangedType", LaneChangedType);
        fun("DriveMode", DriveMode);
        fun("Padding_u8_1", Padding_u8_1);
    }

    bool operator==(const ::hozon::hmi::FunctionState_Struct& t) const
    {
        return (LaneChangeStatus == t.LaneChangeStatus) && (LaneChangedType == t.LaneChangedType) && (DriveMode == t.DriveMode) && (Padding_u8_1 == t.Padding_u8_1);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_FUNCTIONSTATE_STRUCT_H
