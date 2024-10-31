/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_LOCLANEFUSIONRESULTEXTERNAL_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_LOCLANEFUSIONRESULTEXTERNAL_STRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint64_t.h"
#include "impl_type_uint8_t.h"
#include "impl_type_uint16_t.h"

namespace hozon {
namespace hmi {
struct LocLaneFusionResultExternal_Struct {
    ::uint64_t TickTime;
    ::uint8_t Indices;
    ::uint8_t Padding_u8_1;
    ::uint16_t Padding_u16_1;
    float Probs;
    float Latera10ffsetLeft;
    float Latera10ffsetLeftACC;
    float Latera10ffsetRight;
    float Latera10ffsetRightACC;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(TickTime);
        fun(Indices);
        fun(Padding_u8_1);
        fun(Padding_u16_1);
        fun(Probs);
        fun(Latera10ffsetLeft);
        fun(Latera10ffsetLeftACC);
        fun(Latera10ffsetRight);
        fun(Latera10ffsetRightACC);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(TickTime);
        fun(Indices);
        fun(Padding_u8_1);
        fun(Padding_u16_1);
        fun(Probs);
        fun(Latera10ffsetLeft);
        fun(Latera10ffsetLeftACC);
        fun(Latera10ffsetRight);
        fun(Latera10ffsetRightACC);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("TickTime", TickTime);
        fun("Indices", Indices);
        fun("Padding_u8_1", Padding_u8_1);
        fun("Padding_u16_1", Padding_u16_1);
        fun("Probs", Probs);
        fun("Latera10ffsetLeft", Latera10ffsetLeft);
        fun("Latera10ffsetLeftACC", Latera10ffsetLeftACC);
        fun("Latera10ffsetRight", Latera10ffsetRight);
        fun("Latera10ffsetRightACC", Latera10ffsetRightACC);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("TickTime", TickTime);
        fun("Indices", Indices);
        fun("Padding_u8_1", Padding_u8_1);
        fun("Padding_u16_1", Padding_u16_1);
        fun("Probs", Probs);
        fun("Latera10ffsetLeft", Latera10ffsetLeft);
        fun("Latera10ffsetLeftACC", Latera10ffsetLeftACC);
        fun("Latera10ffsetRight", Latera10ffsetRight);
        fun("Latera10ffsetRightACC", Latera10ffsetRightACC);
    }

    bool operator==(const ::hozon::hmi::LocLaneFusionResultExternal_Struct& t) const
    {
        return (TickTime == t.TickTime) && (Indices == t.Indices) && (Padding_u8_1 == t.Padding_u8_1) && (Padding_u16_1 == t.Padding_u16_1) && (fabs(static_cast<double>(Probs - t.Probs)) < DBL_EPSILON) && (fabs(static_cast<double>(Latera10ffsetLeft - t.Latera10ffsetLeft)) < DBL_EPSILON) && (fabs(static_cast<double>(Latera10ffsetLeftACC - t.Latera10ffsetLeftACC)) < DBL_EPSILON) && (fabs(static_cast<double>(Latera10ffsetRight - t.Latera10ffsetRight)) < DBL_EPSILON) && (fabs(static_cast<double>(Latera10ffsetRightACC - t.Latera10ffsetRightACC)) < DBL_EPSILON);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_LOCLANEFUSIONRESULTEXTERNAL_STRUCT_H
