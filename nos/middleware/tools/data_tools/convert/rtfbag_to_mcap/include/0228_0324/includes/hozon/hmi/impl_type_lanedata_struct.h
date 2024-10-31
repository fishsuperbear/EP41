/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_LANEDATA_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_LANEDATA_STRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace hmi {
struct LaneData_Struct {
    ::uint8_t Lane_State;
    ::uint8_t Lane_Color;
    ::uint8_t Lane_Type;
    ::uint8_t Lane_ID;
    float Lane_Equation_C0;
    float Lane_Equation_C1;
    float Lane_Equation_C2;
    float Lane_Equation_C3;
    float Lane_Width;
    float LaneLineWidth;
    float Lane_Start_X;
    float Lane_Start_Y;
    float Lane_End_X;
    float Lane_End_Y;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Lane_State);
        fun(Lane_Color);
        fun(Lane_Type);
        fun(Lane_ID);
        fun(Lane_Equation_C0);
        fun(Lane_Equation_C1);
        fun(Lane_Equation_C2);
        fun(Lane_Equation_C3);
        fun(Lane_Width);
        fun(LaneLineWidth);
        fun(Lane_Start_X);
        fun(Lane_Start_Y);
        fun(Lane_End_X);
        fun(Lane_End_Y);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Lane_State);
        fun(Lane_Color);
        fun(Lane_Type);
        fun(Lane_ID);
        fun(Lane_Equation_C0);
        fun(Lane_Equation_C1);
        fun(Lane_Equation_C2);
        fun(Lane_Equation_C3);
        fun(Lane_Width);
        fun(LaneLineWidth);
        fun(Lane_Start_X);
        fun(Lane_Start_Y);
        fun(Lane_End_X);
        fun(Lane_End_Y);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Lane_State", Lane_State);
        fun("Lane_Color", Lane_Color);
        fun("Lane_Type", Lane_Type);
        fun("Lane_ID", Lane_ID);
        fun("Lane_Equation_C0", Lane_Equation_C0);
        fun("Lane_Equation_C1", Lane_Equation_C1);
        fun("Lane_Equation_C2", Lane_Equation_C2);
        fun("Lane_Equation_C3", Lane_Equation_C3);
        fun("Lane_Width", Lane_Width);
        fun("LaneLineWidth", LaneLineWidth);
        fun("Lane_Start_X", Lane_Start_X);
        fun("Lane_Start_Y", Lane_Start_Y);
        fun("Lane_End_X", Lane_End_X);
        fun("Lane_End_Y", Lane_End_Y);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Lane_State", Lane_State);
        fun("Lane_Color", Lane_Color);
        fun("Lane_Type", Lane_Type);
        fun("Lane_ID", Lane_ID);
        fun("Lane_Equation_C0", Lane_Equation_C0);
        fun("Lane_Equation_C1", Lane_Equation_C1);
        fun("Lane_Equation_C2", Lane_Equation_C2);
        fun("Lane_Equation_C3", Lane_Equation_C3);
        fun("Lane_Width", Lane_Width);
        fun("LaneLineWidth", LaneLineWidth);
        fun("Lane_Start_X", Lane_Start_X);
        fun("Lane_Start_Y", Lane_Start_Y);
        fun("Lane_End_X", Lane_End_X);
        fun("Lane_End_Y", Lane_End_Y);
    }

    bool operator==(const ::hozon::hmi::LaneData_Struct& t) const
    {
        return (Lane_State == t.Lane_State) && (Lane_Color == t.Lane_Color) && (Lane_Type == t.Lane_Type) && (Lane_ID == t.Lane_ID) && (fabs(static_cast<double>(Lane_Equation_C0 - t.Lane_Equation_C0)) < DBL_EPSILON) && (fabs(static_cast<double>(Lane_Equation_C1 - t.Lane_Equation_C1)) < DBL_EPSILON) && (fabs(static_cast<double>(Lane_Equation_C2 - t.Lane_Equation_C2)) < DBL_EPSILON) && (fabs(static_cast<double>(Lane_Equation_C3 - t.Lane_Equation_C3)) < DBL_EPSILON) && (fabs(static_cast<double>(Lane_Width - t.Lane_Width)) < DBL_EPSILON) && (fabs(static_cast<double>(LaneLineWidth - t.LaneLineWidth)) < DBL_EPSILON) && (fabs(static_cast<double>(Lane_Start_X - t.Lane_Start_X)) < DBL_EPSILON) && (fabs(static_cast<double>(Lane_Start_Y - t.Lane_Start_Y)) < DBL_EPSILON) && (fabs(static_cast<double>(Lane_End_X - t.Lane_End_X)) < DBL_EPSILON) && (fabs(static_cast<double>(Lane_End_Y - t.Lane_End_Y)) < DBL_EPSILON);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_LANEDATA_STRUCT_H
