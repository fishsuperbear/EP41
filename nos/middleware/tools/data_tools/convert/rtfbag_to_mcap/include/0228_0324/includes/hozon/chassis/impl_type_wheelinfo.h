/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_WHEELINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_WHEELINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "impl_type_boolean.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace hozon {
namespace chassis {
struct WheelInfo {
    ::Double ESC_FLWheelSpeed;
    ::Boolean ESC_FLWheelSpeedValid;
    ::UInt8 ESC_FLWheelDirection;
    ::Double ESC_FRWheelSpeed;
    ::Boolean ESC_FRWheelSpeedValid;
    ::UInt8 ESC_FRWheelDirection;
    ::Double ESC_RLWheelSpeed;
    ::Boolean ESC_RLWheelSpeedValid;
    ::UInt8 ESC_RLWheelDirection;
    ::Double ESC_RRWheelSpeed;
    ::Boolean ESC_RRWheelSpeedValid;
    ::UInt8 ESC_RRWheelDirection;
    ::Float ESC_FL_WhlPulCnt;
    ::Float ESC_FR_WhlPulCnt;
    ::Float ESC_RL_WhlPulCnt;
    ::Float ESC_RR_WhlPulCnt;
    ::Boolean ESC_FL_WhlPulCntValid;
    ::Boolean ESC_FR_WhlPulCntValid;
    ::Boolean ESC_RL_WhlPulCntValid;
    ::Boolean ESC_RR_WhlPulCntValid;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ESC_FLWheelSpeed);
        fun(ESC_FLWheelSpeedValid);
        fun(ESC_FLWheelDirection);
        fun(ESC_FRWheelSpeed);
        fun(ESC_FRWheelSpeedValid);
        fun(ESC_FRWheelDirection);
        fun(ESC_RLWheelSpeed);
        fun(ESC_RLWheelSpeedValid);
        fun(ESC_RLWheelDirection);
        fun(ESC_RRWheelSpeed);
        fun(ESC_RRWheelSpeedValid);
        fun(ESC_RRWheelDirection);
        fun(ESC_FL_WhlPulCnt);
        fun(ESC_FR_WhlPulCnt);
        fun(ESC_RL_WhlPulCnt);
        fun(ESC_RR_WhlPulCnt);
        fun(ESC_FL_WhlPulCntValid);
        fun(ESC_FR_WhlPulCntValid);
        fun(ESC_RL_WhlPulCntValid);
        fun(ESC_RR_WhlPulCntValid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ESC_FLWheelSpeed);
        fun(ESC_FLWheelSpeedValid);
        fun(ESC_FLWheelDirection);
        fun(ESC_FRWheelSpeed);
        fun(ESC_FRWheelSpeedValid);
        fun(ESC_FRWheelDirection);
        fun(ESC_RLWheelSpeed);
        fun(ESC_RLWheelSpeedValid);
        fun(ESC_RLWheelDirection);
        fun(ESC_RRWheelSpeed);
        fun(ESC_RRWheelSpeedValid);
        fun(ESC_RRWheelDirection);
        fun(ESC_FL_WhlPulCnt);
        fun(ESC_FR_WhlPulCnt);
        fun(ESC_RL_WhlPulCnt);
        fun(ESC_RR_WhlPulCnt);
        fun(ESC_FL_WhlPulCntValid);
        fun(ESC_FR_WhlPulCntValid);
        fun(ESC_RL_WhlPulCntValid);
        fun(ESC_RR_WhlPulCntValid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ESC_FLWheelSpeed", ESC_FLWheelSpeed);
        fun("ESC_FLWheelSpeedValid", ESC_FLWheelSpeedValid);
        fun("ESC_FLWheelDirection", ESC_FLWheelDirection);
        fun("ESC_FRWheelSpeed", ESC_FRWheelSpeed);
        fun("ESC_FRWheelSpeedValid", ESC_FRWheelSpeedValid);
        fun("ESC_FRWheelDirection", ESC_FRWheelDirection);
        fun("ESC_RLWheelSpeed", ESC_RLWheelSpeed);
        fun("ESC_RLWheelSpeedValid", ESC_RLWheelSpeedValid);
        fun("ESC_RLWheelDirection", ESC_RLWheelDirection);
        fun("ESC_RRWheelSpeed", ESC_RRWheelSpeed);
        fun("ESC_RRWheelSpeedValid", ESC_RRWheelSpeedValid);
        fun("ESC_RRWheelDirection", ESC_RRWheelDirection);
        fun("ESC_FL_WhlPulCnt", ESC_FL_WhlPulCnt);
        fun("ESC_FR_WhlPulCnt", ESC_FR_WhlPulCnt);
        fun("ESC_RL_WhlPulCnt", ESC_RL_WhlPulCnt);
        fun("ESC_RR_WhlPulCnt", ESC_RR_WhlPulCnt);
        fun("ESC_FL_WhlPulCntValid", ESC_FL_WhlPulCntValid);
        fun("ESC_FR_WhlPulCntValid", ESC_FR_WhlPulCntValid);
        fun("ESC_RL_WhlPulCntValid", ESC_RL_WhlPulCntValid);
        fun("ESC_RR_WhlPulCntValid", ESC_RR_WhlPulCntValid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ESC_FLWheelSpeed", ESC_FLWheelSpeed);
        fun("ESC_FLWheelSpeedValid", ESC_FLWheelSpeedValid);
        fun("ESC_FLWheelDirection", ESC_FLWheelDirection);
        fun("ESC_FRWheelSpeed", ESC_FRWheelSpeed);
        fun("ESC_FRWheelSpeedValid", ESC_FRWheelSpeedValid);
        fun("ESC_FRWheelDirection", ESC_FRWheelDirection);
        fun("ESC_RLWheelSpeed", ESC_RLWheelSpeed);
        fun("ESC_RLWheelSpeedValid", ESC_RLWheelSpeedValid);
        fun("ESC_RLWheelDirection", ESC_RLWheelDirection);
        fun("ESC_RRWheelSpeed", ESC_RRWheelSpeed);
        fun("ESC_RRWheelSpeedValid", ESC_RRWheelSpeedValid);
        fun("ESC_RRWheelDirection", ESC_RRWheelDirection);
        fun("ESC_FL_WhlPulCnt", ESC_FL_WhlPulCnt);
        fun("ESC_FR_WhlPulCnt", ESC_FR_WhlPulCnt);
        fun("ESC_RL_WhlPulCnt", ESC_RL_WhlPulCnt);
        fun("ESC_RR_WhlPulCnt", ESC_RR_WhlPulCnt);
        fun("ESC_FL_WhlPulCntValid", ESC_FL_WhlPulCntValid);
        fun("ESC_FR_WhlPulCntValid", ESC_FR_WhlPulCntValid);
        fun("ESC_RL_WhlPulCntValid", ESC_RL_WhlPulCntValid);
        fun("ESC_RR_WhlPulCntValid", ESC_RR_WhlPulCntValid);
    }

    bool operator==(const ::hozon::chassis::WheelInfo& t) const
    {
        return (fabs(static_cast<double>(ESC_FLWheelSpeed - t.ESC_FLWheelSpeed)) < DBL_EPSILON) && (ESC_FLWheelSpeedValid == t.ESC_FLWheelSpeedValid) && (ESC_FLWheelDirection == t.ESC_FLWheelDirection) && (fabs(static_cast<double>(ESC_FRWheelSpeed - t.ESC_FRWheelSpeed)) < DBL_EPSILON) && (ESC_FRWheelSpeedValid == t.ESC_FRWheelSpeedValid) && (ESC_FRWheelDirection == t.ESC_FRWheelDirection) && (fabs(static_cast<double>(ESC_RLWheelSpeed - t.ESC_RLWheelSpeed)) < DBL_EPSILON) && (ESC_RLWheelSpeedValid == t.ESC_RLWheelSpeedValid) && (ESC_RLWheelDirection == t.ESC_RLWheelDirection) && (fabs(static_cast<double>(ESC_RRWheelSpeed - t.ESC_RRWheelSpeed)) < DBL_EPSILON) && (ESC_RRWheelSpeedValid == t.ESC_RRWheelSpeedValid) && (ESC_RRWheelDirection == t.ESC_RRWheelDirection) && (fabs(static_cast<double>(ESC_FL_WhlPulCnt - t.ESC_FL_WhlPulCnt)) < DBL_EPSILON) && (fabs(static_cast<double>(ESC_FR_WhlPulCnt - t.ESC_FR_WhlPulCnt)) < DBL_EPSILON) && (fabs(static_cast<double>(ESC_RL_WhlPulCnt - t.ESC_RL_WhlPulCnt)) < DBL_EPSILON) && (fabs(static_cast<double>(ESC_RR_WhlPulCnt - t.ESC_RR_WhlPulCnt)) < DBL_EPSILON) && (ESC_FL_WhlPulCntValid == t.ESC_FL_WhlPulCntValid) && (ESC_FR_WhlPulCntValid == t.ESC_FR_WhlPulCntValid) && (ESC_RL_WhlPulCntValid == t.ESC_RL_WhlPulCntValid) && (ESC_RR_WhlPulCntValid == t.ESC_RR_WhlPulCntValid);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_WHEELINFO_H
