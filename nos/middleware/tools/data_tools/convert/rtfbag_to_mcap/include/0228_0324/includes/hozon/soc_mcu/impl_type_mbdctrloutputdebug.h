/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MBDCTRLOUTPUTDEBUG_H
#define HOZON_SOC_MCU_IMPL_TYPE_MBDCTRLOUTPUTDEBUG_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct MbdCtrlOutputDebug {
    ::Float ctrlout_brake_cmd;
    ::Float ctrlout_throttle_cmd;
    ::Float ctrlout_acc_cmd;
    ::UInt8 ctrlout_gear_enable;
    ::UInt8 ctrlout_gear_cmd;
    ::UInt8 ctrlout_emerg_enable;
    ::Float ctrlout_steer_cmd;
    ::Float ctrlout_steer_torque_cmd;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ctrlout_brake_cmd);
        fun(ctrlout_throttle_cmd);
        fun(ctrlout_acc_cmd);
        fun(ctrlout_gear_enable);
        fun(ctrlout_gear_cmd);
        fun(ctrlout_emerg_enable);
        fun(ctrlout_steer_cmd);
        fun(ctrlout_steer_torque_cmd);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ctrlout_brake_cmd);
        fun(ctrlout_throttle_cmd);
        fun(ctrlout_acc_cmd);
        fun(ctrlout_gear_enable);
        fun(ctrlout_gear_cmd);
        fun(ctrlout_emerg_enable);
        fun(ctrlout_steer_cmd);
        fun(ctrlout_steer_torque_cmd);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ctrlout_brake_cmd", ctrlout_brake_cmd);
        fun("ctrlout_throttle_cmd", ctrlout_throttle_cmd);
        fun("ctrlout_acc_cmd", ctrlout_acc_cmd);
        fun("ctrlout_gear_enable", ctrlout_gear_enable);
        fun("ctrlout_gear_cmd", ctrlout_gear_cmd);
        fun("ctrlout_emerg_enable", ctrlout_emerg_enable);
        fun("ctrlout_steer_cmd", ctrlout_steer_cmd);
        fun("ctrlout_steer_torque_cmd", ctrlout_steer_torque_cmd);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ctrlout_brake_cmd", ctrlout_brake_cmd);
        fun("ctrlout_throttle_cmd", ctrlout_throttle_cmd);
        fun("ctrlout_acc_cmd", ctrlout_acc_cmd);
        fun("ctrlout_gear_enable", ctrlout_gear_enable);
        fun("ctrlout_gear_cmd", ctrlout_gear_cmd);
        fun("ctrlout_emerg_enable", ctrlout_emerg_enable);
        fun("ctrlout_steer_cmd", ctrlout_steer_cmd);
        fun("ctrlout_steer_torque_cmd", ctrlout_steer_torque_cmd);
    }

    bool operator==(const ::hozon::soc_mcu::MbdCtrlOutputDebug& t) const
    {
        return (fabs(static_cast<double>(ctrlout_brake_cmd - t.ctrlout_brake_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrlout_throttle_cmd - t.ctrlout_throttle_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrlout_acc_cmd - t.ctrlout_acc_cmd)) < DBL_EPSILON) && (ctrlout_gear_enable == t.ctrlout_gear_enable) && (ctrlout_gear_cmd == t.ctrlout_gear_cmd) && (ctrlout_emerg_enable == t.ctrlout_emerg_enable) && (fabs(static_cast<double>(ctrlout_steer_cmd - t.ctrlout_steer_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(ctrlout_steer_torque_cmd - t.ctrlout_steer_torque_cmd)) < DBL_EPSILON);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MBDCTRLOUTPUTDEBUG_H
