/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGEGOMCUAVPMSG_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGEGOMCUAVPMSG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace chassis {
struct AlgEgoMcuAVPMsg {
    ::UInt8 m_iuss_state_obs;
    bool need_replan_stop;
    bool plan_trigger;
    bool control_enable;
    ::UInt8 parking_status;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(m_iuss_state_obs);
        fun(need_replan_stop);
        fun(plan_trigger);
        fun(control_enable);
        fun(parking_status);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(m_iuss_state_obs);
        fun(need_replan_stop);
        fun(plan_trigger);
        fun(control_enable);
        fun(parking_status);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("m_iuss_state_obs", m_iuss_state_obs);
        fun("need_replan_stop", need_replan_stop);
        fun("plan_trigger", plan_trigger);
        fun("control_enable", control_enable);
        fun("parking_status", parking_status);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("m_iuss_state_obs", m_iuss_state_obs);
        fun("need_replan_stop", need_replan_stop);
        fun("plan_trigger", plan_trigger);
        fun("control_enable", control_enable);
        fun("parking_status", parking_status);
    }

    bool operator==(const ::hozon::chassis::AlgEgoMcuAVPMsg& t) const
    {
        return (m_iuss_state_obs == t.m_iuss_state_obs) && (need_replan_stop == t.need_replan_stop) && (plan_trigger == t.plan_trigger) && (control_enable == t.control_enable) && (parking_status == t.parking_status);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGEGOMCUAVPMSG_H
