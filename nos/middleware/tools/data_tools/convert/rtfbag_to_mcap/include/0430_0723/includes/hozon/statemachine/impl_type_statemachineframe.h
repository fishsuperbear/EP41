/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STATEMACHINE_IMPL_TYPE_STATEMACHINEFRAME_H
#define HOZON_STATEMACHINE_IMPL_TYPE_STATEMACHINEFRAME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint64.h"
#include "impl_type_uint32.h"
#include "hozon/statemachine/impl_type_autopilotstatus.h"
#include "hozon/statemachine/impl_type_command.h"
#include "hozon/statemachine/impl_type_workingstatus.h"
#include "hozon/statemachine/impl_type_pnccontrolstate.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace statemachine {
struct StateMachineFrame {
    ::UInt64 timestamp;
    ::UInt32 counter;
    ::hozon::statemachine::AutopilotStatus pilot_status;
    ::hozon::statemachine::Command hpp_command;
    ::hozon::statemachine::WorkingStatus hpp_perception_status;
    ::hozon::statemachine::PNCControlState pnc_control_state;
    ::Boolean isValid;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(timestamp);
        fun(counter);
        fun(pilot_status);
        fun(hpp_command);
        fun(hpp_perception_status);
        fun(pnc_control_state);
        fun(isValid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(timestamp);
        fun(counter);
        fun(pilot_status);
        fun(hpp_command);
        fun(hpp_perception_status);
        fun(pnc_control_state);
        fun(isValid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("timestamp", timestamp);
        fun("counter", counter);
        fun("pilot_status", pilot_status);
        fun("hpp_command", hpp_command);
        fun("hpp_perception_status", hpp_perception_status);
        fun("pnc_control_state", pnc_control_state);
        fun("isValid", isValid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("timestamp", timestamp);
        fun("counter", counter);
        fun("pilot_status", pilot_status);
        fun("hpp_command", hpp_command);
        fun("hpp_perception_status", hpp_perception_status);
        fun("pnc_control_state", pnc_control_state);
        fun("isValid", isValid);
    }

    bool operator==(const ::hozon::statemachine::StateMachineFrame& t) const
    {
        return (timestamp == t.timestamp) && (counter == t.counter) && (pilot_status == t.pilot_status) && (hpp_command == t.hpp_command) && (hpp_perception_status == t.hpp_perception_status) && (pnc_control_state == t.pnc_control_state) && (isValid == t.isValid);
    }
};
} // namespace statemachine
} // namespace hozon


#endif // HOZON_STATEMACHINE_IMPL_TYPE_STATEMACHINEFRAME_H
