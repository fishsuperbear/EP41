/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGMCUEGOAVPMSG_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGMCUEGOAVPMSG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace chassis {
struct AlgMcuEgoAVPMsg {
    ::UInt8 AVPSysMode;
    ::UInt8 system_command;
    ::uint8_t avp_run_state;
    ::uint8_t pnc_warninginfo;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(AVPSysMode);
        fun(system_command);
        fun(avp_run_state);
        fun(pnc_warninginfo);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(AVPSysMode);
        fun(system_command);
        fun(avp_run_state);
        fun(pnc_warninginfo);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("AVPSysMode", AVPSysMode);
        fun("system_command", system_command);
        fun("avp_run_state", avp_run_state);
        fun("pnc_warninginfo", pnc_warninginfo);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("AVPSysMode", AVPSysMode);
        fun("system_command", system_command);
        fun("avp_run_state", avp_run_state);
        fun("pnc_warninginfo", pnc_warninginfo);
    }

    bool operator==(const ::hozon::chassis::AlgMcuEgoAVPMsg& t) const
    {
        return (AVPSysMode == t.AVPSysMode) && (system_command == t.system_command) && (avp_run_state == t.avp_run_state) && (pnc_warninginfo == t.pnc_warninginfo);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGMCUEGOAVPMSG_H
