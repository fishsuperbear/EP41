/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_P_CURR_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_P_CURR_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "hozon/soc_mcu/impl_type_dtcloud_p_tcb.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_p_curr {
    ::UInt32 tid;
    ::UInt8 ready_prio;
    ::UInt8 dispatch_prio;
    ::hozon::soc_mcu::DtCloud_p_tcb p_tcb;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(tid);
        fun(ready_prio);
        fun(dispatch_prio);
        fun(p_tcb);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(tid);
        fun(ready_prio);
        fun(dispatch_prio);
        fun(p_tcb);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("tid", tid);
        fun("ready_prio", ready_prio);
        fun("dispatch_prio", dispatch_prio);
        fun("p_tcb", p_tcb);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("tid", tid);
        fun("ready_prio", ready_prio);
        fun("dispatch_prio", dispatch_prio);
        fun("p_tcb", p_tcb);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_p_curr& t) const
    {
        return (tid == t.tid) && (ready_prio == t.ready_prio) && (dispatch_prio == t.dispatch_prio) && (p_tcb == t.p_tcb);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_P_CURR_H
