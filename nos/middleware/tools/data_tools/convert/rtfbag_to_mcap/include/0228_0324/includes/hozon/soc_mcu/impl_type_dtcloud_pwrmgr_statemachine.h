/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWRMGR_STATEMACHINE_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWRMGR_STATEMACHINE_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtcloud_pwrevent.h"
#include "impl_type_uint8.h"
#include "hozon/soc_mcu/impl_type_dtcloud_pwrmgrerrorstatus.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_PwrMgr_StateMachine {
    ::hozon::soc_mcu::DtCloud_PwrEvent pwrMgrEvent;
    ::UInt8 curPwrMgrState;
    ::hozon::soc_mcu::DtCloud_PwrMgrErrorstatus PwrMgrErrorState;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(pwrMgrEvent);
        fun(curPwrMgrState);
        fun(PwrMgrErrorState);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(pwrMgrEvent);
        fun(curPwrMgrState);
        fun(PwrMgrErrorState);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("pwrMgrEvent", pwrMgrEvent);
        fun("curPwrMgrState", curPwrMgrState);
        fun("PwrMgrErrorState", PwrMgrErrorState);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("pwrMgrEvent", pwrMgrEvent);
        fun("curPwrMgrState", curPwrMgrState);
        fun("PwrMgrErrorState", PwrMgrErrorState);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_PwrMgr_StateMachine& t) const
    {
        return (pwrMgrEvent == t.pwrMgrEvent) && (curPwrMgrState == t.curPwrMgrState) && (PwrMgrErrorState == t.PwrMgrErrorState);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_PWRMGR_STATEMACHINE_H
