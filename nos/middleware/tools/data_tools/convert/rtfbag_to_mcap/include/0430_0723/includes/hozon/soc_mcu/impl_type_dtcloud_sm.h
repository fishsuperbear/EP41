/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_SM_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_SM_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtcloud_pwrmgr_statemachine.h"
#include "hozon/soc_mcu/impl_type_dtcloud_modechange_overtiming.h"
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"
#include "impl_type_uint32.h"
#include "hozon/soc_mcu/impl_type_dtcloud_pwrmgrcallfmfail.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_SM {
    ::hozon::soc_mcu::DtCloud_PwrMgr_StateMachine PwrMgr_StateMachine;
    ::hozon::soc_mcu::DtCloud_ModeChange_OverTiming PwrMgr_ModeChange_OverTiming;
    ::UInt8 PwrMgr_DelayCoolSended;
    ::UInt8 PwrMgr_AccDisable;
    ::UInt16 SOC_NMIndication_Cnt;
    ::UInt8 pwrmgr_upgrademode;
    ::UInt8 MCU28ServiceFlag;
    ::UInt8 HwSm_SetMDCPwrStatus_Debug;
    ::UInt32 pwrmgr_request_cycle;
    ::hozon::soc_mcu::DtCloud_PwrMgrCallFMFail PwrMgr_CallFm_failMsg;
    ::UInt8 TimingFlag;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(PwrMgr_StateMachine);
        fun(PwrMgr_ModeChange_OverTiming);
        fun(PwrMgr_DelayCoolSended);
        fun(PwrMgr_AccDisable);
        fun(SOC_NMIndication_Cnt);
        fun(pwrmgr_upgrademode);
        fun(MCU28ServiceFlag);
        fun(HwSm_SetMDCPwrStatus_Debug);
        fun(pwrmgr_request_cycle);
        fun(PwrMgr_CallFm_failMsg);
        fun(TimingFlag);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(PwrMgr_StateMachine);
        fun(PwrMgr_ModeChange_OverTiming);
        fun(PwrMgr_DelayCoolSended);
        fun(PwrMgr_AccDisable);
        fun(SOC_NMIndication_Cnt);
        fun(pwrmgr_upgrademode);
        fun(MCU28ServiceFlag);
        fun(HwSm_SetMDCPwrStatus_Debug);
        fun(pwrmgr_request_cycle);
        fun(PwrMgr_CallFm_failMsg);
        fun(TimingFlag);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("PwrMgr_StateMachine", PwrMgr_StateMachine);
        fun("PwrMgr_ModeChange_OverTiming", PwrMgr_ModeChange_OverTiming);
        fun("PwrMgr_DelayCoolSended", PwrMgr_DelayCoolSended);
        fun("PwrMgr_AccDisable", PwrMgr_AccDisable);
        fun("SOC_NMIndication_Cnt", SOC_NMIndication_Cnt);
        fun("pwrmgr_upgrademode", pwrmgr_upgrademode);
        fun("MCU28ServiceFlag", MCU28ServiceFlag);
        fun("HwSm_SetMDCPwrStatus_Debug", HwSm_SetMDCPwrStatus_Debug);
        fun("pwrmgr_request_cycle", pwrmgr_request_cycle);
        fun("PwrMgr_CallFm_failMsg", PwrMgr_CallFm_failMsg);
        fun("TimingFlag", TimingFlag);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("PwrMgr_StateMachine", PwrMgr_StateMachine);
        fun("PwrMgr_ModeChange_OverTiming", PwrMgr_ModeChange_OverTiming);
        fun("PwrMgr_DelayCoolSended", PwrMgr_DelayCoolSended);
        fun("PwrMgr_AccDisable", PwrMgr_AccDisable);
        fun("SOC_NMIndication_Cnt", SOC_NMIndication_Cnt);
        fun("pwrmgr_upgrademode", pwrmgr_upgrademode);
        fun("MCU28ServiceFlag", MCU28ServiceFlag);
        fun("HwSm_SetMDCPwrStatus_Debug", HwSm_SetMDCPwrStatus_Debug);
        fun("pwrmgr_request_cycle", pwrmgr_request_cycle);
        fun("PwrMgr_CallFm_failMsg", PwrMgr_CallFm_failMsg);
        fun("TimingFlag", TimingFlag);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_SM& t) const
    {
        return (PwrMgr_StateMachine == t.PwrMgr_StateMachine) && (PwrMgr_ModeChange_OverTiming == t.PwrMgr_ModeChange_OverTiming) && (PwrMgr_DelayCoolSended == t.PwrMgr_DelayCoolSended) && (PwrMgr_AccDisable == t.PwrMgr_AccDisable) && (SOC_NMIndication_Cnt == t.SOC_NMIndication_Cnt) && (pwrmgr_upgrademode == t.pwrmgr_upgrademode) && (MCU28ServiceFlag == t.MCU28ServiceFlag) && (HwSm_SetMDCPwrStatus_Debug == t.HwSm_SetMDCPwrStatus_Debug) && (pwrmgr_request_cycle == t.pwrmgr_request_cycle) && (PwrMgr_CallFm_failMsg == t.PwrMgr_CallFm_failMsg) && (TimingFlag == t.TimingFlag);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_SM_H
