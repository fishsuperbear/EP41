/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_ADAS_DATARECORDFRAME_H
#define HOZON_SOC2MCU_IMPL_TYPE_ADAS_DATARECORDFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/soc2mcu/impl_type_fcteeprominfo.h"
#include "hozon/soc2mcu/impl_type_adptr_out.h"
#include "hozon/soc2mcu/impl_type_ctrleeprominfo.h"
#include "hozon/soc2mcu/impl_type_hafchassis.h"
#include "hozon/soc2mcu/impl_type_hafglobaltime.h"
#include "hozon/soc2mcu/impl_type_inputs_mcu.h"
#include "hozon/soc2mcu/impl_type_veh_can_inputs_ep40.h"

namespace hozon {
namespace soc2mcu {
struct ADAS_DataRecordFrame {
    ::hozon::soc2mcu::FctEEPromInfo pFCT_FctEEPromInfo;
    ::hozon::soc2mcu::Adptr_Out pAdptrOut_Adptr_Out;
    ::hozon::soc2mcu::CtrlEEPromInfo pCtrl_CtrlEEPromInfo;
    ::hozon::soc2mcu::HafChassis rAdptrOut_HafChassisInfo;
    ::hozon::soc2mcu::HafGlobalTime rCtrl_GlobalTime;
    ::hozon::soc2mcu::Inputs_MCU rAdptrIn_MCU;
    ::hozon::soc2mcu::VEH_CAN_Inputs_EP40 rAdptrIn_VEH_CAN;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(pFCT_FctEEPromInfo);
        fun(pAdptrOut_Adptr_Out);
        fun(pCtrl_CtrlEEPromInfo);
        fun(rAdptrOut_HafChassisInfo);
        fun(rCtrl_GlobalTime);
        fun(rAdptrIn_MCU);
        fun(rAdptrIn_VEH_CAN);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(pFCT_FctEEPromInfo);
        fun(pAdptrOut_Adptr_Out);
        fun(pCtrl_CtrlEEPromInfo);
        fun(rAdptrOut_HafChassisInfo);
        fun(rCtrl_GlobalTime);
        fun(rAdptrIn_MCU);
        fun(rAdptrIn_VEH_CAN);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("pFCT_FctEEPromInfo", pFCT_FctEEPromInfo);
        fun("pAdptrOut_Adptr_Out", pAdptrOut_Adptr_Out);
        fun("pCtrl_CtrlEEPromInfo", pCtrl_CtrlEEPromInfo);
        fun("rAdptrOut_HafChassisInfo", rAdptrOut_HafChassisInfo);
        fun("rCtrl_GlobalTime", rCtrl_GlobalTime);
        fun("rAdptrIn_MCU", rAdptrIn_MCU);
        fun("rAdptrIn_VEH_CAN", rAdptrIn_VEH_CAN);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("pFCT_FctEEPromInfo", pFCT_FctEEPromInfo);
        fun("pAdptrOut_Adptr_Out", pAdptrOut_Adptr_Out);
        fun("pCtrl_CtrlEEPromInfo", pCtrl_CtrlEEPromInfo);
        fun("rAdptrOut_HafChassisInfo", rAdptrOut_HafChassisInfo);
        fun("rCtrl_GlobalTime", rCtrl_GlobalTime);
        fun("rAdptrIn_MCU", rAdptrIn_MCU);
        fun("rAdptrIn_VEH_CAN", rAdptrIn_VEH_CAN);
    }

    bool operator==(const ::hozon::soc2mcu::ADAS_DataRecordFrame& t) const
    {
        return (pFCT_FctEEPromInfo == t.pFCT_FctEEPromInfo) && (pAdptrOut_Adptr_Out == t.pAdptrOut_Adptr_Out) && (pCtrl_CtrlEEPromInfo == t.pCtrl_CtrlEEPromInfo) && (rAdptrOut_HafChassisInfo == t.rAdptrOut_HafChassisInfo) && (rCtrl_GlobalTime == t.rCtrl_GlobalTime) && (rAdptrIn_MCU == t.rAdptrIn_MCU) && (rAdptrIn_VEH_CAN == t.rAdptrIn_VEH_CAN);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_ADAS_DATARECORDFRAME_H
