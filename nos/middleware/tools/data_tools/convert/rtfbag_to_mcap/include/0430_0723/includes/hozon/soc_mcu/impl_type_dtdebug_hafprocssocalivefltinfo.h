/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFPROCSSOCALIVEFLTINFO_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFPROCSSOCALIVEFLTINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_HafprocsSocAliveFltInfo {
    ::UInt8 ProcsAliveFlt_hz_extra_trans;
    ::UInt8 ProcsAliveFlt_stateMachine_calmcar;
    ::UInt8 ProcsAliveFlt_egoPlanning;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ProcsAliveFlt_hz_extra_trans);
        fun(ProcsAliveFlt_stateMachine_calmcar);
        fun(ProcsAliveFlt_egoPlanning);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ProcsAliveFlt_hz_extra_trans);
        fun(ProcsAliveFlt_stateMachine_calmcar);
        fun(ProcsAliveFlt_egoPlanning);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ProcsAliveFlt_hz_extra_trans", ProcsAliveFlt_hz_extra_trans);
        fun("ProcsAliveFlt_stateMachine_calmcar", ProcsAliveFlt_stateMachine_calmcar);
        fun("ProcsAliveFlt_egoPlanning", ProcsAliveFlt_egoPlanning);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ProcsAliveFlt_hz_extra_trans", ProcsAliveFlt_hz_extra_trans);
        fun("ProcsAliveFlt_stateMachine_calmcar", ProcsAliveFlt_stateMachine_calmcar);
        fun("ProcsAliveFlt_egoPlanning", ProcsAliveFlt_egoPlanning);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_HafprocsSocAliveFltInfo& t) const
    {
        return (ProcsAliveFlt_hz_extra_trans == t.ProcsAliveFlt_hz_extra_trans) && (ProcsAliveFlt_stateMachine_calmcar == t.ProcsAliveFlt_stateMachine_calmcar) && (ProcsAliveFlt_egoPlanning == t.ProcsAliveFlt_egoPlanning);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFPROCSSOCALIVEFLTINFO_H
