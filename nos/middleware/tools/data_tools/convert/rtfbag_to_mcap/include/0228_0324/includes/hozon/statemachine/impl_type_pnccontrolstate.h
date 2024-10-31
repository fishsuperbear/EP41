/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STATEMACHINE_IMPL_TYPE_PNCCONTROLSTATE_H
#define HOZON_STATEMACHINE_IMPL_TYPE_PNCCONTROLSTATE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_boolean.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace statemachine {
struct PNCControlState {
    ::UInt8 fct_state;
    ::UInt8 m_iuss_state_obs;
    ::UInt8 need_replan_stop;
    ::UInt8 plan_trigger;
    ::UInt8 control_enable;
    ::UInt8 control_status;
    ::UInt8 pnc_run_state;
    ::UInt8 pnc_warninginfo;
    ::UInt8 pnc_ADCS4_Tex;
    ::UInt8 pnc_ADCS4_PA_failinfo;
    ::Boolean FAPA;
    ::Boolean RPA;
    ::Boolean TBA;
    ::Boolean LAPA_MapBuilding;
    ::Boolean LAPA_Cruising;
    ::Boolean LAPA_PickUp;
    ::Boolean ISM;
    ::Boolean AVP;
    ::UInt8 pnc_ADCS4_TBA_failinfo;
    ::UInt8 pnc_ADCS4_RPA_failinfo;
    ::UInt8 pnc_ADCS4_LAPA_MapBuilding_failinfo;
    ::UInt8 pnc_ADCS4_LAPA_Cruising_failinfo;
    ::UInt8 pnc_ADCS4_LAPA_PickUp_failinfo;
    ::UInt8 pnc_ADCS4_ISM_failinfo;
    ::UInt8 pnc_ADCS4_AVP_failinfo;
    ::UInt8 TBA_text;
    ::uint8_t reserved2;
    ::uint8_t reserved3;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(fct_state);
        fun(m_iuss_state_obs);
        fun(need_replan_stop);
        fun(plan_trigger);
        fun(control_enable);
        fun(control_status);
        fun(pnc_run_state);
        fun(pnc_warninginfo);
        fun(pnc_ADCS4_Tex);
        fun(pnc_ADCS4_PA_failinfo);
        fun(FAPA);
        fun(RPA);
        fun(TBA);
        fun(LAPA_MapBuilding);
        fun(LAPA_Cruising);
        fun(LAPA_PickUp);
        fun(ISM);
        fun(AVP);
        fun(pnc_ADCS4_TBA_failinfo);
        fun(pnc_ADCS4_RPA_failinfo);
        fun(pnc_ADCS4_LAPA_MapBuilding_failinfo);
        fun(pnc_ADCS4_LAPA_Cruising_failinfo);
        fun(pnc_ADCS4_LAPA_PickUp_failinfo);
        fun(pnc_ADCS4_ISM_failinfo);
        fun(pnc_ADCS4_AVP_failinfo);
        fun(TBA_text);
        fun(reserved2);
        fun(reserved3);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(fct_state);
        fun(m_iuss_state_obs);
        fun(need_replan_stop);
        fun(plan_trigger);
        fun(control_enable);
        fun(control_status);
        fun(pnc_run_state);
        fun(pnc_warninginfo);
        fun(pnc_ADCS4_Tex);
        fun(pnc_ADCS4_PA_failinfo);
        fun(FAPA);
        fun(RPA);
        fun(TBA);
        fun(LAPA_MapBuilding);
        fun(LAPA_Cruising);
        fun(LAPA_PickUp);
        fun(ISM);
        fun(AVP);
        fun(pnc_ADCS4_TBA_failinfo);
        fun(pnc_ADCS4_RPA_failinfo);
        fun(pnc_ADCS4_LAPA_MapBuilding_failinfo);
        fun(pnc_ADCS4_LAPA_Cruising_failinfo);
        fun(pnc_ADCS4_LAPA_PickUp_failinfo);
        fun(pnc_ADCS4_ISM_failinfo);
        fun(pnc_ADCS4_AVP_failinfo);
        fun(TBA_text);
        fun(reserved2);
        fun(reserved3);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("fct_state", fct_state);
        fun("m_iuss_state_obs", m_iuss_state_obs);
        fun("need_replan_stop", need_replan_stop);
        fun("plan_trigger", plan_trigger);
        fun("control_enable", control_enable);
        fun("control_status", control_status);
        fun("pnc_run_state", pnc_run_state);
        fun("pnc_warninginfo", pnc_warninginfo);
        fun("pnc_ADCS4_Tex", pnc_ADCS4_Tex);
        fun("pnc_ADCS4_PA_failinfo", pnc_ADCS4_PA_failinfo);
        fun("FAPA", FAPA);
        fun("RPA", RPA);
        fun("TBA", TBA);
        fun("LAPA_MapBuilding", LAPA_MapBuilding);
        fun("LAPA_Cruising", LAPA_Cruising);
        fun("LAPA_PickUp", LAPA_PickUp);
        fun("ISM", ISM);
        fun("AVP", AVP);
        fun("pnc_ADCS4_TBA_failinfo", pnc_ADCS4_TBA_failinfo);
        fun("pnc_ADCS4_RPA_failinfo", pnc_ADCS4_RPA_failinfo);
        fun("pnc_ADCS4_LAPA_MapBuilding_failinfo", pnc_ADCS4_LAPA_MapBuilding_failinfo);
        fun("pnc_ADCS4_LAPA_Cruising_failinfo", pnc_ADCS4_LAPA_Cruising_failinfo);
        fun("pnc_ADCS4_LAPA_PickUp_failinfo", pnc_ADCS4_LAPA_PickUp_failinfo);
        fun("pnc_ADCS4_ISM_failinfo", pnc_ADCS4_ISM_failinfo);
        fun("pnc_ADCS4_AVP_failinfo", pnc_ADCS4_AVP_failinfo);
        fun("TBA_text", TBA_text);
        fun("reserved2", reserved2);
        fun("reserved3", reserved3);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("fct_state", fct_state);
        fun("m_iuss_state_obs", m_iuss_state_obs);
        fun("need_replan_stop", need_replan_stop);
        fun("plan_trigger", plan_trigger);
        fun("control_enable", control_enable);
        fun("control_status", control_status);
        fun("pnc_run_state", pnc_run_state);
        fun("pnc_warninginfo", pnc_warninginfo);
        fun("pnc_ADCS4_Tex", pnc_ADCS4_Tex);
        fun("pnc_ADCS4_PA_failinfo", pnc_ADCS4_PA_failinfo);
        fun("FAPA", FAPA);
        fun("RPA", RPA);
        fun("TBA", TBA);
        fun("LAPA_MapBuilding", LAPA_MapBuilding);
        fun("LAPA_Cruising", LAPA_Cruising);
        fun("LAPA_PickUp", LAPA_PickUp);
        fun("ISM", ISM);
        fun("AVP", AVP);
        fun("pnc_ADCS4_TBA_failinfo", pnc_ADCS4_TBA_failinfo);
        fun("pnc_ADCS4_RPA_failinfo", pnc_ADCS4_RPA_failinfo);
        fun("pnc_ADCS4_LAPA_MapBuilding_failinfo", pnc_ADCS4_LAPA_MapBuilding_failinfo);
        fun("pnc_ADCS4_LAPA_Cruising_failinfo", pnc_ADCS4_LAPA_Cruising_failinfo);
        fun("pnc_ADCS4_LAPA_PickUp_failinfo", pnc_ADCS4_LAPA_PickUp_failinfo);
        fun("pnc_ADCS4_ISM_failinfo", pnc_ADCS4_ISM_failinfo);
        fun("pnc_ADCS4_AVP_failinfo", pnc_ADCS4_AVP_failinfo);
        fun("TBA_text", TBA_text);
        fun("reserved2", reserved2);
        fun("reserved3", reserved3);
    }

    bool operator==(const ::hozon::statemachine::PNCControlState& t) const
    {
        return (fct_state == t.fct_state) && (m_iuss_state_obs == t.m_iuss_state_obs) && (need_replan_stop == t.need_replan_stop) && (plan_trigger == t.plan_trigger) && (control_enable == t.control_enable) && (control_status == t.control_status) && (pnc_run_state == t.pnc_run_state) && (pnc_warninginfo == t.pnc_warninginfo) && (pnc_ADCS4_Tex == t.pnc_ADCS4_Tex) && (pnc_ADCS4_PA_failinfo == t.pnc_ADCS4_PA_failinfo) && (FAPA == t.FAPA) && (RPA == t.RPA) && (TBA == t.TBA) && (LAPA_MapBuilding == t.LAPA_MapBuilding) && (LAPA_Cruising == t.LAPA_Cruising) && (LAPA_PickUp == t.LAPA_PickUp) && (ISM == t.ISM) && (AVP == t.AVP) && (pnc_ADCS4_TBA_failinfo == t.pnc_ADCS4_TBA_failinfo) && (pnc_ADCS4_RPA_failinfo == t.pnc_ADCS4_RPA_failinfo) && (pnc_ADCS4_LAPA_MapBuilding_failinfo == t.pnc_ADCS4_LAPA_MapBuilding_failinfo) && (pnc_ADCS4_LAPA_Cruising_failinfo == t.pnc_ADCS4_LAPA_Cruising_failinfo) && (pnc_ADCS4_LAPA_PickUp_failinfo == t.pnc_ADCS4_LAPA_PickUp_failinfo) && (pnc_ADCS4_ISM_failinfo == t.pnc_ADCS4_ISM_failinfo) && (pnc_ADCS4_AVP_failinfo == t.pnc_ADCS4_AVP_failinfo) && (TBA_text == t.TBA_text) && (reserved2 == t.reserved2) && (reserved3 == t.reserved3);
    }
};
} // namespace statemachine
} // namespace hozon


#endif // HOZON_STATEMACHINE_IMPL_TYPE_PNCCONTROLSTATE_H
