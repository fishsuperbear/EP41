#include "proxy_pnc_control.h"
#include <memory>
#include "common.h"

namespace hozon {
namespace netaos {
namespace sensor {

PncCtrProxy::PncCtrProxy() : _pnc_ctr_seqid(0u) {}

std::shared_ptr<hozon::state::StateMachine> PncCtrProxy::Trans(ara::com::SamplePtr<::hozon::netaos::PNCControlState const> data) {

    std::shared_ptr<hozon::state::StateMachine> stateMachine_proto = std::make_shared<hozon::state::StateMachine>();
    
    if (stateMachine_proto == nullptr) {
        SENSOR_LOG_ERROR << "stateMachine_proto Allocate got nullptr!";
        return nullptr;
    }
    stateMachine_proto->mutable_header()->set_frame_id("pcn_ctrl");
    stateMachine_proto->mutable_header()->set_seq(_pnc_ctr_seqid++);
    stateMachine_proto->mutable_header()->set_publish_stamp(GetRealTimestamp());
    stateMachine_proto->mutable_header()->set_gnss_stamp(GetAbslTimestamp());
    auto Pnc_proto = stateMachine_proto->mutable_pnc_control_state();
    // std::shared_ptr<hozon::state::PNCControlState> Pnc_proto = std::make_shared<hozon::state::PNCControlState>();

    // receive someip pnc struct data to pnc proto
    Pnc_proto->set_fct_state((hozon::state::PNCControlState::FctState)data->fct_state);
    Pnc_proto->set_m_iuss_state_obs(data->m_iuss_state_obs);
    Pnc_proto->set_need_replan_stop(data->need_replan_stop);
    Pnc_proto->set_plan_trigger(data->plan_trigger);
    Pnc_proto->set_control_enable(data->control_enable);
    Pnc_proto->set_control_status(data->control_status);
    Pnc_proto->set_pnc_run_state(data->pnc_run_state);
    Pnc_proto->set_pnc_warninginfo(data->pnc_warninginfo);
    Pnc_proto->set_pnc_adcs4__tex(data->pnc_ADCS4_Tex);
    Pnc_proto->set_pnc_adcs4_pa_failinfo(data->pnc_ADCS4_PA_failinfo);
    Pnc_proto->set_fapa(data->FAPA);
    Pnc_proto->set_rpa(data->RPA);
    Pnc_proto->set_tba(data->TBA);
    Pnc_proto->set_lapa_map_building(data->LAPA_MapBuilding);
    Pnc_proto->set_lapa_cruising(data->LAPA_Cruising);
    Pnc_proto->set_lapa_pick_up(data->LAPA_PickUp);
    Pnc_proto->set_ism(data->ISM);
    Pnc_proto->set_avp(data->AVP);
    Pnc_proto->set_pnc_adcs4_tba_failinfo(data->pnc_ADCS4_TBA_failinfo);
    Pnc_proto->set_pnc_adcs4_rpa_failinfo(data->pnc_ADCS4_RPA_failinfo);
    Pnc_proto->set_pnc_adcs4_lapa__map_building_failinfo(data->pnc_ADCS4_LAPA_MapBuilding_failinfo);
    Pnc_proto->set_pnc_adcs4_lapa__cruising_failinfo(data->pnc_ADCS4_LAPA_Cruising_failinfo);
    Pnc_proto->set_pnc_adcs4_lapa__pick_up_failinfo(data->pnc_ADCS4_LAPA_PickUp_failinfo);
    Pnc_proto->set_pnc_adcs4_ism_failinfo(data->pnc_ADCS4_ISM_failinfo);
    Pnc_proto->set_pnc_adcs4_avp_failinfo(data->pnc_ADCS4_AVP_failinfo);
    Pnc_proto->set_tba_text(data->TBA_text);
    Pnc_proto->set_reserved2(data->reserved3);

    return stateMachine_proto;
}

}  // namespace sensor
}  // namespace netaos
}  // namespace hozon