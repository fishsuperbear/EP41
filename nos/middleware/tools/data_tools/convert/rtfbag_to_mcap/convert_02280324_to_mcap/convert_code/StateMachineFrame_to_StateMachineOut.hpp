#pragma once
#include "hozon/statemachine/impl_type_statemachineframe.h"        //mdc 数据变量
#include "proto/statemachine/state_machine.pb.h"                   //proto 数据变量

hozon::state::StateMachine StateMachineFrameToStateMachineOut(hozon::statemachine::StateMachineFrame mdc_data) {
    hozon::state::StateMachine proto_data;

    // Header
    proto_data.mutable_header()->set_seq(mdc_data.counter);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.timestamp) / 1e9);

    // AutopilotStatus pilot_status
    proto_data.mutable_pilot_status()->set_processing_status(static_cast<uint32_t>(mdc_data.pilot_status.processing_status));
    proto_data.mutable_pilot_status()->set_camera_status(static_cast<uint32_t>(mdc_data.pilot_status.camera_status));
    proto_data.mutable_pilot_status()->set_uss_status(static_cast<uint32_t>(mdc_data.pilot_status.uss_status));
    proto_data.mutable_pilot_status()->set_radar_status(static_cast<uint32_t>(mdc_data.pilot_status.radar_status));
    proto_data.mutable_pilot_status()->set_lidar_status(static_cast<uint32_t>(mdc_data.pilot_status.lidar_status));
    proto_data.mutable_pilot_status()->set_velocity_status(static_cast<uint32_t>(mdc_data.pilot_status.velocity_status));
    proto_data.mutable_pilot_status()->set_perception_status(static_cast<uint32_t>(mdc_data.pilot_status.perception_status));
    proto_data.mutable_pilot_status()->set_planning_status(static_cast<uint32_t>(mdc_data.pilot_status.planning_status));
    proto_data.mutable_pilot_status()->set_controlling_status(static_cast<uint32_t>(mdc_data.pilot_status.controlling_status));
    proto_data.mutable_pilot_status()->set_turn_light_status(static_cast<uint32_t>(mdc_data.pilot_status.turn_light_status));
    proto_data.mutable_pilot_status()->set_localization_status(static_cast<uint32_t>(mdc_data.pilot_status.localization_status));

    // Command hpp_command
    proto_data.mutable_hpp_command()->set_enable_parking_slot_detection(static_cast<uint32_t>(mdc_data.hpp_command.enable_parking_slot_detection));
    proto_data.mutable_hpp_command()->set_enable_object_detection(static_cast<uint32_t>(mdc_data.hpp_command.enable_object_detection));
    proto_data.mutable_hpp_command()->set_enable_freespace_detection(static_cast<uint32_t>(mdc_data.hpp_command.enable_freespace_detection));
    proto_data.mutable_hpp_command()->set_enable_uss(static_cast<uint32_t>(mdc_data.hpp_command.enable_uss));
    proto_data.mutable_hpp_command()->set_enable_radar(static_cast<uint32_t>(mdc_data.hpp_command.enable_radar));
    proto_data.mutable_hpp_command()->set_enable_lidar(static_cast<uint32_t>(mdc_data.hpp_command.enable_lidar));
    proto_data.mutable_hpp_command()->set_system_command(static_cast<uint32_t>(mdc_data.hpp_command.system_command));
    proto_data.mutable_hpp_command()->set_emergencybrake_state(static_cast<uint32_t>(mdc_data.hpp_command.emergencybrake_state));
    proto_data.mutable_hpp_command()->set_system_reset(static_cast<uint32_t>(mdc_data.hpp_command.system_reset));
    proto_data.mutable_hpp_command()->set_reserved1(static_cast<uint32_t>(mdc_data.hpp_command.reserved1));
    proto_data.mutable_hpp_command()->set_reserved2(static_cast<uint32_t>(mdc_data.hpp_command.reserved2));
    proto_data.mutable_hpp_command()->set_reserved3(static_cast<uint32_t>(mdc_data.hpp_command.reserved3));

    // WorkingStatus hpp_perception_status
    proto_data.mutable_hpp_perception_status()->set_processing_status(static_cast<uint32_t>(mdc_data.hpp_perception_status.processing_status));
    proto_data.mutable_hpp_perception_status()->set_error_code(static_cast<uint32_t>(mdc_data.hpp_perception_status.error_code));
    proto_data.mutable_hpp_perception_status()->set_perception_warninginfo(static_cast<uint32_t>(mdc_data.hpp_perception_status.perception_warninginfo));
    proto_data.mutable_hpp_perception_status()->set_perception_adcs4__tex(static_cast<uint32_t>(mdc_data.hpp_perception_status.perception_ADCS4_Tex));
    proto_data.mutable_hpp_perception_status()->set_perception_adcs4_pa_failinfo(static_cast<uint32_t>(mdc_data.hpp_perception_status.perception_ADCS4_PA_failinfo));
    proto_data.mutable_hpp_perception_status()->set_tba__distance(static_cast<uint32_t>(mdc_data.hpp_perception_status.TBA_Distance));
    proto_data.mutable_hpp_perception_status()->set_tba(static_cast<bool>(mdc_data.hpp_perception_status.TBA));
    proto_data.mutable_hpp_perception_status()->set_tba_text(static_cast<uint32_t>(mdc_data.hpp_perception_status.TBA_text));
    proto_data.mutable_hpp_perception_status()->set_reserved2(static_cast<uint32_t>(mdc_data.hpp_perception_status.reserved2));
    proto_data.mutable_hpp_perception_status()->set_reserved3(static_cast<uint32_t>(mdc_data.hpp_perception_status.reserved3));

    // PNCControlState pnc_control_state
    proto_data.mutable_pnc_control_state()->set_fct_state(static_cast<hozon::state::PNCControlState::FctState>(mdc_data.pnc_control_state.fct_state));
    proto_data.mutable_pnc_control_state()->set_m_iuss_state_obs(static_cast<uint32_t>(mdc_data.pnc_control_state.m_iuss_state_obs));
    proto_data.mutable_pnc_control_state()->set_need_replan_stop(static_cast<uint32_t>(mdc_data.pnc_control_state.need_replan_stop));
    proto_data.mutable_pnc_control_state()->set_plan_trigger(static_cast<uint32_t>(mdc_data.pnc_control_state.plan_trigger));
    proto_data.mutable_pnc_control_state()->set_control_enable(static_cast<uint32_t>(mdc_data.pnc_control_state.control_enable));
    proto_data.mutable_pnc_control_state()->set_control_status(static_cast<uint32_t>(mdc_data.pnc_control_state.control_status));
    proto_data.mutable_pnc_control_state()->set_pnc_run_state(static_cast<uint32_t>(mdc_data.pnc_control_state.pnc_run_state));
    proto_data.mutable_pnc_control_state()->set_pnc_warninginfo(static_cast<uint32_t>(mdc_data.pnc_control_state.pnc_warninginfo));
    proto_data.mutable_pnc_control_state()->set_pnc_adcs4__tex(static_cast<uint32_t>(mdc_data.pnc_control_state.pnc_ADCS4_Tex));
    proto_data.mutable_pnc_control_state()->set_pnc_adcs4_pa_failinfo(static_cast<uint32_t>(mdc_data.pnc_control_state.pnc_ADCS4_PA_failinfo));
    proto_data.mutable_pnc_control_state()->set_fapa(mdc_data.pnc_control_state.FAPA);
    proto_data.mutable_pnc_control_state()->set_rpa(mdc_data.pnc_control_state.RPA);
    proto_data.mutable_pnc_control_state()->set_tba(mdc_data.pnc_control_state.TBA);
    proto_data.mutable_pnc_control_state()->set_lapa_map_building(mdc_data.pnc_control_state.LAPA_MapBuilding);
    proto_data.mutable_pnc_control_state()->set_lapa_cruising(mdc_data.pnc_control_state.LAPA_Cruising);
    proto_data.mutable_pnc_control_state()->set_lapa_pick_up(mdc_data.pnc_control_state.LAPA_PickUp);
    proto_data.mutable_pnc_control_state()->set_ism(mdc_data.pnc_control_state.ISM);
    proto_data.mutable_pnc_control_state()->set_avp(mdc_data.pnc_control_state.AVP);
    proto_data.mutable_pnc_control_state()->set_pnc_adcs4_tba_failinfo(static_cast<uint32_t>(mdc_data.pnc_control_state.pnc_ADCS4_TBA_failinfo));
    proto_data.mutable_pnc_control_state()->set_pnc_adcs4_rpa_failinfo(static_cast<uint32_t>(mdc_data.pnc_control_state.pnc_ADCS4_RPA_failinfo));
    proto_data.mutable_pnc_control_state()->set_pnc_adcs4_lapa__map_building_failinfo(static_cast<uint32_t>(mdc_data.pnc_control_state.pnc_ADCS4_LAPA_MapBuilding_failinfo));
    proto_data.mutable_pnc_control_state()->set_pnc_adcs4_lapa__cruising_failinfo(static_cast<uint32_t>(mdc_data.pnc_control_state.pnc_ADCS4_LAPA_Cruising_failinfo));
    proto_data.mutable_pnc_control_state()->set_pnc_adcs4_lapa__pick_up_failinfo(static_cast<uint32_t>(mdc_data.pnc_control_state.pnc_ADCS4_LAPA_PickUp_failinfo));
    proto_data.mutable_pnc_control_state()->set_pnc_adcs4_ism_failinfo(static_cast<uint32_t>(mdc_data.pnc_control_state.pnc_ADCS4_ISM_failinfo));
    proto_data.mutable_pnc_control_state()->set_pnc_adcs4_avp_failinfo(static_cast<uint32_t>(mdc_data.pnc_control_state.pnc_ADCS4_AVP_failinfo));
    proto_data.mutable_pnc_control_state()->set_tba_text(static_cast<uint32_t>(mdc_data.pnc_control_state.TBA_text));
    proto_data.mutable_pnc_control_state()->set_reserved2(static_cast<uint32_t>(mdc_data.pnc_control_state.reserved2));
    proto_data.mutable_pnc_control_state()->set_reserved3(static_cast<uint32_t>(mdc_data.pnc_control_state.reserved3));

    return proto_data;
}