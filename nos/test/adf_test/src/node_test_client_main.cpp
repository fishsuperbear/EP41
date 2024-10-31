#include <signal.h>
#include <unistd.h>
#include <functional>
#include <iostream>
#include "adf/include/data_types/common/types.h"
#include "adf/include/node_base.h"
#include "adf/include/log.h"
#include "adf/include/node_bundle.h"
#include "custom_signal.h"
#include "idl/generated/chassis.h"
#include "idl/generated/chassisPubSubTypes.h"
#include "idl/generated/location.h"
#include "idl/generated/locationPubSubTypes.h"
#include "idl/generated/lane.h"
#include "idl/generated/lanePubSubTypes.h"
#include "idl/generated/freespace.h"
#include "idl/generated/freespacePubSubTypes.h"
#include "idl/generated/obj_signal.h"
#include "idl/generated/obj_signalPubSubTypes.h"
#include "idl/generated/obj_fusion.h"
#include "idl/generated/obj_fusionPubSubTypes.h"
#include "idl/generated/map_msg.h"
#include "idl/generated/map_msgPubSubTypes.h"
#include "idl/generated/parkinglot.h"
#include "idl/generated/parkinglotPubSubTypes.h"
#include "idl/generated/prediction.h"
#include "idl/generated/predictionPubSubTypes.h"
#include "idl/generated/mcu2ego_info.h"
#include "idl/generated/mcu2ego_infoPubSubTypes.h"
#include "idl/generated/mcu_debug.h"
#include "idl/generated/mcu_debugPubSubTypes.h"
#include "idl/generated/sensor_uss.h"
#include "idl/generated/sensor_ussPubSubTypes.h"
#include "idl/generated/sensor_ussinfo.h"
#include "idl/generated/sensor_ussinfoPubSubTypes.h"
#include "idl/generated/perception_info.h"
#include "idl/generated/perception_infoPubSubTypes.h"
#include "idl/generated/hmi_soc_type.h"
#include "idl/generated/hmi_soc_typePubSubTypes.h"
#include "idl/generated/ego_trajectory.h"
#include "idl/generated/ego_trajectoryPubSubTypes.h"
#include "idl/generated/planning_dec.h"
#include "idl/generated/planning_decPubSubTypes.h"
#include "idl/generated/ego2mcu_info.h"
#include "idl/generated/ego2mcu_infoPubSubTypes.h"
#include "idl/generated/low_spd_bsd.h"
#include "idl/generated/low_spd_bsdPubSubTypes.h"
#include "idl/generated/planning_debug.h"
#include "idl/generated/planning_debugPubSubTypes.h"
#include "proto/fsm/function_manager.pb.h"
#include "proto/planning/warning.pb.h"
#include "proto/hmi/avp.pb.h"
#include "proto/statemachine/state_machine.pb.h"
#include "proto/test/soc/for_test.pb.h"
#include "adf/include/node_proto_register.h"
#include "proto/planning/planning.pb.h"
#include "proto/soc/apa2mcu_chassis.pb.h"
#include "proto/perception/transport_element.pb.h"
#include "proto/localization/localization.pb.h"
#include "proto/perception/perception_obstacle.pb.h"
#include "proto/statemachine/state_machine.pb.h"

using namespace hozon::netaos::adf;
using namespace hozon::netaos::log;

struct ConfigMap {
    std::string module_name;
    // std::string process_name;
    std::string config_name;
    std::function<void (NodeBundle&, NodeBundle&)> func;
};

int g_config_index = -1;
int g_parent_pid = -1;
void ChassisTest(NodeBundle& input , NodeBundle& output) {
    #ifdef NODE_DATA_TEST
    BaseDataTypePtr alg_data = input.GetOne("chassis");
    if(alg_data == nullptr) {
        NODE_LOG_ERROR << "Fail to get Chassis out.";
    }
    else {
        std::shared_ptr<AlgChassisInfo> data = std::static_pointer_cast<AlgChassisInfo>(alg_data->idl_msg);
        NODE_LOG_INFO << "Chassis receive value: " << data->header().seq();
    }
    #endif
}
void ProtoTest(NodeBundle& input, NodeBundle& output) {
    #ifdef NODE_DATA_TEST
    BaseDataTypePtr base_ptr = input.GetOne("workresult");
    if(base_ptr == nullptr) {
        NODE_LOG_ERROR << "Fail to get WorkResult data.";
    }
    #endif
    BaseDataTypePtr bast_ptr = input.GetOne("chassis");
    if (bast_ptr == nullptr) {
        NODE_LOG_ERROR << "Fail to get chassis data.";
    }
}
void SensorTransTest(NodeBundle& input, NodeBundle& output) {

}
void PlanningTest(NodeBundle& input , NodeBundle& output) {
    #ifdef NODE_DATA_TEST
    BaseDataTypePtr alg_data = input.GetOne("parking_lot");
    if(alg_data == nullptr) {
        NODE_LOG_ERROR << "Fail to get parking_lot out.";
    }
    else {
        std::shared_ptr<AlgParkingLotOutArray> data = std::static_pointer_cast<AlgParkingLotOutArray>(alg_data->idl_msg);
        NODE_LOG_INFO << "parking_lot receive value: " << data->PathPointSize();
    }
    #endif

    std::shared_ptr<AlgEgoTrajectory> ego_trajectory = std::make_shared<AlgEgoTrajectory>();
    BaseDataTypePtr alg_ego_trajectory_in_data = std::make_shared<BaseData>();
    alg_ego_trajectory_in_data->idl_msg = ego_trajectory;
    output.Add("ego_trajectory", alg_ego_trajectory_in_data);

    std::shared_ptr<AlgPlanningDecisionInfo> ego_planning_dec = std::make_shared<AlgPlanningDecisionInfo>();
    BaseDataTypePtr alg_ego_planning_dec_data = std::make_shared<BaseData>();
    alg_ego_planning_dec_data->idl_msg = ego_planning_dec;
    output.Add("ego_planning_dec", alg_ego_planning_dec_data);

    std::shared_ptr<AlgEgo2McuFrame> ego_to_mcu = std::make_shared<AlgEgo2McuFrame>();
    BaseDataTypePtr alg_ego_to_mcu_data = std::make_shared<BaseData>();
    alg_ego_to_mcu_data->idl_msg = ego_to_mcu;
    output.Add("ego_to_mcu", alg_ego_to_mcu_data);

    std::shared_ptr<AlgEgoHmiFrame> warning_info = std::make_shared<AlgEgoHmiFrame>();
    BaseDataTypePtr alg_warning_info_data = std::make_shared<BaseData>();
    alg_warning_info_data->idl_msg = warning_info;
    output.Add("warning_info", alg_warning_info_data);

    std::shared_ptr<DebugPlanningFrame> planning_debug = std::make_shared<DebugPlanningFrame>();
    BaseDataTypePtr alg_planning_debug_data = std::make_shared<BaseData>();
    alg_planning_debug_data->idl_msg = planning_debug;
    output.Add("planning_debug", alg_planning_debug_data);

    std::shared_ptr<Alglowspeedbsd> low_spd_bsd = std::make_shared<Alglowspeedbsd>();
    BaseDataTypePtr alg_low_spd_bsd_data = std::make_shared<BaseData>();
    alg_low_spd_bsd_data->idl_msg = low_spd_bsd;
    output.Add("low_spd_bsd", alg_low_spd_bsd_data);

    /* Ego2Hmi */
        auto proto_warning = std::make_shared<hozon::planning::WarningOutput>();
        BaseDataTypePtr idl_warning = std::make_shared<BaseData>();
        proto_warning->mutable_warning_status()->set_lca_left_warning(static_cast<hozon::planning::WarningLevel>(3));
        proto_warning->mutable_warning_status()->set_lca_state(3);
        proto_warning->mutable_warning_status()->set_fcta_state(3);
        proto_warning->mutable_warning_status()->set_fcta_left_warning(static_cast<hozon::planning::WarningLevel>(3));
        proto_warning->mutable_warning_status()->set_fcta_right_warning(static_cast<hozon::planning::WarningLevel>(3));
        proto_warning->mutable_warning_status()->set_fcta_object_type(static_cast<hozon::planning::FCTAObjType>(3));
        proto_warning->mutable_warning_status()->set_fcta_fault_status(static_cast<hozon::planning::FunctionFaultStatus>(3));
        proto_warning->mutable_warning_status()->set_rcta_state(3);
        proto_warning->mutable_warning_status()->set_rcta_left_warning(static_cast<hozon::planning::WarningLevel>(3));
        proto_warning->mutable_warning_status()->set_rcta_right_warning(static_cast<hozon::planning::WarningLevel>(3));
        proto_warning->mutable_warning_status()->set_rcta_fault_status(static_cast<hozon::planning::FunctionFaultStatus>(3));
        proto_warning->mutable_warning_status()->set_rcw_state(3);
        proto_warning->mutable_warning_status()->set_rcw_warning(static_cast<hozon::planning::WarningLevel>(3));
        proto_warning->mutable_warning_status()->set_rcw_fault_status(static_cast<hozon::planning::FunctionFaultStatus>(3));
        proto_warning->mutable_warning_status()->set_dow_state(3);
        proto_warning->mutable_warning_status()->set_dow_left_warning(static_cast<hozon::planning::WarningLevel>(3));
        proto_warning->mutable_warning_status()->set_dow_right_warning(static_cast<hozon::planning::WarningLevel>(3));
        proto_warning->mutable_warning_status()->set_dow_fault_status(static_cast<hozon::planning::FunctionFaultStatus>(3));
        proto_warning->mutable_warning_status()->set_lca_right_warning(static_cast<hozon::planning::WarningLevel>(3));
        proto_warning->mutable_warning_status()->set_lca_fault_status(static_cast<hozon::planning::FunctionFaultStatus>(3));
        proto_warning->mutable_warning_status()->set_dow_audio_play(1);
        proto_warning->mutable_warning_status()->set_rcta_audio_play(static_cast<hozon::planning::AudioPlay>(1));
        proto_warning->mutable_warning_status()->set_fcta_audio_play(static_cast<hozon::planning::AudioPlay>(1));
        proto_warning->mutable_warning_status()->set_rcw_audio_play(1);
        idl_warning->proto_msg = proto_warning;
        output.Add("ego2hmi", idl_warning);

        /* Ego2HmiPark */
        auto proto_park = std::make_shared<hozon::hmi::AvpToHmi>();
        BaseDataTypePtr idl_park = std::make_shared<BaseData>();
        proto_park->set_park_bar_percent(15);
        proto_park->set_guild_line_a(655.35);
        proto_park->set_guild_line_b(655.35);
        proto_park->set_guild_line_c(655.35);
        proto_park->set_guild_line_d(655.35);
        proto_park->set_guild_line_xmin(655.35);
        proto_park->set_guild_line_xmax(655.35);
        proto_park->set_hour_of_day(63.0);
        proto_park->set_minute_of_hour(63.0);
        proto_park->set_second_of_minute(63.0);
        proto_park->set_nns_distance(65535.0);
        proto_park->set_hpa_distance(4095.0);
        proto_park->set_park_time_remaining(1023.0);
        idl_park->proto_msg = proto_park;
        output.Add("ego2hmi_park", idl_park);

        /* Ego2Mcu */
        auto proto_ego2mcu = std::make_shared<hozon::functionmanager::FunctionManagerOut>();
        BaseDataTypePtr idl_ego2mcu = std::make_shared<BaseData>();
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_active_on_off_set(static_cast<hozon::functionmanager::NnpActiveOnOffSet>(2));
        proto_ego2mcu->mutable_nnp_fct_out()->set_lane_change_infor(static_cast<hozon::functionmanager::ChangeLaneInfor>(7));
        proto_ego2mcu->mutable_nnp_fct_out()->set_lane_change_direction(static_cast<hozon::functionmanager::LaneChangeDir>(2));
        proto_ego2mcu->mutable_nnp_fct_out()->set_lc_audio_play(static_cast<hozon::functionmanager::LcAudioPlay>(2));
        proto_ego2mcu->mutable_nnp_fct_out()->set_lcsndrequest(3);
        proto_ego2mcu->mutable_nnp_fct_out()->set_dclc_audio_play(static_cast<hozon::functionmanager::DclcAudioPlay>(3));
        proto_ego2mcu->mutable_nnp_fct_out()->set_lane_change_warning(static_cast<hozon::functionmanager::LaneChangeWarn>(3));
        proto_ego2mcu->mutable_nnp_fct_out()->set_light_request(static_cast<hozon::functionmanager::LightReq>(3));
        proto_ego2mcu->mutable_nnp_fct_out()->set_lane_change_pending_alert(3);
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_light_remind(1);
        proto_ego2mcu->mutable_nnp_fct_out()->set_lateralctr_takeover(1);
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_scenarios_audio_play(static_cast<hozon::functionmanager::NNPScenarios>(10));
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_scenarios(static_cast<hozon::functionmanager::NNPScenarios>(10));
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_rino_status(static_cast<hozon::functionmanager::NNPRinoStatus>(7));
        proto_ego2mcu->mutable_nnp_fct_out()->set_paymode_confirm_feedback(3);
        proto_ego2mcu->mutable_nnp_fct_out()->set_spdadapt_comfirm_feedback(3);
        proto_ego2mcu->mutable_nnp_fct_out()->set_tsr_speedsign(15);
        proto_ego2mcu->mutable_nnp_fct_out()->set_aalc_mode(1);
        proto_ego2mcu->mutable_avp_fct_out()->set_iuss_state_obs(15);
        proto_ego2mcu->mutable_avp_fct_out()->set_need_replan_stop(1);
        proto_ego2mcu->mutable_avp_fct_out()->set_plan_trigger(1);
        proto_ego2mcu->mutable_avp_fct_out()->set_control_enable(1);
        proto_ego2mcu->mutable_avp_fct_out()->set_parking_status(static_cast<hozon::functionmanager::AvpFctOut_ParkState>(12));
        proto_ego2mcu->set_soc_2_fct_tbd_u32_01(4294967295);
        proto_ego2mcu->set_soc_2_fct_tbd_u32_02(4294967295);
        proto_ego2mcu->set_soc_2_fct_tbd_u32_03(4294967295);
        proto_ego2mcu->set_soc_2_fct_tbd_u32_04(4294967295);
        proto_ego2mcu->set_soc_2_fct_tbd_u32_05(4294967295);
        // proto_ego2mcu->mutable_nnp_fct_out()->mutable_nnp_activation_conditions()->set_nnp_hd_map_location_navi(1);
        // proto_ego2mcu->mutable_nnp_fct_out()->mutable_nnp_activation_conditions()->set_nnp_indx_crrnt_lane_cond(1);
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_d_distance2_onramp_sg(1);;
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_d_distance2_downramp_sg(1);
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_d_distance_into_odd_sg(1);
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_d_distance_outof_odd_sg(1);
        proto_ego2mcu->mutable_nnp_fct_out()->mutable_nnp_activation_conditions()->set_appropriate_current_lane_width(100);
        proto_ego2mcu->mutable_nnp_fct_out()->mutable_nnp_activation_conditions()->set_appropriate_current_lane_curve(1);
        proto_ego2mcu->mutable_nnp_fct_out()->mutable_nnp_activation_conditions()->set_appropriate_current_lane_headingerr(1);
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_is_nnpmrmflf_bl(1);
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_is_nnpmrm_doneflf_bl(1);
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_is_nnpemflf_bl(1);
        proto_ego2mcu->mutable_nnp_fct_out()->set_nnp_is_nnpem_doneflf_bl(1);
        // proto_ego2mcu->mutable_nnp_fct_out()->mutable_nnp_activation_conditions()->set_nnp_software_fault(1);
        proto_ego2mcu->mutable_nnp_fct_out()->mutable_light_signal_reqst()->set_highbeamreqst(1);
        proto_ego2mcu->mutable_nnp_fct_out()->mutable_light_signal_reqst()->set_lowbeamreqst(1);
        proto_ego2mcu->mutable_nnp_fct_out()->mutable_light_signal_reqst()->set_lowhighbeamreqst(1);
        proto_ego2mcu->mutable_nnp_fct_out()->mutable_light_signal_reqst()->set_hazardlampreqst(1);
        proto_ego2mcu->mutable_nnp_fct_out()->mutable_light_signal_reqst()->set_hornreqst(1);
        idl_ego2mcu->proto_msg = proto_ego2mcu;
        output.Add("ego2mcu", idl_ego2mcu);

        /* ego2state_machine */
        auto proto_state_machine = std::make_shared<hozon::state::StateMachine>();
        BaseDataTypePtr idl_state_machine = std::make_shared<BaseData>();
        proto_state_machine->mutable_pilot_status()->set_processing_status(255);
        proto_state_machine->mutable_pilot_status()->set_camera_status(255);
        proto_state_machine->mutable_pilot_status()->set_uss_status(255);
        proto_state_machine->mutable_pilot_status()->set_radar_status(255);
        proto_state_machine->mutable_pilot_status()->set_lidar_status(255);
        proto_state_machine->mutable_pilot_status()->set_velocity_status(255);
        proto_state_machine->mutable_pilot_status()->set_perception_status(255);
        proto_state_machine->mutable_pilot_status()->set_planning_status(255);
        proto_state_machine->mutable_pilot_status()->set_controlling_status(255);
        proto_state_machine->mutable_pilot_status()->set_turn_light_status(255);
        proto_state_machine->mutable_pilot_status()->set_localization_status(255);
        proto_state_machine->mutable_hpp_command()->set_enable_parking_slot_detection(255);
        proto_state_machine->mutable_hpp_command()->set_enable_object_detection(255);
        proto_state_machine->mutable_hpp_command()->set_enable_freespace_detection(255);
        proto_state_machine->mutable_hpp_command()->set_enable_uss(255);
        proto_state_machine->mutable_hpp_command()->set_enable_radar(255);
        proto_state_machine->mutable_hpp_command()->set_enable_lidar(255);
        proto_state_machine->mutable_hpp_command()->set_system_command(255);
        proto_state_machine->mutable_hpp_command()->set_system_reset(255);
        proto_state_machine->mutable_hpp_command()->set_emergencybrake_state(255);
        idl_state_machine->proto_msg = proto_state_machine;
        output.Add("ego2state_machine", idl_state_machine);
}

ConfigMap g_config_map[] = {
    {"chassis",          "chassis_config",                ChassisTest},
    {"planning",         "planning_config",               PlanningTest},
    {"proto",            "proto_test_conf",               ProtoTest},
    {"sensor_trans",     "sensor_trans_config",           SensorTransTest},
};

class RecvNode : public NodeBase {
public:
    RecvNode() {
        REGISTER_CM_TYPE_CLASS("chassis", AlgChassisInfoPubSubType);
        REGISTER_CM_TYPE_CLASS("chassis_ego_hmi", AlgEgoHmiFramePubSubType);

        REGISTER_CM_TYPE_CLASS("nnp_localization", AlgLocationPubSubType);
        REGISTER_CM_TYPE_CLASS("hpp_localization", AlgLocationPubSubType);
        REGISTER_CM_TYPE_CLASS("nnp_fusion_lane", AlgLaneDetectionOutArrayPubSubType);
        REGISTER_CM_TYPE_CLASS("hpp_lane", AlgLaneDetectionOutArrayPubSubType);
        REGISTER_CM_TYPE_CLASS("nnp_fusion_freespace", AlgFreeSpaceOutArrayPubSubType);
        REGISTER_CM_TYPE_CLASS("hpp_freespace", AlgFreeSpaceOutArrayPubSubType);
        REGISTER_CM_TYPE_CLASS("nnp_cam_obj_signal", AlgObjSignalArrayPubSubType);
        REGISTER_CM_TYPE_CLASS("hpp_obj_signal", AlgObjSignalArrayPubSubType);

        REGISTER_CM_TYPE_CLASS("nnp_obj_fusion", AlgFusionOutArrayPubSubType);
        REGISTER_CM_TYPE_CLASS("hpp_obj_fusion", AlgFusionOutArrayPubSubType);
        REGISTER_CM_TYPE_CLASS("map_msg", AlgMapMessagePubSubType);

        REGISTER_CM_TYPE_CLASS("parking_lot", AlgParkingLotOutArrayPubSubType);
        REGISTER_CM_TYPE_CLASS("prediction", AlgPredictionObstacleArrayPubSubType);
        REGISTER_CM_TYPE_CLASS("mcu_to_ego", AlgAebToEgoFramePubSubType);

        REGISTER_CM_TYPE_CLASS("mcu_debug", AlgMcuDebugFramePubSubType);
        REGISTER_CM_TYPE_CLASS("aeb_to_ego", AlgAebToEgoFramePubSubType);
        REGISTER_CM_TYPE_CLASS("uss", AlgUssRawDataSetPubSubType);
        REGISTER_CM_TYPE_CLASS("uss_info", AlgUssInfoPubSubType);

        REGISTER_CM_TYPE_CLASS("perception_info", AlgPerceptionInfoFramePubSubType);
        REGISTER_CM_TYPE_CLASS("hmi_to_location_hpp_per_in", AlgHmiAvpLocFramePubSubType);
        REGISTER_CM_TYPE_CLASS("hmi_to_router_hpp_per_in", AlgNNSRouteInfoPubSubType);
        REGISTER_CM_TYPE_CLASS("mcu_adas_record", ADAS_DataRecordFramePubSubType);

        REGISTER_CM_TYPE_CLASS("ego_trajectory", AlgEgoTrajectoryPubSubType);
        REGISTER_CM_TYPE_CLASS("ego_planning_dec", AlgPlanningDecisionInfoPubSubType);
        REGISTER_CM_TYPE_CLASS("ego_to_mcu", AlgEgo2McuFramePubSubType);
        REGISTER_CM_TYPE_CLASS("warning_info", AlgEgoHmiFramePubSubType);
        REGISTER_CM_TYPE_CLASS("planning_debug", DebugPlanningFramePubSubType);
        REGISTER_CM_TYPE_CLASS("low_spd_bsd", AlglowspeedbsdPubSubType);
        REGISTER_PROTO_MESSAGE_TYPE("workresult", adf::lite::dbg::WorkflowResult)
        REGISTER_PROTO_MESSAGE_TYPE("apa2chassis", hozon::soc::Apa2Chassis)
        REGISTER_PROTO_MESSAGE_TYPE("ego2chassis", hozon::planning::ADCTrajectory)
        REGISTER_PROTO_MESSAGE_TYPE("nnplane", hozon::perception::TransportElement)
        REGISTER_PROTO_MESSAGE_TYPE("hpplane", hozon::perception::TransportElement)
        REGISTER_PROTO_MESSAGE_TYPE("nnplocation", hozon::localization::Localization)
        REGISTER_PROTO_MESSAGE_TYPE("hpplocation", hozon::localization::Localization)
        REGISTER_PROTO_MESSAGE_TYPE("nnpobject", hozon::perception::PerceptionObstacles)
        REGISTER_PROTO_MESSAGE_TYPE("hppobject", hozon::perception::PerceptionObstacles)
        REGISTER_PROTO_MESSAGE_TYPE("sm2mcu", hozon::state::StateMachine)
        REGISTER_PROTO_MESSAGE_TYPE("parkinglot2hmi_2", hozon::soc::Apa2Chassis)
    }

    virtual int32_t AlgInit() {
        // DefaultLogger::GetInstance().InitLogger();
        return 0;
    }

    virtual int32_t AlgProcess1(
            NodeBundle* input,
            const ProfileToken& token) {
        NODE_LOG_INFO << "======================================================================";
        std::unordered_map<std::string, std::vector<BaseDataTypePtr>>& raw = input->GetRaw();
        for (auto it = raw.begin(); it != raw.end(); ++it) {
            NODE_LOG_INFO << "-----------RECV: " << it->first;
        }
        
        if (g_parent_pid != -1) {
            bool all_received = true;
            auto srcs = GetConfig().trigger[0].aux_sources;
            for (auto it = srcs.begin(); it != srcs.end(); ++it) {
                if (raw.find(it->name) == raw.end()) {
                    all_received = false;
                }
            }
            if (all_received) {
                kill(g_parent_pid, CLIENT_SUCC);
                NODE_LOG_INFO << "Report all received";
            }
        }
        
        NodeBundle output;
        g_config_map[g_config_index].func(*input, output);
        SendOutput(&output, token);

        return 0;
    }

    virtual void AlgRelease() {}
};

int main(int argc, char* argv[]) {
    RecvNode sm;
    std::cout << "start node test recv module " << std::string(argv[1])  << std::endl; 
    sm.RegistAlgProcessWithProfilerFunc("main", std::bind(&RecvNode::AlgProcess1, &sm, std::placeholders::_1, std::placeholders::_2));

    for (long unsigned int i = 0; i < (sizeof(g_config_map) / sizeof(g_config_map[0])); ++i) {
        if (g_config_map[i].module_name == std::string(argv[1])) {
            g_config_index = i;
            break;
        }
    }

    if (g_config_index == -1) {
        std::cout << "Fail to find module " << std::string(argv[1]) << std::endl;
        return -1;
    }

    std::string binary_path = std::string(argv[0]);
    int pos = 0;
    for (int i = 0; i < static_cast<int>(binary_path.size()); ++i) {
        if (binary_path[i] == '/') {
            pos = i;
        }
    }
    std::string folder_path = binary_path.substr(0, pos);

    // std::string cm_config_path = folder_path + "/../etc/" + g_config_map[g_config_index].process_name;
    // if (setenv("CM_CONFIG_FILE_PATH", cm_config_path.c_str(), 1) < 0) {
    //     std::cout << "Fail to set env " << cm_config_path << std::endl;
    //     return -1;
    // }
    // std::cout << "Succ to set env " << cm_config_path << std::endl;

    if (argc >= 3) {
        g_parent_pid = atoi(argv[2]);
    }
    std::string cm_conf_path = folder_path + std::string("/../conf/") + g_config_map[g_config_index].config_name + ".yaml";
    std::cout << "Succ to set yaml: " << cm_conf_path << std::endl;
    sm.Start(cm_conf_path);
    // NODE_LOG_INFO << "pause trigger main in 20s";
    // sm.PauseTrigger("main");
    // sleep(20);
    // NODE_LOG_INFO << "resume trigger main";
    // sm.ResumeTrigger("main");

    sm.NeedStopBlocking();
    sm.Stop();
    return 0;
}
