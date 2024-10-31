#pragma once

#include <algorithm>
#include <array>
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <memory>
#include "hozon/netaos/impl_type_algegohmiframe.h"
#include "logger.h"
#include "proto/planning/planning.pb.h"
#include "proto/planning/warning.pb.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "adf/include/node_base.h"
#include "common.h"
#include "hozon/netaos/impl_type_algegotomcuframe.h"
#include "hozon/netaos/impl_type_hafegotrajectory.h"

namespace hozon {
namespace netaos {
namespace sensor {
class SkeletonEgo2McuChassis {
public:
    SkeletonEgo2McuChassis() = default;
    ~SkeletonEgo2McuChassis() = default;
    std::shared_ptr<hozon::planning::ADCTrajectory> Trans2Proto(adf::NodeBundle data) {

        // std::shared_ptr<hozon::planning::ADCTrajectory> data_proto = std::make_shared<hozon::planning::ADCTrajectory>();
        // std::shared_ptr<CmProtoBuf> origin_data = std::static_pointer_cast<CmProtoBuf>(data);
        // data_proto->ParseFromArray(origin_data->str().data(), origin_data->str().size());
        adf::BaseDataTypePtr idl_data = data.GetOne("ego2chassis");
        if(idl_data == nullptr) {
            SENSOR_LOG_WARN << "Fail to get ego data.";
            return nullptr;
        }
        
        std::shared_ptr<hozon::planning::ADCTrajectory> data_proto = 
            std::static_pointer_cast<hozon::planning::ADCTrajectory>(idl_data->proto_msg);
        return data_proto;
    }
    int TransEgo2McuChassis(std::shared_ptr<hozon::planning::ADCTrajectory> data_proto, 
            hozon::netaos::AlgEgoHmiFrame &send_data) {
        // trans data from proto to someip
        send_data.header.seq = data_proto->mutable_header()->seq();
        std::string frame_id = data_proto->mutable_header()->frame_id();
        if(frame_id.size() < 20) {
            std::copy(frame_id.begin(), frame_id.end(), send_data.header.frameId.data());
        }

        send_data.header.stamp.sec = StampConverS(data_proto->mutable_header()->publish_stamp());
        send_data.header.stamp.nsec = StampConverNs(data_proto->mutable_header()->publish_stamp());
        send_data.header.gnssStamp.sec = StampConverS(data_proto->mutable_header()->gnss_stamp());
        send_data.header.gnssStamp.nsec = StampConverNs(data_proto->mutable_header()->gnss_stamp());
        send_data.isValid = true;

        auto pb_warning_status = data_proto->mutable_warning_status();

        send_data.warnning_info.ADCS8_LCARightWarnSt = static_cast<uint8_t>(pb_warning_status->lca_right_warning());
        send_data.warnning_info.ADCS8_LCALeftWarnSt = static_cast<uint8_t>(pb_warning_status->lca_left_warning());
        send_data.warnning_info.ADCS8_LCASystemFaultStatus = static_cast<uint8_t>(pb_warning_status->lca_fault_status());
        send_data.warnning_info.ADCS8_LCAState = static_cast<uint8_t>(pb_warning_status->lca_state());

        send_data.warnning_info.ADCS8_DOW_State = static_cast<uint8_t>(pb_warning_status->dow_state());
        send_data.warnning_info.ADCS12_DOWWarnAudioplay = static_cast<uint8_t>(pb_warning_status->dow_audio_play());
        send_data.warnning_info.ADCS8_DOWLeftWarnState = static_cast<uint8_t>(pb_warning_status->dow_left_warning());
        send_data.warnning_info.ADCS8_DOWRightWarnState = static_cast<uint8_t>(pb_warning_status->dow_right_warning());
        send_data.warnning_info.ADCS8_DOW_SystemFaultStatus = static_cast<uint8_t>(pb_warning_status->dow_fault_status());

        send_data.warnning_info.ADCS8_RCTA_State = static_cast<uint8_t>(pb_warning_status->rcta_state());
        send_data.warnning_info.ADCS12_RCTAWarnAudioplay = static_cast<uint8_t>(pb_warning_status->rcta_audio_play());
        send_data.warnning_info.ADCS8_RCTAWarnObjType = static_cast<uint8_t>(pb_warning_status->rcta_object_type());
        send_data.warnning_info.ADCS8_RCTALeftWarnSt = static_cast<uint8_t>(pb_warning_status->rcta_left_warning());
        send_data.warnning_info.ADCS8_RCTARightWarnSt = static_cast<uint8_t>(pb_warning_status->rcta_right_warning());
        send_data.warnning_info.ADCS8_RCTA_SystemFaultStatus = static_cast<uint8_t>(pb_warning_status->rcta_fault_status());

        send_data.warnning_info.ADCS8_FCTA_State = static_cast<uint8_t>(pb_warning_status->fcta_state());
        send_data.warnning_info.ADCS12_FCTAWarnAudioplay = static_cast<uint8_t>(pb_warning_status->fcta_audio_play());
        send_data.warnning_info.ADCS8_FCTAWarnObjType = static_cast<uint8_t>(pb_warning_status->fcta_object_type());
        send_data.warnning_info.ADCS8_FCTALeftActiveSt = static_cast<uint8_t>(pb_warning_status->fcta_left_warning());
        send_data.warnning_info.ADCS8_FCTARightActiveSt = static_cast<uint8_t>(pb_warning_status->fcta_right_warning());
        send_data.warnning_info.ADCS8_FCTA_SystemFaultStatus = static_cast<uint8_t>(pb_warning_status->fcta_fault_status());

        send_data.warnning_info.ADCS8_RCW_State = static_cast<uint8_t>(pb_warning_status->rcw_state());
        send_data.warnning_info.ADCS12_RCWWarnAudioplay = static_cast<uint8_t>(pb_warning_status->rcw_audio_play());
        send_data.warnning_info.ADCS8_RCW_WarnState = static_cast<uint8_t>(pb_warning_status->rcw_warning());
        send_data.warnning_info.ADCS8_RCW_SystemFaultStatus = static_cast<uint8_t>(pb_warning_status->rcw_fault_status());
        // send_data.warnning_info.ADCS8_VoiceMode;

        auto pb_avp_to_hmi = data_proto->mutable_avp_to_hmi();
        send_data.park_hmi_info.ADCS4_PA_ParkBarPercent = pb_avp_to_hmi->park_bar_percent();

        // PA_GuideLineE value range is [0-655.35] in can dbc, temp add offset to
        // solve this issue. add factor for change 0.01 to 0.001
        // static constexpr double kOffset = 327.68;
        // static constexpr double kFactor = 10.0;
        send_data.park_hmi_info.ADCS4_PA_GuideLineE_a = 
            pb_avp_to_hmi->guild_line_a();
        send_data.park_hmi_info.ADCS4_PA_GuideLineE_b =
            pb_avp_to_hmi->guild_line_b();
        send_data.park_hmi_info.ADCS4_PA_GuideLineE_c =
            pb_avp_to_hmi->guild_line_c();
        send_data.park_hmi_info.ADCS4_PA_GuideLineE_d =
            pb_avp_to_hmi->guild_line_d();
        send_data.park_hmi_info.ADCS4_PA_GuideLineE_Xmin =
            pb_avp_to_hmi->guild_line_xmin();
        send_data.park_hmi_info.ADCS4_PA_GuideLineE_Xmax  =
            pb_avp_to_hmi->guild_line_xmax();
        send_data.park_hmi_info.ADCS4_HourOfDay = pb_avp_to_hmi->hour_of_day();
        send_data.park_hmi_info.ADCS4_MinuteOfHour = pb_avp_to_hmi->minute_of_hour();
        send_data.park_hmi_info.ADCS4_SecondOfMinute = pb_avp_to_hmi->second_of_minute();
        send_data.park_hmi_info.ADCS11_NNS_distance = pb_avp_to_hmi->nns_distance();
        send_data.park_hmi_info.ADCS11_HPA_distance = pb_avp_to_hmi->hpa_distance();
        send_data.park_hmi_info.ADCS11_Parkingtimeremaining = pb_avp_to_hmi->park_time_remaining();
        return 0;
    }
    int TransEgo2Mcu(std::shared_ptr<hozon::planning::ADCTrajectory> proto_data, 
            hozon::netaos::AlgEgoToMcuFrame &data ) {
        auto Sample = proto_data->mutable_function_manager_out();
        data.header.seq = Sample->header().seq();
        data.header.stamp.sec = static_cast<uint64_t>(Sample->header().publish_stamp()) * 1e9 / 1e9;
        data.header.stamp.nsec = static_cast<uint64_t>(Sample->header().publish_stamp() * 1e9) - data.header.stamp.sec * 1e9;
        data.header.gnssStamp.sec = static_cast<uint64_t>(Sample->header().gnss_stamp()) * 1e9 / 1e9;
        data.header.gnssStamp.nsec = static_cast<uint64_t>(Sample->header().gnss_stamp() * 1e9) - data.header.stamp.sec * 1e9;
        // struct timespec time;
        // if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
        //     INTRA_LOG_WARN << "clock_gettime fail ";
        // }
        // struct timespec gnss_time;
        // if (0 != clock_gettime(CLOCK_MONOTONIC_RAW, &gnss_time)) {
        //     INTRA_LOG_WARN << "clock_gettime fail ";
        // }
        // data.header.stamp.sec = time.tv_sec;
        // data.header.stamp.nsec = time.tv_nsec;
        // data.header.gnssStamp.sec = gnss_time.tv_sec;
        // data.header.gnssStamp.nsec = gnss_time.tv_nsec;
        std::string frameid = Sample->header().frame_id();
        frameid = frameid.substr(0, stringSize);
        memset(data.header.frameId.data(), 0, stringSize);
        memcpy(data.header.frameId.data(), frameid.data(), frameid.size());
        data.SOC2FCT_TBD_u32_01 = Sample->soc_2_fct_tbd_u32_01();
        data.SOC2FCT_TBD_u32_02 = Sample->soc_2_fct_tbd_u32_02();
        data.SOC2FCT_TBD_u32_03 = Sample->soc_2_fct_tbd_u32_03();
        data.SOC2FCT_TBD_u32_04 = Sample->soc_2_fct_tbd_u32_04();
        data.SOC2FCT_TBD_u32_05 = Sample->soc_2_fct_tbd_u32_05();
        // NNP
        data.msg_ego_nnp.NNP_Active_OnOffSet = Sample->nnp_fct_out().nnp_active_on_off_set();
        data.msg_ego_nnp.Lanechangeinfor = Sample->nnp_fct_out().lane_change_infor();
        data.msg_ego_nnp.Lanechangedirection = Sample->nnp_fct_out().lane_change_direction();
        data.msg_ego_nnp.LCAudioPlay = Sample->nnp_fct_out().lc_audio_play();
        data.msg_ego_nnp.Lcsndrequest = Sample->nnp_fct_out().lcsndrequest();
        data.msg_ego_nnp.DCLCAudioplay = Sample->nnp_fct_out().dclc_audio_play();
        data.msg_ego_nnp.LaneChangeWarning = Sample->nnp_fct_out().lane_change_warning();
        data.msg_ego_nnp.LightRequest = Sample->nnp_fct_out().light_request();
        data.msg_ego_nnp.LaneChangePendingAlert = Sample->nnp_fct_out().lane_change_pending_alert();
        data.msg_ego_nnp.NNP_LightRemind = Sample->nnp_fct_out().nnp_light_remind();
        data.msg_ego_nnp.lateralCtrtakeover = Sample->nnp_fct_out().lateralctr_takeover();
        data.msg_ego_nnp.NNP_Scenarios_AudioPlay = Sample->nnp_fct_out().nnp_scenarios_audio_play();
        data.msg_ego_nnp.NNP_Scenarios = Sample->nnp_fct_out().nnp_scenarios();
        data.msg_ego_nnp.NNP_RINO_Status = Sample->nnp_fct_out().nnp_rino_status();
        data.msg_ego_nnp.PayModeConfirmFeedBack = Sample->nnp_fct_out().paymode_confirm_feedback();
        data.msg_ego_nnp.SpdAdaptComfirmFeedback = Sample->nnp_fct_out().spdadapt_comfirm_feedback();
        data.msg_ego_nnp.TSR_SpeedSign = Sample->nnp_fct_out().tsr_speedsign();
        data.msg_ego_nnp.ALC_mode = Sample->nnp_fct_out().aalc_mode();
        const auto& nnp_activation_conditions = Sample->nnp_fct_out().nnp_activation_conditions();
        data.msg_ego_nnp.NNP_indx_HDMapLocationNavi_u8 =
            SetUint8ByBit(nnp_activation_conditions.vehicle_in_hdmap(), nnp_activation_conditions.valid_of_lane_localization(), nnp_activation_conditions.valid_of_lane_routing(),
                        nnp_activation_conditions.vehicle_not_in_reverselane(), nnp_activation_conditions.vehicle_not_in_forbidlane(), nnp_activation_conditions.vehicle_not_in_otherforbidarea(),
                        Sample->nnp_fct_out().is_in_hdmap(), 0);
        data.msg_ego_nnp.NNP_indx_CrrntLaneCond_u8 = SetUint8ByBit(nnp_activation_conditions.appropriate_current_lane_curve(), nnp_activation_conditions.appropriate_current_lane_width(),
                                                                    nnp_activation_conditions.appropriate_current_lane_headingerr(), 0, 0, 0, 0, 0);
        data.msg_ego_nnp.NNP_d_Distance2OnRamp_sg = Sample->nnp_fct_out().nnp_d_distance2_onramp_sg();
        data.msg_ego_nnp.NNP_d_Distance2DownRamp_sg = Sample->nnp_fct_out().nnp_d_distance2_downramp_sg();
        data.msg_ego_nnp.NNP_d_DistanceIntoODD_sg = Sample->nnp_fct_out().nnp_d_distance_into_odd_sg();
        data.msg_ego_nnp.NNP_d_DistanceOutofODD_sg = Sample->nnp_fct_out().nnp_d_distance_outof_odd_sg();
        data.msg_ego_nnp.NNP_d_CrrntLaneWidth_sg = Sample->nnp_fct_out().crrent_lane_mg().nnp_d_crrntlanewidth_sg();
        data.msg_ego_nnp.NNP_crv_CrrntLaneCurve_sg = Sample->nnp_fct_out().crrent_lane_mg().nnp_crv_crrntlanecurve_sg();
        data.msg_ego_nnp.NNP_rad_CrrntLaneHead_sg = Sample->nnp_fct_out().crrent_lane_mg().nnp_rad_crrntlanehead_sg();
        data.msg_ego_nnp.NNP_is_NNPMRMFlf_bl = Sample->nnp_fct_out().nnp_is_nnpmrmflf_bl();
        data.msg_ego_nnp.NNP_is_NNPMRMDoneFlf_bl = Sample->nnp_fct_out().nnp_is_nnpmrm_doneflf_bl();
        data.msg_ego_nnp.NNP_is_NNPEMFlf_bl = Sample->nnp_fct_out().nnp_is_nnpemflf_bl();
        data.msg_ego_nnp.NNP_is_NNPEMDoneFlf_bl = Sample->nnp_fct_out().nnp_is_nnpem_doneflf_bl();
        const auto& nnp_fault = Sample->nnp_fct_out().nnp_software_fault();
        data.msg_ego_nnp.NNP_indx_NNPSoftwareFault_u8 = SetUint8ByBit(nnp_fault.plan_trajectory_success(), nnp_fault.planning_success(), 0, 0, 0, 0, 0, 0);
        data.msg_ego_nnp.HighBeamReqSt = Sample->nnp_fct_out().light_signal_reqst().highbeamreqst();
        data.msg_ego_nnp.LowBeamReqSt = Sample->nnp_fct_out().light_signal_reqst().lowbeamreqst();
        data.msg_ego_nnp.LowHighBeamReqSt = Sample->nnp_fct_out().light_signal_reqst().lowhighbeamreqst();
        data.msg_ego_nnp.HazardLampReqSt = Sample->nnp_fct_out().light_signal_reqst().hazardlampreqst();
        data.msg_ego_nnp.HornReqSt = Sample->nnp_fct_out().light_signal_reqst().hornreqst();

        // AVP

        data.msg_ego_avp.m_iuss_state_obs = Sample->avp_fct_out().iuss_state_obs();
        data.msg_ego_avp.need_replan_stop = Sample->avp_fct_out().need_replan_stop();
        data.msg_ego_avp.plan_trigger = Sample->avp_fct_out().plan_trigger();
        data.msg_ego_avp.control_enable = Sample->avp_fct_out().control_enable();
        data.msg_ego_avp.parking_status = Sample->avp_fct_out().parking_status();
        return 0;
    }

    int Trans2Traj( std::shared_ptr<hozon::planning::ADCTrajectory> Sample,
         hozon::netaos::HafEgoTrajectory &data) {
        
        data.header.seq = Sample->mutable_header()->seq();
        data.header.stamp.sec = static_cast<uint64_t>(Sample->mutable_header()->publish_stamp()) * 1e9 / 1e9;
        data.header.stamp.nsec = static_cast<uint64_t>(Sample->mutable_header()->publish_stamp() * 1e9) - data.header.stamp.sec * 1e9;
        data.header.gnssStamp.sec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp()) * 1e9 / 1e9;
        data.header.gnssStamp.nsec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp() * 1e9) - data.header.stamp.sec * 1e9;
        // struct timespec time;
        // if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
        //     INTRA_LOG_WARN << "clock_gettime fail ";
        // }
        // struct timespec gnss_time;
        // if (0 != clock_gettime(CLOCK_MONOTONIC_RAW, &gnss_time)) {
        //     INTRA_LOG_WARN << "clock_gettime fail ";
        // }
        // data.header.stamp.sec = time.tv_sec;
        // data.header.stamp.nsec = time.tv_nsec;
        // data.header.gnssStamp.sec = gnss_time.tv_sec;
        // data.header.gnssStamp.nsec = gnss_time.tv_nsec;

        data.locSeq = 0;
        data.isEstop = Sample->mutable_estop()->is_estop();
        data.trajectoryLength = Sample->total_path_length();
        data.trajectoryPeriod = Sample->total_path_time();
        data.isReplanning = Sample->is_replan();
        data.gear = (uint8_t)Sample->gear();
        data.trajectoryType = Sample->trajectory_type();
        data.functionMode = Sample->function_mode();
        // data.drivingMode = Sample->is_vehicle_reference_frame();
        data.driviningMode = Sample->driving_mode();
        data.trajectoryValidPointsSize = Sample->trajectory_point_size();
        size_t traj_size = Sample->trajectory_point_size();
        for (size_t i = 0; i < traj_size; i++) {
            if (i == trajectoryPointsLength) {
                break;
            }
            const auto& point = Sample->trajectory_point(i);
            if (i == 0) {
                data.trajectoryPoint_reference_x = point.path_point().x();
                data.trajectoryPoint_reference_y = point.path_point().y();
            }
            data.trajectoryPoints[i].speed = point.v();
            data.trajectoryPoints[i].acc = point.a();
            data.trajectoryPoints[i].timeRelative = point.relative_time();
            //  data.trajectoryPoints[i].steerAngle = point.steer();
            data.trajectoryPoints[i].x = point.path_point().x() - data.trajectoryPoint_reference_x;
            data.trajectoryPoints[i].y = point.path_point().y() - data.trajectoryPoint_reference_y;
            data.trajectoryPoints[i].z = point.path_point().z();
            data.trajectoryPoints[i].s = point.path_point().s();
            data.trajectoryPoints[i].theta = point.path_point().theta() + Sample->utm2gcs_heading_offset();
            data.trajectoryPoints[i].curvature = point.path_point().kappa();
        }
        data.proj_heading_offset = Sample->utm2gcs_heading_offset();
        const auto nnp_fct_out = Sample->function_manager_out().nnp_fct_out();
        memset(&data.reserve, 0, sizeof(data.reserve));
        data.reserve[0] = static_cast<uint16_t>(nnp_fct_out.lane_change_infor());
        data.reserve[1] = static_cast<uint16_t>(Sample->replan_type());
        data.reserve[2] = static_cast<uint16_t>(Sample->received_ehp_counter());
        return 0;
    }

private:

};
}
}
}