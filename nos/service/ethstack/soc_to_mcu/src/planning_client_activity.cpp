/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: planning_client_activity
 */

#include "planning_client_activity.h"

using namespace std;

namespace hozon {
namespace netaos {
namespace intra {

PlanningClientActivity::PlanningClientActivity() {}

PlanningClientActivity::~PlanningClientActivity() {
    // Stop();
}

void PlanningClientActivity::Init(std::shared_ptr<Skeleton> skeleton, std::string drivemode) {
    skeleton_ = skeleton;
    seq = 0;
    int32_t ret = reader.Init(0, planningtopic.c_str(), std::bind(&PlanningClientActivity::cb, this, std::placeholders::_1));
    INTRA_LOG_INFO << "end..." << ret;
}

void PlanningClientActivity::cb(std::shared_ptr<hozon::planning::ADCTrajectory> msg) {
    // std::string text;
    // bool res = google::protobuf::TextFormat::PrintToString(*msg, &text);
    // INTRA_LOG_INFO << "hozon::perception::ADCTrajectory :"
    //                << " res " << res << " text: " << text.c_str();
    sendPlanningData(msg);
    sendEgoToMcuData((msg->function_manager_out()));
    seq++;
}
void PlanningClientActivity::Stop() {
    INTRA_LOG_INFO << "begin...";
    reader.Deinit();
    INTRA_LOG_INFO << "end...";
}

void PlanningClientActivity::sendEgoToMcuData(const hozon::functionmanager::FunctionManagerOut Sample) {
    std::shared_ptr<::hozon::netaos::AlgEgoToMcuFrame> data_ = std::make_shared<::hozon::netaos::AlgEgoToMcuFrame>();
    if (data_ == nullptr) {
        INTRA_LOG_ERROR << "AlgEgoToMcuFrame dataSkeleton->hozonEvent.Allocate() got nullptr!";
        return;
    }
    data_->header.seq = Sample.header().seq();
    data_->header.stamp.sec = static_cast<uint64_t>(Sample.header().publish_stamp()) * 1e9 / 1e9;
    data_->header.stamp.nsec = static_cast<uint64_t>(Sample.header().publish_stamp() * 1e9) - data_->header.stamp.sec * 1e9;
    data_->header.gnssStamp.sec = static_cast<uint64_t>(Sample.header().gnss_stamp()) * 1e9 / 1e9;
    data_->header.gnssStamp.nsec = static_cast<uint64_t>(Sample.header().gnss_stamp() * 1e9) - data_->header.stamp.sec * 1e9;
    // struct timespec time;
    // if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
    //     INTRA_LOG_WARN << "clock_gettime fail ";
    // }
    // struct timespec gnss_time;
    // if (0 != clock_gettime(CLOCK_MONOTONIC_RAW, &gnss_time)) {
    //     INTRA_LOG_WARN << "clock_gettime fail ";
    // }
    // data_->header.stamp.sec = time.tv_sec;
    // data_->header.stamp.nsec = time.tv_nsec;
    // data_->header.gnssStamp.sec = gnss_time.tv_sec;
    // data_->header.gnssStamp.nsec = gnss_time.tv_nsec;
    std::string frameid = Sample.header().frame_id();
    frameid = frameid.substr(0, stringSize);
    memset(data_->header.frameId.data(), 0, stringSize);
    memcpy(data_->header.frameId.data(), frameid.data(), frameid.size());
    data_->SOC2FCT_TBD_u32_01 = Sample.soc_2_fct_tbd_u32_01();
    data_->SOC2FCT_TBD_u32_02 = Sample.soc_2_fct_tbd_u32_02();
    data_->SOC2FCT_TBD_u32_03 = Sample.soc_2_fct_tbd_u32_03();
    data_->SOC2FCT_TBD_u32_04 = Sample.soc_2_fct_tbd_u32_04();
    data_->SOC2FCT_TBD_u32_05 = Sample.soc_2_fct_tbd_u32_05();
    // NNP
    data_->msg_ego_nnp.NNP_Active_OnOffSet = Sample.nnp_fct_out().nnp_active_on_off_set();
    data_->msg_ego_nnp.Lanechangeinfor = Sample.nnp_fct_out().lane_change_infor();
    data_->msg_ego_nnp.Lanechangedirection = Sample.nnp_fct_out().lane_change_direction();
    data_->msg_ego_nnp.LCAudioPlay = Sample.nnp_fct_out().lc_audio_play();
    data_->msg_ego_nnp.Lcsndrequest = Sample.nnp_fct_out().lcsndrequest();
    data_->msg_ego_nnp.DCLCAudioplay = Sample.nnp_fct_out().dclc_audio_play();
    data_->msg_ego_nnp.LaneChangeWarning = Sample.nnp_fct_out().lane_change_warning();
    data_->msg_ego_nnp.LightRequest = Sample.nnp_fct_out().light_request();
    data_->msg_ego_nnp.LaneChangePendingAlert = Sample.nnp_fct_out().lane_change_pending_alert();
    data_->msg_ego_nnp.NNP_LightRemind = Sample.nnp_fct_out().nnp_light_remind();
    data_->msg_ego_nnp.lateralCtrtakeover = Sample.nnp_fct_out().lateralctr_takeover();
    data_->msg_ego_nnp.NNP_Scenarios_AudioPlay = Sample.nnp_fct_out().nnp_scenarios_audio_play();
    data_->msg_ego_nnp.NNP_Scenarios = Sample.nnp_fct_out().nnp_scenarios();
    data_->msg_ego_nnp.NNP_RINO_Status = Sample.nnp_fct_out().nnp_rino_status();
    data_->msg_ego_nnp.PayModeConfirmFeedBack = Sample.nnp_fct_out().paymode_confirm_feedback();
    data_->msg_ego_nnp.SpdAdaptComfirmFeedback = Sample.nnp_fct_out().spdadapt_comfirm_feedback();
    data_->msg_ego_nnp.TSR_SpeedSign = Sample.nnp_fct_out().tsr_speedsign();
    data_->msg_ego_nnp.ALC_mode = Sample.nnp_fct_out().aalc_mode();
    const auto& nnp_activation_conditions = Sample.nnp_fct_out().nnp_activation_conditions();
    data_->msg_ego_nnp.NNP_indx_HDMapLocationNavi_u8 =
        SetUint8ByBit(nnp_activation_conditions.vehicle_in_hdmap(), nnp_activation_conditions.valid_of_lane_localization(), nnp_activation_conditions.valid_of_lane_routing(),
                      nnp_activation_conditions.vehicle_not_in_reverselane(), nnp_activation_conditions.vehicle_not_in_forbidlane(), nnp_activation_conditions.vehicle_not_in_otherforbidarea(),
                      Sample.nnp_fct_out().is_in_hdmap(), 0);
    data_->msg_ego_nnp.NNP_indx_CrrntLaneCond_u8 = SetUint8ByBit(nnp_activation_conditions.appropriate_current_lane_curve(), nnp_activation_conditions.appropriate_current_lane_width(),
                                                                 nnp_activation_conditions.appropriate_current_lane_headingerr(), 0, 0, 0, 0, 0);
    data_->msg_ego_nnp.NNP_d_Distance2OnRamp_sg = Sample.nnp_fct_out().nnp_d_distance2_onramp_sg();
    data_->msg_ego_nnp.NNP_d_Distance2DownRamp_sg = Sample.nnp_fct_out().nnp_d_distance2_downramp_sg();
    data_->msg_ego_nnp.NNP_d_DistanceIntoODD_sg = Sample.nnp_fct_out().nnp_d_distance_into_odd_sg();
    data_->msg_ego_nnp.NNP_d_DistanceOutofODD_sg = Sample.nnp_fct_out().nnp_d_distance_outof_odd_sg();
    data_->msg_ego_nnp.NNP_d_CrrntLaneWidth_sg = Sample.nnp_fct_out().crrent_lane_mg().nnp_d_crrntlanewidth_sg();
    data_->msg_ego_nnp.NNP_crv_CrrntLaneCurve_sg = Sample.nnp_fct_out().crrent_lane_mg().nnp_crv_crrntlanecurve_sg();
    data_->msg_ego_nnp.NNP_rad_CrrntLaneHead_sg = Sample.nnp_fct_out().crrent_lane_mg().nnp_rad_crrntlanehead_sg();
    data_->msg_ego_nnp.NNP_is_NNPMRMFlf_bl = Sample.nnp_fct_out().nnp_is_nnpmrmflf_bl();
    data_->msg_ego_nnp.NNP_is_NNPMRMDoneFlf_bl = Sample.nnp_fct_out().nnp_is_nnpmrm_doneflf_bl();
    data_->msg_ego_nnp.NNP_is_NNPEMFlf_bl = Sample.nnp_fct_out().nnp_is_nnpemflf_bl();
    data_->msg_ego_nnp.NNP_is_NNPEMDoneFlf_bl = Sample.nnp_fct_out().nnp_is_nnpem_doneflf_bl();
    const auto& nnp_fault = Sample.nnp_fct_out().nnp_software_fault();
    data_->msg_ego_nnp.NNP_indx_NNPSoftwareFault_u8 = SetUint8ByBit(nnp_fault.plan_trajectory_success(), nnp_fault.planning_success(), 0, 0, 0, 0, 0, 0);
    data_->msg_ego_nnp.HighBeamReqSt = Sample.nnp_fct_out().light_signal_reqst().highbeamreqst();
    data_->msg_ego_nnp.LowBeamReqSt = Sample.nnp_fct_out().light_signal_reqst().lowbeamreqst();
    data_->msg_ego_nnp.LowHighBeamReqSt = Sample.nnp_fct_out().light_signal_reqst().lowhighbeamreqst();
    data_->msg_ego_nnp.HazardLampReqSt = Sample.nnp_fct_out().light_signal_reqst().hazardlampreqst();
    data_->msg_ego_nnp.HornReqSt = Sample.nnp_fct_out().light_signal_reqst().hornreqst();

    // AVP

    data_->msg_ego_avp.m_iuss_state_obs = Sample.avp_fct_out().iuss_state_obs();
    data_->msg_ego_avp.need_replan_stop = Sample.avp_fct_out().need_replan_stop();
    data_->msg_ego_avp.plan_trigger = Sample.avp_fct_out().plan_trigger();
    data_->msg_ego_avp.control_enable = Sample.avp_fct_out().control_enable();
    data_->msg_ego_avp.parking_status = Sample.avp_fct_out().parking_status();
    if (seq % 100 == 0) {
        INTRA_LOG_INFO << "sendEgoToMcuData::  "
                       << " seq: " << data_->header.seq;
    }
    if (skeleton_) {
        skeleton_->AlgEgoToMCU.Send(*data_);
    } else {
        INTRA_LOG_ERROR << "skeleton  is nullptr...";
    }
}

void PlanningClientActivity::sendPlanningData(const std::shared_ptr<hozon::planning::ADCTrajectory> Sample) {
    std::shared_ptr<::hozon::netaos::HafEgoTrajectory> data_ = std::make_shared<::hozon::netaos::HafEgoTrajectory>();
    if (data_ == nullptr) {
        INTRA_LOG_ERROR << "TrajData dataSkeleton->hozonEvent.Allocate() got nullptr!";
        return;
    }
    data_->header.seq = Sample->mutable_header()->seq();
    data_->header.stamp.sec = static_cast<uint64_t>(Sample->mutable_header()->publish_stamp()) * 1e9 / 1e9;
    data_->header.stamp.nsec = static_cast<uint64_t>(Sample->mutable_header()->publish_stamp() * 1e9) - data_->header.stamp.sec * 1e9;
    data_->header.gnssStamp.sec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp()) * 1e9 / 1e9;
    data_->header.gnssStamp.nsec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp() * 1e9) - data_->header.stamp.sec * 1e9;
    // struct timespec time;
    // if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
    //     INTRA_LOG_WARN << "clock_gettime fail ";
    // }
    // struct timespec gnss_time;
    // if (0 != clock_gettime(CLOCK_MONOTONIC_RAW, &gnss_time)) {
    //     INTRA_LOG_WARN << "clock_gettime fail ";
    // }
    // data_->header.stamp.sec = time.tv_sec;
    // data_->header.stamp.nsec = time.tv_nsec;
    // data_->header.gnssStamp.sec = gnss_time.tv_sec;
    // data_->header.gnssStamp.nsec = gnss_time.tv_nsec;

    data_->locSeq = 0;
    data_->isEstop = Sample->mutable_estop()->is_estop();
    data_->trajectoryLength = Sample->total_path_length();
    data_->trajectoryPeriod = Sample->total_path_time();
    data_->isReplanning = Sample->is_replan();
    data_->gear = (uint8_t)Sample->gear();
    data_->trajectoryType = Sample->trajectory_type();
    data_->functionMode = Sample->function_mode();
    // data_->drivingMode = Sample->is_vehicle_reference_frame();
    data_->driviningMode = Sample->driving_mode();
    data_->trajectoryValidPointsSize = Sample->trajectory_point_size();
    size_t traj_size = Sample->trajectory_point_size();
    for (size_t i = 0; i < traj_size; i++) {
        if (i == trajectoryPointsLength) {
            break;
        }
        const auto& point = Sample->trajectory_point(i);
        if (i == 0) {
            data_->trajectoryPoint_reference_x = point.path_point().x();
            data_->trajectoryPoint_reference_y = point.path_point().y();
        }
        data_->trajectoryPoints[i].speed = point.v();
        data_->trajectoryPoints[i].acc = point.a();
        data_->trajectoryPoints[i].timeRelative = point.relative_time();
        //  data_->trajectoryPoints[i].steerAngle = point.steer();
        data_->trajectoryPoints[i].x = point.path_point().x() - data_->trajectoryPoint_reference_x;
        data_->trajectoryPoints[i].y = point.path_point().y() - data_->trajectoryPoint_reference_y;
        data_->trajectoryPoints[i].z = point.path_point().z();
        data_->trajectoryPoints[i].s = point.path_point().s();
        data_->trajectoryPoints[i].theta = point.path_point().theta() + Sample->utm2gcs_heading_offset();
        data_->trajectoryPoints[i].curvature = point.path_point().kappa();
    }
    data_->proj_heading_offset = Sample->utm2gcs_heading_offset();
    const auto nnp_fct_out = Sample->function_manager_out().nnp_fct_out();
    memset(&data_->reserve, 0, sizeof(data_->reserve));
    data_->reserve[0] = static_cast<uint16_t>(nnp_fct_out.lane_change_infor());
    data_->reserve[1] = static_cast<uint16_t>(Sample->replan_type());
    data_->reserve[2] = static_cast<uint16_t>(Sample->received_ehp_counter());

    data_->utmzoneID = Sample->utm_zone_id();
    if (seq % 100 == 0) {
        INTRA_LOG_INFO << "sendPlanningData::  "
                       << " seq: " << data_->header.seq;
    }
    if (skeleton_) {
        skeleton_->TrajData.Send(*data_);
    } else {
        INTRA_LOG_ERROR << "skeleton  is nullptr...";
    }
}

}  // namespace intra
}  // namespace netaos
}  // namespace hozon
