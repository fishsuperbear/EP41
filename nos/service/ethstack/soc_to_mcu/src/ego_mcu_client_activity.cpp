/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-09-22 09:47:16
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-09-22 17:20:03
 * @FilePath: /nos/service/ethstack/soc_to_mcu/src/ego_mcu_client_activity.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: ego_mcu_client_activity
 */

#include "ego_mcu_client_activity.h"
using namespace std;

namespace hozon {
namespace netaos {
namespace intra {

EgoToMcuClientActivity::EgoToMcuClientActivity() : egomcu_event_proxy_(std::make_shared<AlgEgo2McuFramePubSubType>()) {}

EgoToMcuClientActivity::~EgoToMcuClientActivity() {
    // Stop();
}

void EgoToMcuClientActivity::Init(std::shared_ptr<Skeleton> skeleton) {
    seq = 0;
    skeleton_ = skeleton;
    if (egomcu_event_proxy_.Init(0, soctomcutopic.c_str()) == -1) {
        INTRA_LOG_ERROR << "egomcu_event_proxy_.Init Error:";
    }
    egomcu_event_proxy_.Listen(std::bind(&EgoToMcuClientActivity::cb, this));
}

void EgoToMcuClientActivity::cb() {
    if (egomcu_event_proxy_.IsMatched()) {
        std::shared_ptr<AlgEgo2McuFrame> data = std::make_shared<AlgEgo2McuFrame>();
        if (0 == egomcu_event_proxy_.Take(data)) {
            sendEgoToMcuData(data);
        } else {
            INTRA_LOG_WARN << "cfg_paramupdatedatares_event_proxy_ Take error";
        }
    }
}

void EgoToMcuClientActivity::sendEgoToMcuData(const std::shared_ptr<AlgEgo2McuFrame> Sample) {
    std::shared_ptr<::hozon::netaos::AlgEgoToMcuFrame> data_ = std::make_shared<::hozon::netaos::AlgEgoToMcuFrame>();
    if (data_ == nullptr) {
        INTRA_LOG_ERROR << "SnsrFsnLaneDateEvent1 dataSkeleton->hozonEvent.Allocate() got nullptr!";
        return;
    }
    data_->header.seq = 0;
    // data_->header.stamp.sec = static_cast<uint64_t>(Sample->mutable_header()->publish_stamp()) * 1e9 / 1e9;
    // data_->header.stamp.nsec = static_cast<uint64_t>(Sample->mutable_header()->publish_stamp() * 1e9) - data_->header.stamp.sec * 1e9;
    // data_->header.gnssStamp.sec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp()) * 1e9 / 1e9;
    // data_->header.gnssStamp.nsec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp() * 1e9) - data_->header.stamp.sec * 1e9;
    struct timespec time;
    if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
        INTRA_LOG_WARN << "clock_gettime fail ";
    }
    struct timespec gnss_time;
    if (0 != clock_gettime(CLOCK_MONOTONIC_RAW, &gnss_time)) {
        INTRA_LOG_WARN << "clock_gettime fail ";
    }
    data_->header.stamp.sec = time.tv_sec;
    data_->header.stamp.nsec = time.tv_nsec;
    data_->header.gnssStamp.sec = gnss_time.tv_sec;
    data_->header.gnssStamp.nsec = gnss_time.tv_nsec;
    std::string frameid = "";
    frameid = frameid.substr(0, stringSize);
    memset(data_->header.frameId.data(), 0, stringSize);
    memcpy(data_->header.frameId.data(), frameid.data(), frameid.size());
    /*
    data_->msg_ego_nnp.NNP_Active_OnOffSet = Sample->msg_ego_nnp().NNP_Active_OnOffSet();
    data_->msg_ego_nnp.Lanechangeinfor = Sample->msg_ego_nnp().Lanechangeinfor();
    data_->msg_ego_nnp.Lanechangedirection = Sample->msg_ego_nnp().Lanechangedirection();
    data_->msg_ego_nnp.LCAudioPlay = Sample->msg_ego_nnp().LCAudioPlay();
    data_->msg_ego_nnp.Lcsndrequest = Sample->msg_ego_nnp().Lcsndrequest();
    data_->msg_ego_nnp.DCLCAudioplay = Sample->msg_ego_nnp().DCLCAudioplay();
    data_->msg_ego_nnp.LaneChangeWarning = Sample->msg_ego_nnp().LaneChangeWarning();
    data_->msg_ego_nnp.LightRequest = Sample->msg_ego_nnp().LightRequest();
    data_->msg_ego_nnp.LaneChangePendingAlert = Sample->msg_ego_nnp().LaneChangePendingAlert();
    data_->msg_ego_nnp.NNP_LightRemind = Sample->msg_ego_nnp().NNP_LightRemind();
    data_->msg_ego_nnp.lateralCtrtakeover = Sample->msg_ego_nnp().lateralCtrtakeover();
    data_->msg_ego_nnp.NNP_Scenarios_AudioPlay = Sample->msg_ego_nnp().NNP_Scenarios_AudioPlay();
    data_->msg_ego_nnp.NNP_Scenarios = Sample->msg_ego_nnp().NNP_Scenarios();
    data_->msg_ego_nnp.NNP_RINO_Status = Sample->msg_ego_nnp().NNP_RINO_Status();
    data_->msg_ego_nnp.PayModeConfirmFeedBack = Sample->msg_ego_nnp().PayModeConfirmFeedBack();
    data_->msg_ego_nnp.SpdAdaptComfirmFeedback = Sample->msg_ego_nnp().SpdAdaptComfirmFeedback();
    data_->msg_ego_nnp.TSR_SpeedSign = Sample->msg_ego_nnp().TSR_SpeedSign();
    data_->msg_ego_nnp.ALC_mode = Sample->msg_ego_nnp().ALC_mode();
    data_->msg_ego_nnp.NNP_indx_HDMapLocationNavi_u8 = Sample->msg_ego_nnp().NNP_indx_HDMapLocationNavi_u8();
    data_->msg_ego_nnp.NNP_indx_CrrntLaneCond_u8 = Sample->msg_ego_nnp().NNP_indx_CrrntLaneCond_u8();
    data_->msg_ego_nnp.NNP_d_Distance2OnRamp_sg = Sample->msg_ego_nnp().NNP_d_Distance2OnRamp_sg();
    data_->msg_ego_nnp.NNP_d_Distance2DownRamp_sg = Sample->msg_ego_nnp().NNP_d_Distance2DownRamp_sg();
    data_->msg_ego_nnp.NNP_d_DistanceIntoODD_sg = Sample->msg_ego_nnp().NNP_d_DistanceIntoODD_sg();
    data_->msg_ego_nnp.NNP_d_DistanceOutofODD_sg = Sample->msg_ego_nnp().NNP_d_DistanceOutofODD_sg();
    data_->msg_ego_nnp.NNP_d_CrrntLaneWidth_sg = Sample->msg_ego_nnp().NNP_d_CrrntLaneWidth_sg();
    data_->msg_ego_nnp.NNP_crv_CrrntLaneCurve_sg = Sample->msg_ego_nnp().NNP_crv_CrrntLaneCurve_sg();
    data_->msg_ego_nnp.NNP_rad_CrrntLaneHead_sg = Sample->msg_ego_nnp().NNP_rad_CrrntLaneHead_sg();
    data_->msg_ego_nnp.NNP_is_NNPMRMFlf_bl = Sample->msg_ego_nnp().NNP_is_NNPMRMFlf_bl();
    data_->msg_ego_nnp.NNP_is_NNPMRMDoneFlf_bl = Sample->msg_ego_nnp().NNP_is_NNPMRMDoneFlf_bl();
    data_->msg_ego_nnp.NNP_is_NNPEMFlf_bl = Sample->msg_ego_nnp().NNP_is_NNPEMFlf_bl();
    data_->msg_ego_nnp.NNP_is_NNPEMDoneFlf_bl = Sample->msg_ego_nnp().NNP_is_NNPEMDoneFlf_bl();
    data_->msg_ego_nnp.NNP_indx_NNPSoftwareFault_u8 = Sample->msg_ego_nnp().NNP_indx_NNPSoftwareFault_u8();
    data_->msg_ego_nnp.HighBeamReqSt = Sample->msg_ego_nnp().HighBeamReqSt();
    data_->msg_ego_nnp.LowBeamReqSt = Sample->msg_ego_nnp().LowBeamReqSt();
    data_->msg_ego_nnp.LowHighBeamReqSt = Sample->msg_ego_nnp().LowHighBeamReqSt();
    data_->msg_ego_nnp.HazardLampReqSt = Sample->msg_ego_nnp().HazardLampReqSt();
    data_->msg_ego_nnp.HornReqSt = Sample->msg_ego_nnp().HornReqSt();
    data_->msg_ego_avp.m_iuss_state_obs = Sample->msg_ego_avp().m_iuss_state_obs();
    data_->msg_ego_avp.need_replan_stop = Sample->msg_ego_avp().need_replan_stop();
    data_->msg_ego_avp.plan_trigger = Sample->msg_ego_avp().plan_trigger();
    data_->msg_ego_avp.control_enable = Sample->msg_ego_avp().control_enable();
    data_->msg_ego_avp.parking_status = Sample->msg_ego_avp().parking_status();
    data_->SOC2FCT_TBD_u32_01 = Sample->SOC2FCT_TBD_u32_01();
    data_->SOC2FCT_TBD_u32_02 = Sample->SOC2FCT_TBD_u32_02();
    data_->SOC2FCT_TBD_u32_03 = Sample->SOC2FCT_TBD_u32_03();
    data_->SOC2FCT_TBD_u32_04 = Sample->SOC2FCT_TBD_u32_04();
    data_->SOC2FCT_TBD_u32_05 = Sample->SOC2FCT_TBD_u32_05();
    INTRA_LOG_INFO << "sendLanesData::  "
                   << " seq: " << data_->header.seq;
                   */

    //   data_->SOC2FCT_TBD_u32_01 = Sample->soc_2_fct_tbd_u32_01();
    //   data_->SOC2FCT_TBD_u32_02 = Sample->soc_2_fct_tbd_u32_02();
    //   data_->SOC2FCT_TBD_u32_03 = Sample->soc_2_fct_tbd_u32_03();
    //   data_->SOC2FCT_TBD_u32_04 = Sample->soc_2_fct_tbd_u32_04();
    //   data_->SOC2FCT_TBD_u32_05 = Sample->soc_2_fct_tbd_u32_05();
    //   // NNP
    //   data_->msg_ego_nnp.NNP_Active_OnOffSet =
    //       Sample->nnp_fct_out().nnp_active_on_off_set();
    //   data_->msg_ego_nnp.Lanechangeinfor =
    //       Sample->nnp_fct_out().lane_change_infor();
    //   data_->msg_ego_nnp.Lanechangedirection =
    //       Sample->nnp_fct_out().lane_change_direction();
    //   data_->msg_ego_nnp.LCAudioPlay =
    //       Sample->nnp_fct_out().lc_audio_play();
    //   data_->msg_ego_nnp.Lcsndrequest =
    //       Sample->nnp_fct_out().lcsndrequest();
    //   data_->msg_ego_nnp.DCLCAudioplay =
    //       Sample->nnp_fct_out().dclc_audio_play();
    //   data_->msg_ego_nnp.LaneChangeWarning =
    //       Sample->nnp_fct_out().lane_change_warning();
    //   data_->msg_ego_nnp.LightRequest =
    //       Sample->nnp_fct_out().light_request();
    //   data_->msg_ego_nnp.LaneChangePendingAlert =
    //       Sample->nnp_fct_out().lane_change_pending_alert();
    //   data_->msg_ego_nnp.NNP_LightRemind =
    //       Sample->nnp_fct_out().nnp_light_remind();
    //   data_->msg_ego_nnp.lateralCtrtakeover =
    //       Sample->nnp_fct_out().lateralctr_takeover();
    //   data_->msg_ego_nnp.NNP_Scenarios_AudioPlay =
    //       Sample->nnp_fct_out().nnp_scenarios_audio_play();
    //   data_->msg_ego_nnp.NNP_Scenarios =
    //       Sample->nnp_fct_out().nnp_scenarios();
    //   data_->msg_ego_nnp.NNP_RINO_Status =
    //       Sample->nnp_fct_out().nnp_rino_status();
    //   data_->msg_ego_nnp.PayModeConfirmFeedBack =
    //       Sample->nnp_fct_out().paymode_confirm_feedback();
    //   data_->msg_ego_nnp.SpdAdaptComfirmFeedback =
    //       Sample->nnp_fct_out().spdadapt_comfirm_feedback();
    //   data_->msg_ego_nnp.TSR_SpeedSign =
    //       Sample->nnp_fct_out().tsr_speedsign();
    //   data_->msg_ego_nnp.ALC_mode = Sample->nnp_fct_out().aalc_mode();
    //   const auto& nnp_activation_conditions =
    //       Sample->nnp_fct_out().nnp_activation_conditions();
    //   data_->msg_ego_nnp.NNP_indx_HDMapLocationNavi_u8 =
    //       SetUint8ByBit(nnp_activation_conditions.vehicle_in_hdmap(),
    //                     nnp_activation_conditions.valid_of_lane_localization(),
    //                     nnp_activation_conditions.valid_of_lane_routing(),
    //                     nnp_activation_conditions.vehicle_not_in_reverselane(),
    //                     nnp_activation_conditions.vehicle_not_in_forbidlane(),
    //                     nnp_activation_conditions.vehicle_not_in_otherforbidarea(),
    //                     Sample->nnp_fct_out().is_in_hdmap(), 0);
    //   data_->msg_ego_nnp.NNP_indx_CrrntLaneCond_u8 = SetUint8ByBit(
    //       nnp_activation_conditions.appropriate_current_lane_curve(),
    //       nnp_activation_conditions.appropriate_current_lane_width(),
    //       nnp_activation_conditions.appropriate_current_lane_headingerr(), 0, 0, 0,
    //       0, 0);
    //   data_->msg_ego_nnp.NNP_d_Distance2OnRamp_sg =
    //       Sample->nnp_fct_out().nnp_d_distance2_onramp_sg();
    //   data_->msg_ego_nnp.NNP_d_Distance2DownRamp_sg =
    //       Sample->nnp_fct_out().nnp_d_distance2_downramp_sg();
    //   data_->msg_ego_nnp.NNP_d_DistanceIntoODD_sg =
    //       Sample->nnp_fct_out().nnp_d_distance_into_odd_sg();
    //   data_->msg_ego_nnp.NNP_d_DistanceOutofODD_sg =
    //       Sample->nnp_fct_out().nnp_d_distance_outof_odd_sg();
    //   data_->msg_ego_nnp.NNP_d_CrrntLaneWidth_sg =
    //       Sample->nnp_fct_out().crrent_lane_mg().nnp_d_crrntlanewidth_sg();
    //   data_->msg_ego_nnp.NNP_crv_CrrntLaneCurve_sg =
    //       Sample->nnp_fct_out().crrent_lane_mg().nnp_crv_crrntlanecurve_sg();
    //   data_->msg_ego_nnp.NNP_rad_CrrntLaneHead_sg =
    //       Sample->nnp_fct_out().crrent_lane_mg().nnp_rad_crrntlanehead_sg();
    //   data_->msg_ego_nnp.NNP_is_NNPMRMFlf_bl =
    //       Sample->nnp_fct_out().nnp_is_nnpmrmflf_bl();
    //   data_->msg_ego_nnp.NNP_is_NNPMRMDoneFlf_bl =
    //       Sample->nnp_fct_out().nnp_is_nnpmrm_doneflf_bl();
    //   data_->msg_ego_nnp.NNP_is_NNPEMFlf_bl =
    //       Sample->nnp_fct_out().nnp_is_nnpemflf_bl();
    //   data_->msg_ego_nnp.NNP_is_NNPEMDoneFlf_bl =
    //       Sample->nnp_fct_out().nnp_is_nnpem_doneflf_bl();
    //   const auto& nnp_fault = Sample->nnp_fct_out().nnp_software_fault();
    //   data_->msg_ego_nnp.NNP_indx_NNPSoftwareFault_u8 =
    //       SetUint8ByBit(nnp_fault.plan_trajectory_success(),
    //                     nnp_fault.planning_success(), 0, 0, 0, 0, 0, 0);
    //   data_->msg_ego_nnp.HighBeamReqSt =
    //       Sample->nnp_fct_out().light_signal_reqst().highbeamreqst();
    //   data_->msg_ego_nnp.LowBeamReqSt =
    //       Sample->nnp_fct_out().light_signal_reqst().lowbeamreqst();
    //   data_->msg_ego_nnp.LowHighBeamReqSt =
    //       Sample->nnp_fct_out().light_signal_reqst().lowhighbeamreqst();
    //   data_->msg_ego_nnp.HazardLampReqSt =
    //       Sample->nnp_fct_out().light_signal_reqst().hazardlampreqst();
    //   data_->msg_ego_nnp.HornReqSt =
    //       Sample->nnp_fct_out().light_signal_reqst().hornreqst();

    //   // AVP

    //   data_->msg_ego_avp.m_iuss_state_obs =
    //       Sample->avp_fct_out().iuss_state_obs();
    //   data_->msg_ego_avp.need_replan_stop =
    //       Sample->avp_fct_out().need_replan_stop();
    //   data_->msg_ego_avp.plan_trigger =
    //       Sample->avp_fct_out().plan_trigger();
    //   data_->msg_ego_avp.control_enable =
    //       Sample->avp_fct_out().control_enable();
    //   data_->msg_ego_avp.parking_status =
    //       Sample->avp_fct_out().parking_status();

    //     if (skeleton_) {
    //         skeleton_->AlgEgoToMCU.Send(*data_);
    //     } else {
    //         INTRA_LOG_ERROR << "skeleton  is nullptr...";
    //     }
}
void EgoToMcuClientActivity::Stop() {
    INTRA_LOG_INFO << "begin...";
    egomcu_event_proxy_.Deinit();
    INTRA_LOG_INFO << "end...";
}

}  // namespace intra
}  // namespace netaos
}  // namespace hozon
