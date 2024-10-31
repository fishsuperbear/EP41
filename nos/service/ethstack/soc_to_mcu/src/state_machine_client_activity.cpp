/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-09-22 09:41:32
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-11-24 11:24:11
 * @FilePath: /nos/service/ethstack/soc_to_mcu/src/state_machine_client_activity.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: state_machine_client_activity
 */

#include "state_machine_client_activity.h"
using namespace std;

namespace hozon {
namespace netaos {
namespace intra {

StateMachineClientActivity::StateMachineClientActivity() {}

StateMachineClientActivity::~StateMachineClientActivity() {
    // Stop();
}

void StateMachineClientActivity::Init(std::shared_ptr<Skeleton> skeleton, std::string drivemode) {
    seq = 0;
    skeleton_ = skeleton;
    int32_t ret = reader.Init(0, statemachinetopic.c_str(), std::bind(&StateMachineClientActivity::cb, this, std::placeholders::_1));
    INTRA_LOG_INFO << "end..." << ret;
}
void StateMachineClientActivity::cb(std::shared_ptr<hozon::state::StateMachine> msg) {
    // std::string text;
    // bool res = google::protobuf::TextFormat::PrintToString(*msg, &text);
    // INTRA_LOG_INFO << "hozon::state::StateMachine "
    //                << " res " << res << " text: " << text.c_str();
    sendStateMachineData(msg);
    seq++;
}

void StateMachineClientActivity::sendStateMachineData(const std::shared_ptr<hozon::state::StateMachine> Sample) {
    std::shared_ptr<::hozon::netaos::APAStateMachineFrame> data_ = std::make_shared<::hozon::netaos::APAStateMachineFrame>();
    if (data_ == nullptr) {
        INTRA_LOG_ERROR << "ApaStateMachine dataSkeleton->hozonEvent.Allocate() got nullptr!";
        return;
    }
    // data_->header.seq = Sample->mutable_header()->seq();
    // // data_->header.stamp.sec = static_cast<uint64_t>(Sample->mutable_header()->publish_stamp()) * 1e9 / 1e9;
    // // data_->header.stamp.nsec = static_cast<uint64_t>(Sample->mutable_header()->publish_stamp() * 1e9) - data_->header.stamp.sec * 1e9;
    // // data_->header.gnssStamp.sec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp()) * 1e9 / 1e9;
    // // data_->header.gnssStamp.nsec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp() * 1e9) - data_->header.stamp.sec * 1e9;
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
    // std::string frameid = Sample->mutable_header()->frame_id();
    // frameid = frameid.substr(0, stringSize);
    // memset(data_->header.frameId.data(), 0, stringSize);
    // memcpy(data_->header.frameId.data(), frameid.data(), frameid.size());
    data_->pilot_status.processing_status = Sample->mutable_pilot_status()->processing_status();
    data_->pilot_status.turn_light_status = Sample->mutable_pilot_status()->turn_light_status();
    data_->pilot_status.localization_status = Sample->mutable_pilot_status()->localization_status();
    data_->pilot_status.camera_status = Sample->mutable_pilot_status()->camera_status();
    data_->pilot_status.uss_status = Sample->mutable_pilot_status()->uss_status();
    data_->pilot_status.radar_status = Sample->mutable_pilot_status()->radar_status();
    data_->pilot_status.lidar_status = Sample->mutable_pilot_status()->lidar_status();
    data_->pilot_status.velocity_status = Sample->mutable_pilot_status()->velocity_status();
    data_->pilot_status.perception_status = Sample->mutable_pilot_status()->perception_status();
    data_->pilot_status.planning_status = Sample->mutable_pilot_status()->planning_status();
    data_->pilot_status.controlling_status = Sample->mutable_pilot_status()->controlling_status();
    data_->hpp_command.enable_parking_slot_detection = Sample->mutable_hpp_command()->enable_parking_slot_detection();
    data_->hpp_command.reserved1 = Sample->mutable_hpp_command()->reserved1();
    data_->hpp_command.reserved2 = Sample->mutable_hpp_command()->reserved2();
    data_->hpp_command.reserved3 = Sample->mutable_hpp_command()->reserved3();
    data_->hpp_command.enable_object_detection = Sample->mutable_hpp_command()->enable_object_detection();
    data_->hpp_command.enable_freespace_detection = Sample->mutable_hpp_command()->enable_freespace_detection();
    data_->hpp_command.enable_uss = Sample->mutable_hpp_command()->enable_uss();
    data_->hpp_command.enable_radar = Sample->mutable_hpp_command()->enable_radar();
    data_->hpp_command.enable_lidar = Sample->mutable_hpp_command()->enable_lidar();
    data_->hpp_command.system_command = Sample->mutable_hpp_command()->system_command();
    data_->hpp_command.emergencybrake_state = Sample->mutable_hpp_command()->emergencybrake_state();
    data_->hpp_command.system_reset = Sample->mutable_hpp_command()->system_reset();
    if (seq % 100 == 0) {
        INTRA_LOG_INFO << "sendStateMachineData::  ";
    }
    if (skeleton_) {
        skeleton_->ApaStateMachine.Send(*data_);
    } else {
        INTRA_LOG_ERROR << "skeleton  is nullptr...";
    }
}  // namespace soc_mcu

void StateMachineClientActivity::Stop() {
    INTRA_LOG_INFO << "begin...";
    reader.Deinit();
    INTRA_LOG_INFO << "end...";
}

}  // namespace intra
}  // namespace netaos
}  // namespace hozon
