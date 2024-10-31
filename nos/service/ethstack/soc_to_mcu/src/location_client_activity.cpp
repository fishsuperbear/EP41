/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: location_client_activity
 */
#include "location_client_activity.h"

// #define ACCEPT_USE_OF_DEPRECATED_PROJ_API_H

// #include <proj_api.h>

using namespace std;

namespace hozon {
namespace netaos {
namespace intra {

// const char *WGS84_TEXT = "+proj=longlat +datum=WGS84 +no_defs";

// const char *proj4_text = "+proj=utm +zone=51 +datum=WGS84 +units=m +no_defs";

LocationClientActivity::LocationClientActivity() {}

LocationClientActivity::~LocationClientActivity() {
    // Stop();
}

void LocationClientActivity::Init(std::shared_ptr<Skeleton> skeleton, std::string drivemode) {
    skeleton_ = skeleton;
    seq = 0;
    std::string mode = nnplocationtopic;
    if (drivemode == "2") {
        mode = hpplocationtopic;
    }
    int32_t ret = reader.Init(0, mode.c_str(), std::bind(&LocationClientActivity::cb, this, std::placeholders::_1));
    INTRA_LOG_INFO << "end..." << ret;
}
void LocationClientActivity::cb(std::shared_ptr<hozon::localization::Localization> msg) {
    // std::string text;
    // bool res = google::protobuf::TextFormat::PrintToString(*msg, &text);
    // INTRA_LOG_INFO << "hozon::perception::Localization "
    //                << " res " << res << " text: " << text.c_str();
    sendLocationData(msg);
    seq++;
}

void LocationClientActivity::sendLocationData(const std::shared_ptr<hozon::localization::Localization> Sample) {
    std::shared_ptr<::hozon::netaos::HafLocation> data_ = std::make_shared<::hozon::netaos::HafLocation>();
    if (data_ == nullptr) {
        INTRA_LOG_ERROR << "PoseData dataSkeleton->hozonEvent.Allocate() got nullptr!";
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

    std::string frameid = Sample->mutable_header()->frame_id();
    frameid = frameid.substr(0, stringSize);
    memset(data_->header.frameId.data(), 0, stringSize);
    memcpy(data_->header.frameId.data(), frameid.data(), frameid.size());
    // data_->gpsWeek = static_cast<uint64_t>(Sample->measurement_time()) * 1e9 / 1e9 / (7 * 24 * 3600);
    // data_->gpsSec = Sample->measurement_time() - data_->gpsWeek * (7 * 24 * 3600);
    data_->gpsWeek = Sample->gps_week();
    data_->gpsSec = Sample->gps_sec();
    data_->rtkStatus = Sample->rtk_status();
    // data_->coordinateType = Sample->mutable_msf_status()->gnsspos_position_type();
    data_->locationState = Sample->location_state();
    data_->pose.poseWGS.position.x = Sample->mutable_pose()->mutable_wgs()->x();
    data_->pose.poseWGS.position.y = Sample->mutable_pose()->mutable_wgs()->y();
    data_->pose.poseWGS.position.z = Sample->mutable_pose()->mutable_wgs()->z();
    data_->pose.poseWGS.rotationVRF.x = 0;
    data_->pose.poseWGS.rotationVRF.y = 0;
    data_->pose.poseWGS.rotationVRF.z = 0;
    // data_->pose.poseLOCAL.position.x = Sample->mutable_pose()->mutable_position()->x();
    // data_->pose.poseLOCAL.position.y = Sample->mutable_pose()->mutable_position()->y();
    // data_->pose.poseLOCAL.position.z = Sample->mutable_pose()->mutable_position()->z();

    /*
        data_->pose.poseLOCAL.position.x = Sample->mutable_pose()->mutable_pose_local()->x();
        data_->pose.poseLOCAL.position.y = Sample->mutable_pose()->mutable_pose_local()->y();
        data_->pose.poseLOCAL.position.z = Sample->mutable_pose()->mutable_pose_local()->z();
        data_->pose.poseLOCAL.quaternion.x = Sample->mutable_pose()->mutable_quaternion()->x();
        data_->pose.poseLOCAL.quaternion.y = Sample->mutable_pose()->mutable_quaternion()->y();
        data_->pose.poseLOCAL.quaternion.z = Sample->mutable_pose()->mutable_quaternion()->z();
        data_->pose.poseLOCAL.quaternion.w = Sample->mutable_pose()->mutable_quaternion()->w();
        data_->pose.poseLOCAL.eulerAngle.x = Sample->mutable_pose()->mutable_euler_angle()->x();
        data_->pose.poseLOCAL.eulerAngle.y = Sample->mutable_pose()->mutable_euler_angle()->y();
        data_->pose.poseLOCAL.eulerAngle.z = Sample->mutable_pose()->mutable_euler_angle()->z();
        data_->pose.poseLOCAL.rotationVRF.x = Sample->mutable_pose()->mutable_rotation_vrf()->x();
        data_->pose.poseLOCAL.rotationVRF.y = Sample->mutable_pose()->mutable_rotation_vrf()->y();
        data_->pose.poseLOCAL.rotationVRF.z = Sample->mutable_pose()->mutable_rotation_vrf()->z();
    */
    data_->pose.poseLOCAL.position.x = Sample->pose_local().position().x();
    data_->pose.poseLOCAL.position.y = Sample->pose_local().position().y();
    data_->pose.poseLOCAL.position.z = Sample->pose_local().position().z();
    data_->pose.poseLOCAL.quaternion.x = Sample->pose_local().quaternion().x();
    data_->pose.poseLOCAL.quaternion.y = Sample->pose_local().quaternion().y();
    data_->pose.poseLOCAL.quaternion.z = Sample->pose_local().quaternion().z();
    data_->pose.poseLOCAL.quaternion.w = Sample->pose_local().quaternion().w();
    data_->pose.poseLOCAL.eulerAngle.x = Sample->pose_local().euler_angle().x();
    data_->pose.poseLOCAL.eulerAngle.y = Sample->pose_local().euler_angle().y();
    data_->pose.poseLOCAL.eulerAngle.z = Sample->pose_local().euler_angle().z();
    data_->pose.poseLOCAL.rotationVRF.x = Sample->pose_local().rotation_vrf().x();
    data_->pose.poseLOCAL.rotationVRF.y = Sample->pose_local().rotation_vrf().y();
    data_->pose.poseLOCAL.rotationVRF.z = Sample->pose_local().rotation_vrf().z();
    data_->pose.poseLOCAL.heading = Sample->pose_local().heading();

    data_->pose.poseGCJ02.position.x = Sample->mutable_pose()->mutable_gcj02()->x();
    data_->pose.poseGCJ02.position.y = Sample->mutable_pose()->mutable_gcj02()->y();
    data_->pose.poseGCJ02.position.z = Sample->mutable_pose()->mutable_gcj02()->z();
    data_->pose.poseGCJ02.quaternion.x = Sample->mutable_pose()->mutable_quaternion()->x();
    data_->pose.poseGCJ02.quaternion.y = Sample->mutable_pose()->mutable_quaternion()->y();
    data_->pose.poseGCJ02.quaternion.z = Sample->mutable_pose()->mutable_quaternion()->z();
    data_->pose.poseGCJ02.quaternion.w = Sample->mutable_pose()->mutable_quaternion()->w();
    data_->pose.poseGCJ02.eulerAngle.x = Sample->mutable_pose()->mutable_euler_angle()->x();
    data_->pose.poseGCJ02.eulerAngle.y = Sample->mutable_pose()->mutable_euler_angle()->y();
    data_->pose.poseGCJ02.eulerAngle.z = Sample->mutable_pose()->mutable_euler_angle()->z();
    data_->pose.poseGCJ02.rotationVRF.x = Sample->mutable_pose()->mutable_rotation_vrf()->x();
    data_->pose.poseGCJ02.rotationVRF.y = Sample->mutable_pose()->mutable_rotation_vrf()->y();
    data_->pose.poseGCJ02.rotationVRF.z = Sample->mutable_pose()->mutable_rotation_vrf()->z();
    data_->pose.poseUTM01.position.x = Sample->mutable_pose()->mutable_pos_utm_01()->x();
    data_->pose.poseUTM01.position.y = Sample->mutable_pose()->mutable_pos_utm_01()->y();
    data_->pose.poseUTM01.position.z = Sample->mutable_pose()->mutable_pos_utm_01()->z();
    data_->pose.poseUTM02.position.x = Sample->mutable_pose()->mutable_pos_utm_02()->x();
    data_->pose.poseUTM02.position.y = Sample->mutable_pose()->mutable_pos_utm_02()->y();
    data_->pose.poseUTM02.position.z = Sample->mutable_pose()->mutable_pos_utm_02()->z();
    data_->pose.poseUTM02.rotationVRF.x = 0;
    data_->pose.poseUTM02.rotationVRF.y = 0;
    data_->pose.poseUTM02.rotationVRF.z = 0;
    double heading = (Sample->mutable_pose()->heading());
    data_->pose.poseUTM01.heading = heading;
    data_->pose.poseUTM02.heading = heading;
    // static constexpr double kRAD2DEG = 180 / M_PI;
    // if (Sample->mutable_pose()->has_heading_gcs()) {
    //   heading = -(Sample->mutable_pose()->heading_gcs() - M_PI / 2) * kRAD2DEG;
    // }
    // data_->pose.poseWGS.heading = heading;
    // data_->pose.poseGCJ02.heading = heading;
    /*
    data_->velocity.twistVRF.linearVRF.x = Sample->mutable_pose()->mutable_linear_velocity_vrf()->x();
    data_->velocity.twistVRF.linearVRF.y = Sample->mutable_pose()->mutable_linear_velocity_vrf()->y();
    data_->velocity.twistVRF.linearVRF.z = Sample->mutable_pose()->mutable_linear_velocity_vrf()->z();
    data_->velocity.twistVRF.angularVRF.x = Sample->mutable_pose()->mutable_angular_velocity_vrf()->x();
    data_->velocity.twistVRF.angularVRF.y = Sample->mutable_pose()->mutable_angular_velocity_vrf()->y();
    data_->velocity.twistVRF.angularVRF.z = Sample->mutable_pose()->mutable_angular_velocity_vrf()->z();
    data_->acceleration.linearVRF.linearVRF.x = Sample->mutable_pose()->mutable_linear_acceleration_vrf()->x();
    data_->acceleration.linearVRF.linearVRF.y = Sample->mutable_pose()->mutable_linear_acceleration_vrf()->y();
    data_->acceleration.linearVRF.linearVRF.z = Sample->mutable_pose()->mutable_linear_acceleration_vrf()->z();
    data_->acceleration.linearVRF.angularVRF.x = 0;
    data_->acceleration.linearVRF.angularVRF.y = 0;
    data_->acceleration.linearVRF.angularVRF.z = 0;
    */
    data_->velocity.twistVRF.linearVRF.x = Sample->mutable_pose_local()->mutable_linear_velocity_vrf()->x();
    data_->velocity.twistVRF.linearVRF.y = Sample->mutable_pose_local()->mutable_linear_velocity_vrf()->y();
    data_->velocity.twistVRF.linearVRF.z = Sample->mutable_pose_local()->mutable_linear_velocity_vrf()->z();
    data_->velocity.twistVRF.angularVRF.x = Sample->mutable_pose_local()->mutable_angular_velocity_vrf()->x();
    data_->velocity.twistVRF.angularVRF.y = Sample->mutable_pose_local()->mutable_angular_velocity_vrf()->y();
    data_->velocity.twistVRF.angularVRF.z = Sample->mutable_pose_local()->mutable_angular_velocity_vrf()->z();
    data_->acceleration.linearVRF.linearVRF.x = Sample->mutable_pose_local()->mutable_linear_acceleration_vrf()->x();
    data_->acceleration.linearVRF.linearVRF.y = Sample->mutable_pose_local()->mutable_linear_acceleration_vrf()->y();
    data_->acceleration.linearVRF.linearVRF.z = Sample->mutable_pose_local()->mutable_linear_acceleration_vrf()->z();
    data_->acceleration.linearVRF.angularVRF.x = 0;
    data_->acceleration.linearVRF.angularVRF.y = 0;
    data_->acceleration.linearVRF.angularVRF.z = 0;
    //   data_->velocity.twistVRFVRF.linearVRF.x =
    //       Sample->mutable_pose().linear_velocity_vrf().y();
    //   data_->velocity.twistVRFVRF.linearVRF.y =
    //       -Sample->mutable_pose().linear_velocity_vrf().x();
    //   data_->velocity.twistVRFVRF.linearVRF.z =
    //       Sample->mutable_pose().linear_velocity_vrf().z();
    //   static constexpr double kG = 9.80665;
    //   data_->velocity.twistVRFVRF.angularVRF.x =
    //       Sample->mutable_pose().angular_velocity_vrf().y() * kRAD2DEG;
    //   data_->velocity.twistVRFVRF.angularVRF.y =
    //       -Sample->mutable_pose().angular_velocity_vrf().x() * kRAD2DEG;
    //   data_->velocity.twistVRFVRF.angularVRF.z =
    //       Sample->mutable_pose().angular_velocity_vrf().z() * kRAD2DEG;
    //   data_->velocity.twistVRFVRF.angularRawVRF.x =
    //       Sample->mutable_pose().angular_velocity_vrf().y() * kRAD2DEG;
    //   data_->velocity.twistVRFVRF.angularRawVRF.y =
    //       -Sample->mutable_pose().angular_velocity_vrf().x() * kRAD2DEG;
    //   data_->velocity.twistVRFVRF.angularRawVRF.z =
    //       Sample->mutable_pose().angular_velocity_vrf().z() * kRAD2DEG;
    //   data_->acceleration.linearVRF.linearVRF.x =
    //       Sample->mutable_pose().linear_acceleration_vrf().y() / kG;
    //   data_->acceleration.linearVRF.linearVRF.y =
    //       -Sample->mutable_pose().linear_acceleration_vrf().x() / kG;
    //   data_->acceleration.linearVRF.linearVRF.z =
    //       -Sample->mutable_pose().linear_acceleration_vrf().z() / kG;
    //   data_->acceleration.linearVRF.linearRawVRF.x =
    //       Sample->mutable_pose().linear_acceleration_vrf().y() / kG;
    //   data_->acceleration.linearVRF.linearRawVRF.y =
    //       -Sample->mutable_pose().linear_acceleration_vrf().x() / kG;
    //   data_->acceleration.linearVRF.linearRawVRF.z =
    //       -Sample->mutable_pose().linear_acceleration_vrf().z() / kG;
    data_->pose.utmZoneID01 = Sample->mutable_pose()->utm_zone_01();
    data_->pose.utmZoneID02 = Sample->mutable_pose()->utm_zone_02();
    if (seq % 100 == 0) {
        INTRA_LOG_INFO << "sendLocationData::  "
                       << " seq: " << data_->header.seq;
    }
    if (skeleton_) {
        skeleton_->PoseData.Send(*data_);
    } else {
        INTRA_LOG_ERROR << "skeleton  is nullptr...";
    }
}

void LocationClientActivity::Stop() {
    INTRA_LOG_INFO << "begin...";
    reader.Deinit();
    INTRA_LOG_INFO << "end...";
}

}  // namespace intra
}  // namespace netaos
}  // namespace hozon
