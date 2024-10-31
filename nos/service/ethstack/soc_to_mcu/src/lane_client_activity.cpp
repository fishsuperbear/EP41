/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: lanes_client_activity
 */

#include "lanes_client_activity.h"
using namespace std;

namespace hozon {
namespace netaos {
namespace intra {

LanesClientActivity::LanesClientActivity() {}

LanesClientActivity::~LanesClientActivity() {
    // Stop();
}

void LanesClientActivity::Init(std::shared_ptr<Skeleton> skeleton, std::string drivemode) {
    skeleton_ = skeleton;
    seq = 0;
    std::string mode = nnplanetopic;
    if (drivemode == "2") {
        mode = hpplanetopic;
    }
    int32_t ret = reader.Init(0, mode.c_str(), std::bind(&LanesClientActivity::cb, this, std::placeholders::_1));
    INTRA_LOG_INFO << "end..." << ret;
}
void LanesClientActivity::cb(std::shared_ptr<hozon::perception::TransportElement> msg) {
    // std::string text;
    // bool res = google::protobuf::TextFormat::PrintToString(*msg, &text);
    // INTRA_LOG_INFO << "hozon::perception::TransportElement "
    //                << " res " << res << " text: " << text.c_str();
    sendLanesData(msg);
    seq++;
}
HafLaneDetectionOut_A LanesClientActivity::LaneMarker2StructPb(const hozon::perception::LaneInfo& ptr_lane, double camera_stamp) {
    HafLaneDetectionOut_A lane_detect_outs;
    ::hozon::netaos::HafLaneDetectionOut alg_lane;
    alg_lane.timeCreation.sec = static_cast<uint64_t>(camera_stamp) * 1e9 / 1e9;
    alg_lane.timeCreation.nsec = static_cast<uint64_t>(camera_stamp * 1e9) - alg_lane.timeCreation.sec * 1e9;
    alg_lane.lanelineSeq = ptr_lane.track_id();
    alg_lane.geometryConfidence = ptr_lane.confidence();
    alg_lane.cls = ptr_lane.lanetype();
    alg_lane.laneLineWidth = 0;
    alg_lane.keyPointVRF.x = ptr_lane.points(0).x();
    alg_lane.keyPointVRF.y = ptr_lane.points(0).y();
    alg_lane.keyPointVRF.z = ptr_lane.points(0).z();
    alg_lane.laneFit.coefficients.a = ptr_lane.lane_param().cubic_curve_set(0).c0();
    alg_lane.laneFit.coefficients.b = ptr_lane.lane_param().cubic_curve_set(0).c1();
    alg_lane.laneFit.coefficients.d = ptr_lane.lane_param().cubic_curve_set(0).c2();
    alg_lane.laneFit.coefficients.c = ptr_lane.lane_param().cubic_curve_set(0).c3();
    alg_lane.laneFit.coefficients.dev_a = 0;
    alg_lane.laneFit.coefficients.dev_b = 0;
    alg_lane.laneFit.coefficients.dev_c = 0;
    alg_lane.laneFit.coefficients.dev_d = 0;
    alg_lane.laneFit.xStartVRF = ptr_lane.lane_param().cubic_curve_set(0).start_point_x();
    alg_lane.laneFit.xEndVRF = ptr_lane.lane_param().cubic_curve_set(0).end_point_x();
    lane_detect_outs[0] = alg_lane;
    return lane_detect_outs;
}

void LanesClientActivity::sendLanesData(const std::shared_ptr<hozon::perception::TransportElement> Sample) {
    std::shared_ptr<::hozon::netaos::HafLaneDetectionOutArray> data_ = std::make_shared<::hozon::netaos::HafLaneDetectionOutArray>();
    if (data_ == nullptr) {
        INTRA_LOG_ERROR << "SnsrFsnLaneDate dataSkeleton->hozonEvent.Allocate() got nullptr!";
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
    for (int32_t i = 0; i < Sample->lane_size(); i++) {
        data_->laneDetectionFrontOut[i] = LaneMarker2StructPb(Sample->lane(i), Sample->mutable_header()->sensor_stamp().camera_stamp());
    }
    if (seq % 100 == 0) {
        INTRA_LOG_INFO << "sendLanesData::  "
                       << " seq: " << data_->header.seq;
    }

    if (skeleton_) {
        skeleton_->SnsrFsnLaneDate.Send(*data_);
    } else {
        INTRA_LOG_ERROR << "skeleton  is nullptr...";
    }
}  // namespace soc_mcu

void LanesClientActivity::Stop() {
    INTRA_LOG_INFO << "begin...";
    reader.Deinit();
    INTRA_LOG_INFO << "end...";
}

}  // namespace intra
}  // namespace netaos
}  // namespace hozon
