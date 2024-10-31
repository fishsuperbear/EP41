/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: object_client_activity
 */

#include "object_client_activity.h"

using namespace std;

namespace hozon {
namespace netaos {
namespace intra {

ObjectClientActivity::ObjectClientActivity() {}
ObjectClientActivity::~ObjectClientActivity() {
    // Stop();
}

void ObjectClientActivity::Init(std::shared_ptr<Skeleton> skeleton, std::string drivemode) {
    skeleton_ = skeleton;
    seq = 0;
    std::string mode = nnpobjecttopic;
    if (drivemode == "2") {
        mode = hppobjecttopic;
    }
    int32_t ret = reader.Init(0, mode.c_str(), std::bind(&ObjectClientActivity::cb, this, std::placeholders::_1));
    INTRA_LOG_INFO << "end..." << ret;
}
void ObjectClientActivity::cb(std::shared_ptr<hozon::perception::PerceptionObstacles> msg) {
    // std::string text;
    // bool res = google::protobuf::TextFormat::PrintToString(*msg, &text);
    // INTRA_LOG_INFO << "hozon::perception::PerceptionObstacles : "
    //                << " res " << res << " text: " << text.c_str();
    sendObjectData(msg);
    seq++;
}

void ObjectClientActivity::sendObjectData(const std::shared_ptr<hozon::perception::PerceptionObstacles> Sample) {
    std::shared_ptr<::hozon::netaos::HafFusionOutArray> data_ = std::make_shared<::hozon::netaos::HafFusionOutArray>();
    if (data_ == nullptr) {
        INTRA_LOG_ERROR << "SnsrFsnObj dataSkeleton->hozonEvent.Allocate() got nullptr!";
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
    data_->locSeq = 0;
    for (int32_t i = 0; i < Sample->perception_obstacle_size(); i++) {
        if (i == fusionOutLength) {
            break;
        }
        hozon::perception::PerceptionObstacle fur_obj;
        ::hozon::netaos::HafFusionOut obj_fus_3d;
        fur_obj = Sample->perception_obstacle(i);
        obj_fus_3d.ObjectID = fur_obj.id();
        obj_fus_3d.VelocityAbs.x = fur_obj.mutable_velocity()->x();
        obj_fus_3d.VelocityAbs.y = fur_obj.mutable_velocity()->y();
        obj_fus_3d.Type = fur_obj.type();
        obj_fus_3d.TypeConfidence = fur_obj.type_confidence() * 100;
        obj_fus_3d.AccelerationAbs.x = fur_obj.mutable_acceleration()->x();
        obj_fus_3d.AccelerationAbs.y = fur_obj.mutable_acceleration()->y();
        // if (obj_fus_3d.ObjectID % 2 == 0) {
        //     obj_fus_3d.BrakeLightSt = 2;
        // } else {
        //     obj_fus_3d.BrakeLightSt = 1;
        // }
        // if (obj_fus_3d.ObjectID % 3 == 0) {
        //     obj_fus_3d.TurnLightSt = 2;
        // } else if (obj_fus_3d.ObjectID % 5 == 0) {
        //     obj_fus_3d.TurnLightSt = 3;
        // } else {
        //     obj_fus_3d.TurnLightSt = 0;
        // }
        //
        obj_fus_3d.BrakeLightSt = fur_obj.mutable_light_status()->brake_visible();
        if (fur_obj.mutable_light_status()->has_turn_light()) {
            obj_fus_3d.TurnLightSt = fur_obj.mutable_light_status()->turn_light();
        }

        obj_fus_3d.MotionPattern = fur_obj.motion_type();
        obj_fus_3d.MaintenanceStatus = fur_obj.maintenance_type();
        obj_fus_3d.ExistenceProbability = fur_obj.existence_probability() * 100;
        uint8_t index = 0;
        if (fur_obj.mutable_current_detect_sensor()->has_from_camera_front_long_range()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_camera_front_long_range();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_camera_front_wide_angle()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_camera_front_wide_angle();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_camera_right_forward_looking()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_camera_right_forward_looking();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_camera_right_backward_looking()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_camera_right_backward_looking();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_camera_left_forward_looking()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_camera_left_forward_looking();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_camera_left_backward_looking()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_camera_left_backward_looking();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_camera_rear()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_camera_rear();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_radar_front()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_radar_front();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_radar_front_right()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_radar_front_right();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_radar_front_left()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_radar_front_left();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_radar_rear_right()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_radar_rear_right();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_radar_rear_left()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_radar_rear_left();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_lidar_front_right()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_lidar_front_right();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_lidar_front_left()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_lidar_front_left();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_camera_fisheye_front()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_camera_fisheye_front();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_camera_fisheye_rear()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_camera_fisheye_rear();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_camera_fisheye_left()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_camera_fisheye_left();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_camera_fisheye_right()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_camera_fisheye_right();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_fol()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_fol();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_fcl()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_fcl();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_fcr()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_fcr();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_for()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_for();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_rol()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_rol();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_rcl()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_rcl();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_rcr()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_rcr();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_ror()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_ror();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_fsl()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_fsl();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_rsl()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_rsl();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_fsr()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_fsr();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_rsr()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_rsr();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index++;
        if (fur_obj.mutable_current_detect_sensor()->has_from_uss_fusion()) {
            uint8_t data = fur_obj.mutable_current_detect_sensor()->from_uss_fusion();
            obj_fus_3d.DetectSensor_Current += (data << (index));
        }
        index = 0;
        if (fur_obj.mutable_history_detect_sensor()->has_from_camera_front_long_range()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_camera_front_long_range();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_camera_front_wide_angle()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_camera_front_wide_angle();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_camera_right_forward_looking()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_camera_right_forward_looking();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_camera_right_backward_looking()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_camera_right_backward_looking();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_camera_left_forward_looking()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_camera_left_forward_looking();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_camera_left_backward_looking()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_camera_left_backward_looking();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_camera_rear()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_camera_rear();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_radar_front()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_radar_front();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_radar_front_right()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_radar_front_right();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_radar_front_left()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_radar_front_left();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_radar_rear_right()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_radar_rear_right();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_radar_rear_left()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_radar_rear_left();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_lidar_front_right()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_lidar_front_right();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_lidar_front_left()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_lidar_front_left();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_camera_fisheye_front()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_camera_fisheye_front();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_camera_fisheye_rear()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_camera_fisheye_rear();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_camera_fisheye_left()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_camera_fisheye_left();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_camera_fisheye_right()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_camera_fisheye_right();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_fol()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_fol();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_fcl()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_fcl();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_fcr()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_fcr();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_for()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_for();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_rol()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_rol();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_rcl()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_rcl();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_rcr()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_rcr();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_ror()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_ror();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_fsl()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_fsl();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_rsl()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_rsl();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_fsr()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_fsr();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_rsr()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_rsr();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        index++;
        if (fur_obj.mutable_history_detect_sensor()->has_from_uss_fusion()) {
            uint8_t data = fur_obj.mutable_history_detect_sensor()->from_uss_fusion();
            obj_fus_3d.DetectSensor_History += (data << (index));
        }
        obj_fus_3d.TimeCreation.sec = static_cast<uint64_t>(fur_obj.creation_time()) * 1e9 / 1e9;
        obj_fus_3d.TimeCreation.nsec = static_cast<uint64_t>(fur_obj.creation_time() * 1e9) - obj_fus_3d.TimeCreation.sec * 1e9;
        obj_fus_3d.LastUpdatedTime.sec = static_cast<uint64_t>(fur_obj.latest_tracked_time()) * 1e9 / 1e9;
        obj_fus_3d.LastUpdatedTime.nsec = static_cast<uint64_t>(fur_obj.latest_tracked_time() * 1e9) - obj_fus_3d.TimeCreation.sec * 1e9;
        obj_fus_3d.NearSide = fur_obj.car_near_side();
        obj_fus_3d.MotionPatternHistory = fur_obj.history_motion_type();
        obj_fus_3d.Age = fur_obj.track_age();
        obj_fus_3d.RectInfo.Center.x = fur_obj.mutable_center()->x();
        obj_fus_3d.RectInfo.Center.y = fur_obj.mutable_center()->y();
        obj_fus_3d.RectInfo.Center.z = fur_obj.mutable_center()->z();
        obj_fus_3d.RectInfo.CenterStdDev.x = 0;
        obj_fus_3d.RectInfo.CenterStdDev.y = 0;
        obj_fus_3d.RectInfo.CenterStdDev.z = 0;
        obj_fus_3d.RectInfo.SizeLWH.x = fur_obj.length();
        obj_fus_3d.RectInfo.SizeLWH.y = fur_obj.width();
        obj_fus_3d.RectInfo.SizeLWH.z = fur_obj.height();
        obj_fus_3d.RectInfo.SizeStdDev.x = 0;
        obj_fus_3d.RectInfo.SizeStdDev.y = 0;
        obj_fus_3d.RectInfo.SizeStdDev.z = 0;
        obj_fus_3d.RectInfo.Orientation = fur_obj.theta() - fur_obj.theta_flu();
        obj_fus_3d.RectInfo.OrientationStdDev = 0;
        // obj_fus_3d.rect.Orientation = fur_obj.theta() - fur_obj.heading();
        data_->fusionOut[i] = obj_fus_3d;
    }
    if (seq % 100 == 0) {
        INTRA_LOG_INFO << "sendObjectData::  "
                       << " seq: " << data_->header.seq;
    }
    if (skeleton_) {
        skeleton_->SnsrFsnObj.Send(*data_);
    } else {
        INTRA_LOG_ERROR << "skeleton  is nullptr...";
    }
}

void ObjectClientActivity::Stop() {
    INTRA_LOG_INFO << "begin...";
    reader.Deinit();
    INTRA_LOG_INFO << "end...";
}

}  // namespace intra
}  // namespace netaos
}  // namespace hozon
