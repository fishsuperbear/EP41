#pragma once
#include "hozon/object/impl_type_objectcameraframe.h"
#include "proto/perception/perception_obstacle.pb.h"

static hozon::perception::BrakeLightStatus converBrakeLightStatus(UInt8 mdc_state) {
    hozon::perception::BrakeLightStatus proto_sate;

    switch (mdc_state) {
        case 0:
            proto_sate = hozon::perception::BrakeLightStatus::BLST_UNKNOWN;
            break;
        case 1:
            proto_sate = hozon::perception::BrakeLightStatus::BLST_OFF;
            break;
        case 2:
            proto_sate = hozon::perception::BrakeLightStatus::BLST_ON;
            break;
        default:
            proto_sate = hozon::perception::BrakeLightStatus::BLST_UNKNOWN;
            break;
    }
    return proto_sate;
}

static hozon::perception::TurnLightStatus converTurnLightStatus(UInt8 mdc_state) {
    hozon::perception::TurnLightStatus proto_sate;
    switch (mdc_state) {
        case 0:
            proto_sate = hozon::perception::TurnLightStatus::TLST_UNKNOWN;
            break;
        case 1:
            proto_sate = hozon::perception::TurnLightStatus::TLST_ALL_OFF;
            break;
        case 2:
            proto_sate = hozon::perception::TurnLightStatus::TLST_LEFT_FLASH;
            break;
        case 3:
            proto_sate = hozon::perception::TurnLightStatus::TLST_RIGHT_FLASH;
            break;
        case 4:
            proto_sate = hozon::perception::TurnLightStatus::TLST_LEFT_RIGHT_FLASH;
            break;
        default:
            proto_sate = hozon::perception::TurnLightStatus::TLST_UNKNOWN;
            break;
    }
    return proto_sate;
}

hozon::perception::PerceptionObstacles ObjCameraFrameToObstacleOut(hozon::object::ObjectCameraFrame mdc_data) {
    hozon::perception::PerceptionObstacles proto_data;

    // PerceptionObstacle perception_obstacle
    for (auto obj_cam_3d : mdc_data.detectedOut3d) {
        hozon::perception::PerceptionObstacle* pept_obj = proto_data.add_perception_obstacle();
        pept_obj->set_id(obj_cam_3d.objectID);

        // hozon.common.Point3D velocity
        pept_obj->mutable_velocity()->set_x(static_cast<double>(obj_cam_3d.velocity.x));
        pept_obj->mutable_velocity()->set_y(static_cast<double>(obj_cam_3d.velocity.y));
        pept_obj->mutable_velocity()->set_z(static_cast<double>(obj_cam_3d.velocity.z));

        pept_obj->set_type(static_cast<hozon::perception::PerceptionObstacle::Type>(obj_cam_3d.type));
        pept_obj->set_type_confidence(static_cast<double>(obj_cam_3d.confidence));

        // hozon.common.Point3D acceleration
        pept_obj->mutable_acceleration()->set_x(static_cast<double>(obj_cam_3d.accel.x));
        pept_obj->mutable_acceleration()->set_y(static_cast<double>(obj_cam_3d.accel.y));
        pept_obj->mutable_acceleration()->set_z(static_cast<double>(obj_cam_3d.accel.z));

        pept_obj->set_sub_type(static_cast<hozon::perception::PerceptionObstacle::SubType>(obj_cam_3d.sub_type));

        // double velocity_covariance
        for (auto vel : obj_cam_3d.velocity_unc) {
            pept_obj->add_velocity_covariance(static_cast<double>(vel));
        }

        // double acceleration_covariance
        for (auto accel : obj_cam_3d.accel_unc) {
            pept_obj->add_acceleration_covariance(static_cast<double>(accel));
        }

        // LightStatus light_status
        pept_obj->mutable_light_status()->set_brake_visible(converBrakeLightStatus(obj_cam_3d.brakeLightSt));
        pept_obj->mutable_light_status()->set_turn_light(converTurnLightStatus(obj_cam_3d.turnLightSt));

        // MotionType motion_type
        pept_obj->set_motion_type(static_cast<hozon::perception::PerceptionObstacle::MotionType>(obj_cam_3d.movState));

        pept_obj->set_existence_probability(static_cast<double>(obj_cam_3d.existConfidence));

        // DetectSensor current_detect_senso
        for (int i = 0; i < 18; i++) {
            if (0x01 & (obj_cam_3d.detCamSensor >> i)) {
                switch (i) {
                    case 0:
                        pept_obj->mutable_current_detect_sensor()->set_from_camera_front_long_range(true);
                        break;
                    case 1:
                        pept_obj->mutable_current_detect_sensor()->set_from_camera_front_wide_angle(true);
                        break;
                    case 2:
                        pept_obj->mutable_current_detect_sensor()->set_from_camera_right_forward_looking(true);
                        break;
                    case 3:
                        pept_obj->mutable_current_detect_sensor()->set_from_camera_right_backward_looking(true);
                        break;
                    case 4:
                        pept_obj->mutable_current_detect_sensor()->set_from_camera_left_forward_looking(true);
                        break;
                    case 5:
                        pept_obj->mutable_current_detect_sensor()->set_from_camera_left_backward_looking(true);
                        break;
                    case 6:
                        pept_obj->mutable_current_detect_sensor()->set_from_camera_rear(true);
                        break;
                    case 7:
                        break;
                    case 8:
                        break;
                    case 9:
                        break;
                    case 10:
                        break;
                    case 11:
                        break;
                    case 12:
                        break;
                    case 13:
                        break;
                    case 14:
                        pept_obj->mutable_current_detect_sensor()->set_from_camera_fisheye_front(true);
                        break;
                    case 15:
                        pept_obj->mutable_current_detect_sensor()->set_from_camera_fisheye_rear(true);
                        break;
                    case 16:
                        pept_obj->mutable_current_detect_sensor()->set_from_camera_fisheye_left(true);
                        break;
                    case 17:
                        pept_obj->mutable_current_detect_sensor()->set_from_camera_fisheye_right(true);
                        break;

                    default:
                        break;
                }
            }
        }

        pept_obj->set_track_age(obj_cam_3d.age);

        // hozon.common.Point3D center
        pept_obj->mutable_center()->set_x(obj_cam_3d.rect.Center.x);
        pept_obj->mutable_center()->set_y(obj_cam_3d.rect.Center.y);
        pept_obj->mutable_center()->set_z(obj_cam_3d.rect.Center.z);

        pept_obj->set_is_truncation(static_cast<bool>(obj_cam_3d.isTruncated));
        pept_obj->set_truncated_prob(static_cast<double>(obj_cam_3d.truncatedProb));
        pept_obj->set_is_occlusion(static_cast<bool>(obj_cam_3d.isOccluded));
        pept_obj->set_occluded_prob(static_cast<double>(obj_cam_3d.occludedProb));

        pept_obj->set_is_onroad(static_cast<bool>(obj_cam_3d.isOnroad));
        pept_obj->set_onroad_prob(static_cast<double>(obj_cam_3d.onRoadProb));
        pept_obj->set_is_sprinkler(static_cast<bool>(obj_cam_3d.isSprinkler));
        pept_obj->set_sprinkler_prob(static_cast<double>(obj_cam_3d.sprinklerProb));
    }

    // hozon.common.Header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);

    // SensorStatus sensor_status
    proto_data.mutable_sensor_status()->set_camera_status(static_cast<hozon::perception::SensorStatus::CameraStatus>(mdc_data.sensorStatus));

    // proto_data.set_coord_state(static_cast<hozon::perception::CoordinateState>(mdc_data.lightIntensity));

    return proto_data;
}