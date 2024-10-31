#pragma once
#include "hozon/object/impl_type_objectfusionframe.h"
#include "proto/perception/perception_obstacle.pb.h"

hozon::perception::PerceptionObstacles ObjFusionFrameToObstacleOut(hozon::object::ObjectFusionFrame mdc_data) {
    hozon::perception::PerceptionObstacles proto_data;

    // PerceptionObstacle perception_obstacle
    for (auto obj_fus_3d : mdc_data.object_fusion) {
        hozon::perception::PerceptionObstacle* fur_obj = proto_data.add_perception_obstacle();
        fur_obj->set_id(obj_fus_3d.objectID);

        // hozon.common.Point3D velocity
        fur_obj->mutable_velocity()->set_x(static_cast<double>(obj_fus_3d.velocity.x));
        fur_obj->mutable_velocity()->set_y(static_cast<double>(obj_fus_3d.velocity.y));
        fur_obj->mutable_velocity()->set_z(static_cast<double>(obj_fus_3d.velocity.z));

        fur_obj->set_type(static_cast<hozon::perception::PerceptionObstacle::Type>(obj_fus_3d.type));
        fur_obj->set_type_confidence(static_cast<double>(obj_fus_3d.type_confidence));

        // hozon.common.Point3D acceleration
        fur_obj->mutable_acceleration()->set_x(static_cast<double>(obj_fus_3d.accel.x));
        fur_obj->mutable_acceleration()->set_y(static_cast<double>(obj_fus_3d.accel.y));
        fur_obj->mutable_acceleration()->set_z(static_cast<double>(obj_fus_3d.accel.z));

        // LightStatus light_status
        fur_obj->mutable_light_status()->set_brake_visible((converBrakeLightStatus(obj_fus_3d.BrakeLightSt)));
        fur_obj->mutable_light_status()->set_turn_light(converTurnLightStatus(obj_fus_3d.TurnLightSt));
        // if (obj_fus_3d.TurnLightSt == 2)
        //     fur_obj->mutable_light_status()->set_left_turn_visible(static_cast<double>(obj_fus_3d.TurnLightSt));
        // else if (obj_fus_3d.TurnLightSt == 3)
        //     fur_obj->mutable_light_status()->set_right_turn_visible(static_cast<double>(obj_fus_3d.TurnLightSt));

        // MotionType motion_type
        fur_obj->set_motion_type(static_cast<hozon::perception::PerceptionObstacle::MotionType>(obj_fus_3d.mottionPattern));

        fur_obj->set_maintenance_type(static_cast<hozon::perception::PerceptionObstacle::MaintenanceType>(obj_fus_3d.mainten_status));

        fur_obj->set_existence_probability(static_cast<double>(obj_fus_3d.existenceProbability));

        // DetectSensor current_detect_sensor
        for (int i = 0; i < 31; i++) {
            if (0x01 & (obj_fus_3d.detectSensor_cur >> i)) {
                switch (i) {
                    case 0:
                        fur_obj->mutable_current_detect_sensor()->set_from_camera_front_long_range(true);
                        break;
                    case 1:
                        fur_obj->mutable_current_detect_sensor()->set_from_camera_front_wide_angle(true);
                        break;
                    case 2:
                        fur_obj->mutable_current_detect_sensor()->set_from_camera_right_forward_looking(true);
                        break;
                    case 3:
                        fur_obj->mutable_current_detect_sensor()->set_from_camera_right_backward_looking(true);
                        break;
                    case 4:
                        fur_obj->mutable_current_detect_sensor()->set_from_camera_left_forward_looking(true);
                        break;
                    case 5:
                        fur_obj->mutable_current_detect_sensor()->set_from_camera_left_backward_looking(true);
                        break;
                    case 6:
                        fur_obj->mutable_current_detect_sensor()->set_from_camera_rear(true);
                        break;
                    case 7:
                        fur_obj->mutable_current_detect_sensor()->set_from_radar_front(true);
                        break;
                    case 8:
                        fur_obj->mutable_current_detect_sensor()->set_from_radar_front_right(true);
                        break;
                    case 9:
                        fur_obj->mutable_current_detect_sensor()->set_from_radar_front_left(true);
                        break;
                    case 10:
                        fur_obj->mutable_current_detect_sensor()->set_from_radar_rear_right(true);
                        break;
                    case 11:
                        fur_obj->mutable_current_detect_sensor()->set_from_radar_rear_left(true);
                        break;
                    case 12:
                        fur_obj->mutable_current_detect_sensor()->set_from_lidar_front_right(true);
                        break;
                    case 13:
                        fur_obj->mutable_current_detect_sensor()->set_from_lidar_front_left(true);
                        break;
                    case 14:
                        fur_obj->mutable_current_detect_sensor()->set_from_camera_fisheye_front(true);
                        break;
                    case 15:
                        fur_obj->mutable_current_detect_sensor()->set_from_camera_fisheye_rear(true);
                        break;
                    case 16:
                        fur_obj->mutable_current_detect_sensor()->set_from_camera_fisheye_left(true);
                        break;
                    case 17:
                        fur_obj->mutable_current_detect_sensor()->set_from_camera_fisheye_right(true);
                        break;
                    case 18:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_fol(true);
                        break;
                    case 19:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_fcl(true);
                        break;
                    case 20:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_fcr(true);
                        break;
                    case 21:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_for(true);
                        break;
                    case 22:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_rol(true);
                        break;
                    case 23:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_rcl(true);
                        break;
                    case 24:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_rcr(true);
                        break;
                    case 25:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_ror(true);
                        break;
                    case 26:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_fsl(true);
                        break;
                    case 27:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_rsl(true);
                        break;
                    case 28:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_fsr(true);
                        break;
                    case 29:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_rsr(true);
                        break;
                    case 30:
                        fur_obj->mutable_current_detect_sensor()->set_from_uss_fusion(true);
                        break;

                    default:
                        break;
                }
            }
        }

        // DetectSensor history_detect_sensor
        for (int i = 0; i < 31; i++) {
            if (0x01 & (obj_fus_3d.detectSensor_his >> i)) {
                switch (i) {
                    case 0:
                        fur_obj->mutable_history_detect_sensor()->set_from_camera_front_long_range(true);
                        break;
                    case 1:
                        fur_obj->mutable_history_detect_sensor()->set_from_camera_front_wide_angle(true);
                        break;
                    case 2:
                        fur_obj->mutable_history_detect_sensor()->set_from_camera_right_forward_looking(true);
                        break;
                    case 3:
                        fur_obj->mutable_history_detect_sensor()->set_from_camera_right_backward_looking(true);
                        break;
                    case 4:
                        fur_obj->mutable_history_detect_sensor()->set_from_camera_left_forward_looking(true);
                        break;
                    case 5:
                        fur_obj->mutable_history_detect_sensor()->set_from_camera_left_backward_looking(true);
                        break;
                    case 6:
                        fur_obj->mutable_history_detect_sensor()->set_from_camera_rear(true);
                        break;
                    case 7:
                        fur_obj->mutable_history_detect_sensor()->set_from_radar_front(true);
                        break;
                    case 8:
                        fur_obj->mutable_history_detect_sensor()->set_from_radar_front_right(true);
                        break;
                    case 9:
                        fur_obj->mutable_history_detect_sensor()->set_from_radar_front_left(true);
                        break;
                    case 10:
                        fur_obj->mutable_history_detect_sensor()->set_from_radar_rear_right(true);
                        break;
                    case 11:
                        fur_obj->mutable_history_detect_sensor()->set_from_radar_rear_left(true);
                        break;
                    case 12:
                        fur_obj->mutable_history_detect_sensor()->set_from_lidar_front_right(true);
                        break;
                    case 13:
                        fur_obj->mutable_history_detect_sensor()->set_from_lidar_front_left(true);
                        break;
                    case 14:
                        fur_obj->mutable_history_detect_sensor()->set_from_camera_fisheye_front(true);
                        break;
                    case 15:
                        fur_obj->mutable_history_detect_sensor()->set_from_camera_fisheye_rear(true);
                        break;
                    case 16:
                        fur_obj->mutable_history_detect_sensor()->set_from_camera_fisheye_left(true);
                        break;
                    case 17:
                        fur_obj->mutable_history_detect_sensor()->set_from_camera_fisheye_right(true);
                        break;
                    case 18:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_fol(true);
                        break;
                    case 19:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_fcl(true);
                        break;
                    case 20:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_fcr(true);
                        break;
                    case 21:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_for(true);
                        break;
                    case 22:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_rol(true);
                        break;
                    case 23:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_rcl(true);
                        break;
                    case 24:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_rcr(true);
                        break;
                    case 25:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_ror(true);
                        break;
                    case 26:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_fsl(true);
                        break;
                    case 27:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_rsl(true);
                        break;
                    case 28:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_fsr(true);
                        break;
                    case 29:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_rsr(true);
                        break;
                    case 30:
                        fur_obj->mutable_history_detect_sensor()->set_from_uss_fusion(true);
                        break;

                    default:
                        break;
                }
            }
        }

        // MotionType history_motion_type
        fur_obj->set_history_motion_type(static_cast<hozon::perception::PerceptionObstacle::MotionType>(obj_fus_3d.MotionPatternHistory));

        fur_obj->set_track_age(obj_fus_3d.age);

        // hozon.common.Point3D center
        fur_obj->mutable_center()->set_x(obj_fus_3d.rect.Center.x);
        fur_obj->mutable_center()->set_y(obj_fus_3d.rect.Center.y);
        fur_obj->mutable_center()->set_z(obj_fus_3d.rect.Center.z);
    }

    // hozon.common.Header header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);

    return proto_data;
}