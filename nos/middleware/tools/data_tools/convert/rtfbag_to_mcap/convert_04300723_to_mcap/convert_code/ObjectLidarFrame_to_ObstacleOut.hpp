#pragma once
#include "hozon/object/impl_type_objectlidarframe.h"
#include "proto/perception/perception_obstacle.pb.h"

hozon::perception::PerceptionObstacles ObjLidarFrameToObstacleOut(hozon::object::ObjectLidarFrame mdc_data) {
    hozon::perception::PerceptionObstacles proto_data;

    // PerceptionObstacle perception_obstacle
    for (auto obj_lidar_3d : mdc_data.object3d) {
        hozon::perception::PerceptionObstacle* lidar_obj = proto_data.add_perception_obstacle();
        lidar_obj->set_id(obj_lidar_3d.objectID);

        // hozon.common.Point3D velocity
        lidar_obj->mutable_velocity()->set_x(static_cast<double>(obj_lidar_3d.velocity.x));
        lidar_obj->mutable_velocity()->set_y(static_cast<double>(obj_lidar_3d.velocity.y));
        lidar_obj->mutable_velocity()->set_z(static_cast<double>(obj_lidar_3d.velocity.z));

        // hozon.common.Point3D polygon_point
        for (auto ctp : obj_lidar_3d.contourPoints) {
            hozon::common::Point3D* point = lidar_obj->add_polygon_point();
            point->set_x(static_cast<double>(ctp.x));
            point->set_y(static_cast<double>(ctp.y));
            point->set_z(static_cast<double>(ctp.z));
        }

        lidar_obj->set_type_confidence(static_cast<double>(obj_lidar_3d.confidence));

        // hozon.common.Point3D acceleration
        lidar_obj->mutable_acceleration()->set_x(static_cast<double>(obj_lidar_3d.accel.x));
        lidar_obj->mutable_acceleration()->set_y(static_cast<double>(obj_lidar_3d.accel.y));
        lidar_obj->mutable_acceleration()->set_z(static_cast<double>(obj_lidar_3d.accel.z));

        lidar_obj->set_sub_type(static_cast<hozon::perception::PerceptionObstacle::SubType>(obj_lidar_3d.cls));

        for (auto vel : obj_lidar_3d.velocity_unc) {
            lidar_obj->add_velocity_covariance(vel);
        }

        for (auto acc : obj_lidar_3d.accel_unc) {
            lidar_obj->add_acceleration_covariance(acc);
        }

        lidar_obj->set_motion_type(static_cast<hozon::perception::PerceptionObstacle::MotionType>(obj_lidar_3d.movingState));
        lidar_obj->set_existence_probability(static_cast<double>(obj_lidar_3d.existenceProbability));
        lidar_obj->set_track_state(static_cast<hozon::perception::PerceptionObstacle::TrackState>(obj_lidar_3d.trackState));

        // hozon.common.Point3D center
        lidar_obj->mutable_center()->set_x(obj_lidar_3d.rect.Center.x);
        lidar_obj->mutable_center()->set_y(obj_lidar_3d.rect.Center.y);
        lidar_obj->mutable_center()->set_z(obj_lidar_3d.rect.Center.z);

        lidar_obj->set_is_back_ground(obj_lidar_3d.isBackground);
    }

    // hozon.common.Header header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);

    // SensorStatus sensor_status
    proto_data.mutable_sensor_status()->set_lidar_status(static_cast<hozon::perception::SensorStatus::LidarStatus>(mdc_data.lidarState));

    return proto_data;
}