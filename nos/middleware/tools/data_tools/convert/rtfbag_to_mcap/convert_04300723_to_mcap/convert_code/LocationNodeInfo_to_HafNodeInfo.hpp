#pragma once
#include "hozon/location/impl_type_locationnodeinfo.h"
#include "proto/localization/node_info.pb.h"  // proto 数据变量

hozon::localization::HafNodeInfo LocationNodeInfoToHafNodeInfo(hozon::location::LocationNodeInfo mdc_data) {
    hozon::localization::HafNodeInfo proto_data;

    proto_data.set_type((hozon::localization::HafNodeInfo_NodeType)mdc_data.type);

    // hozon.common.Header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);

    proto_data.set_gps_week(mdc_data.gpsWeek);
    proto_data.set_gps_sec(mdc_data.gpsSec);

    // hozon.common.Point3D pos_wgs
    proto_data.mutable_pos_wgs()->set_x(static_cast<double>(mdc_data.posSmooth.x));
    proto_data.mutable_pos_wgs()->set_y(static_cast<double>(mdc_data.posSmooth.y));
    proto_data.mutable_pos_wgs()->set_z(static_cast<double>(mdc_data.posSmooth.z));

    // hozon.common.Point3D attitude
    proto_data.mutable_attitude()->set_x(static_cast<double>(mdc_data.attitude.x));
    proto_data.mutable_attitude()->set_y(static_cast<double>(mdc_data.attitude.y));
    proto_data.mutable_attitude()->set_z(static_cast<double>(mdc_data.attitude.z));

    // hozon.common.Quaternion quaternion
    proto_data.mutable_quaternion()->set_w(mdc_data.quaternion.w);
    proto_data.mutable_quaternion()->set_x(mdc_data.quaternion.x);
    proto_data.mutable_quaternion()->set_y(mdc_data.quaternion.y);
    proto_data.mutable_quaternion()->set_z(mdc_data.quaternion.z);

    // hozon.common.Point3D linear_velocity
    proto_data.mutable_linear_velocity()->set_x(static_cast<double>(mdc_data.linearVelocity.x));
    proto_data.mutable_linear_velocity()->set_y(static_cast<double>(mdc_data.linearVelocity.y));
    proto_data.mutable_linear_velocity()->set_z(static_cast<double>(mdc_data.linearVelocity.z));

    // hozon.common.Point3D gyro_bias
    proto_data.mutable_gyro_bias()->set_x(static_cast<double>(mdc_data.gyroBias.x));
    proto_data.mutable_gyro_bias()->set_y(static_cast<double>(mdc_data.gyroBias.y));
    proto_data.mutable_gyro_bias()->set_z(static_cast<double>(mdc_data.gyroBias.z));

    // hozon.common.Point3D accel_bias
    proto_data.mutable_accel_bias()->set_x(static_cast<double>(mdc_data.accelBias.x));
    proto_data.mutable_accel_bias()->set_y(static_cast<double>(mdc_data.accelBias.y));
    proto_data.mutable_accel_bias()->set_z(static_cast<double>(mdc_data.accelBias.z));

    // hozon.common.Point3D sd_position
    proto_data.mutable_sd_position()->set_x(static_cast<double>(mdc_data.sdPosition.x));
    proto_data.mutable_sd_position()->set_y(static_cast<double>(mdc_data.sdPosition.y));
    proto_data.mutable_sd_position()->set_z(static_cast<double>(mdc_data.sdPosition.z));

    // hozon.common.Point3D sd_attitude
    proto_data.mutable_sd_attitude()->set_x(static_cast<double>(mdc_data.sdAttitude.x));
    proto_data.mutable_sd_attitude()->set_y(static_cast<double>(mdc_data.sdAttitude.y));
    proto_data.mutable_sd_attitude()->set_z(static_cast<double>(mdc_data.sdAttitude.z));

    // hozon.common.Point3D sd_velocity
    proto_data.mutable_sd_velocity()->set_x(static_cast<double>(mdc_data.sdVelocity.x));
    proto_data.mutable_sd_velocity()->set_y(static_cast<double>(mdc_data.sdVelocity.y));
    proto_data.mutable_sd_velocity()->set_z(static_cast<double>(mdc_data.sdVelocity.z));

    // repeated float covariance
    for (auto item : mdc_data.covariance) {
        proto_data.add_covariance(item);
    }

    // uint32 sys_status
    proto_data.set_sys_status(mdc_data.sysStatus);
    // uint32 gps_status
    proto_data.set_gps_status(mdc_data.gpsStatus);
    // float heading
    proto_data.set_heading(mdc_data.heading);
    // uint32 warn_info
    proto_data.set_warn_info(mdc_data.warn_info);
    // hozon.common.Point3D pos_gcj02
    proto_data.mutable_pos_gcj02()->set_x(static_cast<double>(mdc_data.posGCJ02.x));
    proto_data.mutable_pos_gcj02()->set_y(static_cast<double>(mdc_data.posGCJ02.y));
    proto_data.mutable_pos_gcj02()->set_z(static_cast<double>(mdc_data.posGCJ02.z));
    // hozon.common.Point3D angular_velocity
    proto_data.mutable_angular_velocity()->set_x(static_cast<double>(mdc_data.angularVelocity.x));
    proto_data.mutable_angular_velocity()->set_y(static_cast<double>(mdc_data.angularVelocity.y));
    proto_data.mutable_angular_velocity()->set_z(static_cast<double>(mdc_data.angularVelocity.z));
    // hozon.common.Point3D linear_acceleration
    proto_data.mutable_linear_acceleration()->set_x(static_cast<double>(mdc_data.linearAcceleration.x));
    proto_data.mutable_angular_velocity()->set_y(static_cast<double>(mdc_data.angularVelocity.y));
    proto_data.mutable_angular_velocity()->set_z(static_cast<double>(mdc_data.angularVelocity.z));
    // hozon.common.Point3D mounting_error
    proto_data.mutable_mounting_error()->set_x(static_cast<double>(mdc_data.mountingError.x));
    proto_data.mutable_angular_velocity()->set_y(static_cast<double>(mdc_data.mountingError.y));
    proto_data.mutable_angular_velocity()->set_z(static_cast<double>(mdc_data.mountingError.z));
    // uint32 sensor_used
    proto_data.set_sensor_used(mdc_data.sensorUsed);
    // float wheel_velocity
    proto_data.set_wheel_velocity(mdc_data.wheelVelocity);
    // float odo_sf
    proto_data.set_odo_sf(mdc_data.odoSF);
    // bool valid_estimate
    proto_data.set_valid_estimate(mdc_data.validEstimate);

    return proto_data;
}