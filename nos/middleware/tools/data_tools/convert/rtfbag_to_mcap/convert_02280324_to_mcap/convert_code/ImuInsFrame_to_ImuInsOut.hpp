#pragma once
#include "hozon/sensors/impl_type_imuinfoframe.h"     // mdc 数据变量
#include "hozon/sensors/impl_type_imuinsinfoframe.h"  // mdc 数据变量
#include "hozon/sensors/impl_type_insinfoframe.h"     // mdc 数据变量
#include "proto/soc/sensor_imu_ins.pb.h"              // proto 数据变量

hozon::soc::ImuIns ImuInsInfoFrameToImuInsOut(hozon::sensors::ImuInsInfoFrame mdc_data) {
    hozon::soc::ImuIns proto_data;

    // ImuIns
    // hozon.common.Header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);
    // proto_data.mutable_header()->mutable_time_stamp()->set_sec(mdc_data.header.stamp.sec);
    // proto_data.mutable_header()->mutable_time_stamp()->set_nsec(mdc_data.header.stamp.nsec);
    // proto_data.mutable_header()->mutable_gnss_stamp()->set_sec(mdc_data.header.gnssStamp.sec);
    // proto_data.mutable_header()->mutable_gnss_stamp()->set_nsec(mdc_data.header.gnssStamp.nsec);

    proto_data.set_gps_week(mdc_data.gpsWeek);
    proto_data.set_gps_sec(mdc_data.gpsWeek);

    // ImuInfo
    proto_data.mutable_imu_info()->mutable_angular_velocity()->set_x(mdc_data.imu_info.angularVelocity.x);
    proto_data.mutable_imu_info()->mutable_angular_velocity()->set_y(mdc_data.imu_info.angularVelocity.y);
    proto_data.mutable_imu_info()->mutable_angular_velocity()->set_z(mdc_data.imu_info.angularVelocity.z);

    proto_data.mutable_imu_info()->mutable_linear_acceleration()->set_x(mdc_data.imu_info.acceleration.x);
    proto_data.mutable_imu_info()->mutable_linear_acceleration()->set_y(mdc_data.imu_info.acceleration.y);
    proto_data.mutable_imu_info()->mutable_linear_acceleration()->set_z(mdc_data.imu_info.acceleration.z);

    proto_data.mutable_imu_info()->set_imu_status(mdc_data.imu_info.imuStatus);
    proto_data.mutable_imu_info()->set_temperature(mdc_data.imu_info.temperature);
    proto_data.mutable_imu_info()->set_imuyaw(mdc_data.imu_info.imuyaw);

    proto_data.mutable_imu_info()->mutable_imuvb_angular_velocity()->set_x(mdc_data.imu_info.imuVBAngularVelocity.x);
    proto_data.mutable_imu_info()->mutable_imuvb_angular_velocity()->set_y(mdc_data.imu_info.imuVBAngularVelocity.y);
    proto_data.mutable_imu_info()->mutable_imuvb_angular_velocity()->set_z(mdc_data.imu_info.imuVBAngularVelocity.z);

    proto_data.mutable_imu_info()->mutable_imuvb_linear_acceleration()->set_x(mdc_data.imu_info.imuVBLinearAcceleration.x);
    proto_data.mutable_imu_info()->mutable_imuvb_linear_acceleration()->set_y(mdc_data.imu_info.imuVBLinearAcceleration.y);
    proto_data.mutable_imu_info()->mutable_imuvb_linear_acceleration()->set_z(mdc_data.imu_info.imuVBLinearAcceleration.y);

    proto_data.mutable_imu_info()->mutable_gyro_offset()->set_x(mdc_data.imu_info.gyroOffset.x);
    proto_data.mutable_imu_info()->mutable_gyro_offset()->set_y(mdc_data.imu_info.gyroOffset.y);
    proto_data.mutable_imu_info()->mutable_gyro_offset()->set_z(mdc_data.imu_info.gyroOffset.z);

    proto_data.mutable_imu_info()->mutable_accel_offset()->set_x(mdc_data.imu_info.accelOffset.x);
    proto_data.mutable_imu_info()->mutable_accel_offset()->set_y(mdc_data.imu_info.accelOffset.y);
    proto_data.mutable_imu_info()->mutable_accel_offset()->set_z(mdc_data.imu_info.accelOffset.z);

    proto_data.mutable_imu_info()->mutable_ins2antoffset()->set_x(mdc_data.imu_info.ins2antoffset.x);
    proto_data.mutable_imu_info()->mutable_ins2antoffset()->set_y(mdc_data.imu_info.ins2antoffset.y);
    proto_data.mutable_imu_info()->mutable_ins2antoffset()->set_z(mdc_data.imu_info.ins2antoffset.z);

    proto_data.mutable_imu_info()->mutable_imu2bodyosffet()->mutable_imu_position()->set_x(mdc_data.imu_info.imu2bodyosffet.imuPosition.x);
    proto_data.mutable_imu_info()->mutable_imu2bodyosffet()->mutable_imu_position()->set_y(mdc_data.imu_info.imu2bodyosffet.imuPosition.y);
    proto_data.mutable_imu_info()->mutable_imu2bodyosffet()->mutable_imu_position()->set_z(mdc_data.imu_info.imu2bodyosffet.imuPosition.z);

    proto_data.mutable_imu_info()->mutable_imu2bodyosffet()->mutable_euler_angle()->set_x(mdc_data.imu_info.imu2bodyosffet.eulerAngle.x);
    proto_data.mutable_imu_info()->mutable_imu2bodyosffet()->mutable_euler_angle()->set_y(mdc_data.imu_info.imu2bodyosffet.eulerAngle.y);
    proto_data.mutable_imu_info()->mutable_imu2bodyosffet()->mutable_euler_angle()->set_z(mdc_data.imu_info.imu2bodyosffet.eulerAngle.z);

    // InsInfo
    proto_data.mutable_ins_info()->set_latitude(mdc_data.ins_info.latitude);
    proto_data.mutable_ins_info()->set_longitude(mdc_data.ins_info.longitude);
    proto_data.mutable_ins_info()->set_altitude(mdc_data.ins_info.altitude);

    proto_data.mutable_ins_info()->mutable_attitude()->set_x(mdc_data.ins_info.attitude.x);
    proto_data.mutable_ins_info()->mutable_attitude()->set_y(mdc_data.ins_info.attitude.y);
    proto_data.mutable_ins_info()->mutable_attitude()->set_z(mdc_data.ins_info.attitude.z);

    proto_data.mutable_ins_info()->mutable_linear_velocity()->set_x(mdc_data.ins_info.linearVelocity.x);
    proto_data.mutable_ins_info()->mutable_linear_velocity()->set_y(mdc_data.ins_info.linearVelocity.y);
    proto_data.mutable_ins_info()->mutable_linear_velocity()->set_z(mdc_data.ins_info.linearVelocity.z);

    proto_data.mutable_ins_info()->mutable_sd_position()->set_x(mdc_data.ins_info.sdPosition.x);
    proto_data.mutable_ins_info()->mutable_sd_position()->set_y(mdc_data.ins_info.sdPosition.y);
    proto_data.mutable_ins_info()->mutable_sd_position()->set_z(mdc_data.ins_info.sdPosition.z);

    proto_data.mutable_ins_info()->mutable_sd_attitude()->set_x(mdc_data.ins_info.sdAttitude.x);
    proto_data.mutable_ins_info()->mutable_sd_attitude()->set_y(mdc_data.ins_info.sdAttitude.y);
    proto_data.mutable_ins_info()->mutable_sd_attitude()->set_z(mdc_data.ins_info.sdAttitude.z);

    proto_data.mutable_ins_info()->mutable_sd_velocity()->set_x(mdc_data.ins_info.sdVelocity.x);
    proto_data.mutable_ins_info()->mutable_sd_velocity()->set_y(mdc_data.ins_info.sdVelocity.y);
    proto_data.mutable_ins_info()->mutable_sd_velocity()->set_z(mdc_data.ins_info.sdVelocity.z);

    proto_data.mutable_ins_info()->set_sys_status(mdc_data.ins_info.sysStatus);
    proto_data.mutable_ins_info()->set_gps_status(mdc_data.ins_info.gpsStatus);

    proto_data.mutable_ins_info()->mutable_augular_velocity()->set_x(mdc_data.ins_info.augularVelocity.x);
    proto_data.mutable_ins_info()->mutable_augular_velocity()->set_y(mdc_data.ins_info.augularVelocity.y);
    proto_data.mutable_ins_info()->mutable_augular_velocity()->set_z(mdc_data.ins_info.augularVelocity.z);

    proto_data.mutable_ins_info()->mutable_linear_acceleration()->set_x(mdc_data.ins_info.linearAcceleration.x);
    proto_data.mutable_ins_info()->mutable_linear_acceleration()->set_y(mdc_data.ins_info.linearAcceleration.y);
    proto_data.mutable_ins_info()->mutable_linear_acceleration()->set_z(mdc_data.ins_info.linearAcceleration.z);

    proto_data.mutable_ins_info()->set_heading(mdc_data.ins_info.heading);

    proto_data.mutable_ins_info()->mutable_mounting_error()->set_x(mdc_data.ins_info.mountingError.x);
    proto_data.mutable_ins_info()->mutable_mounting_error()->set_y(mdc_data.ins_info.mountingError.y);
    proto_data.mutable_ins_info()->mutable_mounting_error()->set_z(mdc_data.ins_info.mountingError.z);

    proto_data.mutable_ins_info()->set_sensor_used(mdc_data.ins_info.sensorUsed);
    proto_data.mutable_ins_info()->set_wheel_velocity(mdc_data.ins_info.wheelVelocity);
    proto_data.mutable_ins_info()->set_odo_sf(mdc_data.ins_info.odoSF);

    // OffsetInfo
    proto_data.mutable_offset_info()->mutable_gyo_bias()->set_x(mdc_data.offset_info.gyoBias.x);
    proto_data.mutable_offset_info()->mutable_gyo_bias()->set_y(mdc_data.offset_info.gyoBias.y);
    proto_data.mutable_offset_info()->mutable_gyo_bias()->set_z(mdc_data.offset_info.gyoBias.z);

    proto_data.mutable_offset_info()->mutable_gyo_sf()->set_x(mdc_data.offset_info.gyoSF.x);
    proto_data.mutable_offset_info()->mutable_gyo_sf()->set_y(mdc_data.offset_info.gyoSF.y);
    proto_data.mutable_offset_info()->mutable_gyo_sf()->set_z(mdc_data.offset_info.gyoSF.z);

    proto_data.mutable_offset_info()->mutable_acc_bias()->set_x(mdc_data.offset_info.accBias.x);
    proto_data.mutable_offset_info()->mutable_acc_bias()->set_y(mdc_data.offset_info.accBias.y);
    proto_data.mutable_offset_info()->mutable_acc_bias()->set_z(mdc_data.offset_info.accBias.z);

    proto_data.mutable_offset_info()->mutable_acc_sf()->set_x(mdc_data.offset_info.accSF.x);
    proto_data.mutable_offset_info()->mutable_acc_sf()->set_y(mdc_data.offset_info.accSF.y);
    proto_data.mutable_offset_info()->mutable_acc_sf()->set_z(mdc_data.offset_info.accSF.z);

    return proto_data;
}