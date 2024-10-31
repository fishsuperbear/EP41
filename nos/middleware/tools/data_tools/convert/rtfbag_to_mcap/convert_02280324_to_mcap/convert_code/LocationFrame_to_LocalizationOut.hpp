#pragma once
#include "hozon/location/impl_type_locationframe.h"  // mdc 数据变量
#include "proto/localization/localization.pb.h"      // proto 数据变量

hozon::localization::Localization LocationFrameToLocalizationOut(hozon::location::LocationFrame mdc_data) {
    hozon::localization::Localization proto_data;

    // hozon.common.Header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);

    // hozon.localization.Pose
    proto_data.mutable_pose()->mutable_position()->set_x(mdc_data.pose.poseLOCAL.position.x);
    proto_data.mutable_pose()->mutable_position()->set_y(mdc_data.pose.poseLOCAL.position.y);
    proto_data.mutable_pose()->mutable_position()->set_z(mdc_data.pose.poseLOCAL.position.z);
    proto_data.mutable_pose()->mutable_quaternion()->set_w(mdc_data.pose.poseLOCAL.quaternion.w);
    proto_data.mutable_pose()->mutable_quaternion()->set_x(mdc_data.pose.poseLOCAL.quaternion.x);
    proto_data.mutable_pose()->mutable_quaternion()->set_y(mdc_data.pose.poseLOCAL.quaternion.y);
    proto_data.mutable_pose()->mutable_quaternion()->set_z(mdc_data.pose.poseLOCAL.quaternion.z);
    proto_data.mutable_pose()->mutable_euler_angles()->set_x(mdc_data.pose.poseLOCAL.eulerAngle.x);
    proto_data.mutable_pose()->mutable_euler_angles()->set_y(mdc_data.pose.poseLOCAL.eulerAngle.y);
    proto_data.mutable_pose()->mutable_euler_angles()->set_z(mdc_data.pose.poseLOCAL.eulerAngle.z);
    proto_data.mutable_pose()->set_heading(mdc_data.pose.poseLOCAL.heading);

    proto_data.mutable_pose()->mutable_linear_velocity_vrf()->set_x(mdc_data.velocity.twist.linear.x);
    proto_data.mutable_pose()->mutable_linear_velocity_vrf()->set_y(mdc_data.velocity.twist.linear.y);
    proto_data.mutable_pose()->mutable_linear_velocity_vrf()->set_z(mdc_data.velocity.twist.linear.z);

    proto_data.mutable_pose()->mutable_angular_velocity_vrf()->set_x(mdc_data.velocity.twist.angular.x);
    proto_data.mutable_pose()->mutable_angular_velocity_vrf()->set_y(mdc_data.velocity.twist.angular.y);
    proto_data.mutable_pose()->mutable_angular_velocity_vrf()->set_z(mdc_data.velocity.twist.angular.z);

    proto_data.mutable_pose()->mutable_linear_acceleration_raw_vrf()->set_x(mdc_data.acceleration.accel.linearRaw.x);
    proto_data.mutable_pose()->mutable_linear_acceleration_raw_vrf()->set_y(mdc_data.acceleration.accel.linearRaw.y);
    proto_data.mutable_pose()->mutable_linear_acceleration_raw_vrf()->set_z(mdc_data.acceleration.accel.linearRaw.z);

    proto_data.mutable_pose()->mutable_linear_acceleration_vrf()->set_x(mdc_data.acceleration.accel.linear.x);
    proto_data.mutable_pose()->mutable_linear_acceleration_vrf()->set_y(mdc_data.acceleration.accel.linear.y);
    proto_data.mutable_pose()->mutable_linear_acceleration_vrf()->set_z(mdc_data.acceleration.accel.linear.z);

    proto_data.set_rtk_status(mdc_data.rtkStatus);
    proto_data.set_location_state(mdc_data.locationState);
    proto_data.set_received_ehp_counter(mdc_data.received_ehp_counter);

    return proto_data;
}