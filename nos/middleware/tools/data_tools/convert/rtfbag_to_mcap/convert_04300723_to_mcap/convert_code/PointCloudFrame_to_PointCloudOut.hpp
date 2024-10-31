#pragma once
#include "hozon/sensors/impl_type_pointcloudframe.h"  // mdc 数据变量
#include "proto/soc/point_cloud.pb.h"                 // proto 数据变量

hozon::soc::PointCloud PointCloudFrameToPointCloudOut(hozon::sensors::PointCloudFrame mdc_data) {
    hozon::soc::PointCloud proto_data;

    // hozon.common.Header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);

    proto_data.set_is_big_endian(mdc_data.isBigEndian);
    proto_data.set_height(mdc_data.height);
    proto_data.set_width(mdc_data.width);
    proto_data.set_point_step(mdc_data.pointStep);
    proto_data.set_row_step(mdc_data.rowStep);
    proto_data.set_is_dense(mdc_data.isDense);

    // PointField
    for (auto point : mdc_data.data) {
        hozon::soc::PointField* point_ptr = proto_data.add_points();
        point_ptr->set_time(point.time);
        point_ptr->set_x(static_cast<double>(point.x));
        point_ptr->set_y(static_cast<double>(point.y));
        point_ptr->set_z(static_cast<double>(point.z));

        point_ptr->set_distance(static_cast<double>(point.distance));
        point_ptr->set_pitch(static_cast<double>(point.pitch));
        point_ptr->set_yaw(static_cast<double>(point.yaw));
        point_ptr->set_intensity(static_cast<uint32_t>(point.intensity));
        point_ptr->set_ring(static_cast<uint32_t>(point.ring));
        point_ptr->set_block(static_cast<uint32_t>(point.block));
    }
    proto_data.set_ecu_serial_number(static_cast<std::string>(mdc_data.lidarSN.ecuSerialNumber));
    // LidarEolCalibStatus
    proto_data.mutable_eol_calib_status()->set_calib_status(static_cast<uint32_t>(mdc_data.eolCalibStatus.calib_status));
    proto_data.mutable_eol_calib_status()->set_rotationx(static_cast<double>(mdc_data.eolCalibStatus.rotationX));
    proto_data.mutable_eol_calib_status()->set_rotationy(static_cast<double>(mdc_data.eolCalibStatus.rotationY));
    proto_data.mutable_eol_calib_status()->set_rotationz(static_cast<double>(mdc_data.eolCalibStatus.rotationZ));
    proto_data.mutable_eol_calib_status()->set_rotationw(static_cast<double>(mdc_data.eolCalibStatus.rotationW));
    proto_data.mutable_eol_calib_status()->set_translationx(static_cast<double>(mdc_data.eolCalibStatus.translationX));
    proto_data.mutable_eol_calib_status()->set_translationy(static_cast<double>(mdc_data.eolCalibStatus.translationY));
    proto_data.mutable_eol_calib_status()->set_translationz(static_cast<double>(mdc_data.eolCalibStatus.translationZ));

    return proto_data;
}