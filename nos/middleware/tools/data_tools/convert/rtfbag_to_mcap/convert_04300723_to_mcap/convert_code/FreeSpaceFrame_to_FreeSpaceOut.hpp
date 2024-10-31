#pragma once
#include "hozon/freespace/impl_type_freespaceframe.h"  // mdc 数据变量
#include "proto/perception/perception_freespace.pb.h"  // proto 数据变量

hozon::perception::FreeSpaceOutArray FreeSpaceFrameToFreeSpaceOut(hozon::freespace::FreeSpaceFrame mdc_data) {
    hozon::perception::FreeSpaceOutArray proto_data;

    // hozon.common.Header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);

    proto_data.set_locseq(mdc_data.locSeq);

    for (auto space : mdc_data.spaces) {
        hozon::perception::FreeSpaceOut* sp_ptr = proto_data.add_freespace_out();
        sp_ptr->set_freespace_seq(static_cast<uint32_t>(space.spaceSeq));
        sp_ptr->set_cls(static_cast<hozon::perception::FreeSpaceOut::ClassType>(space.cls));
        sp_ptr->set_height_type(static_cast<hozon::perception::FreeSpaceOut::HeightType>(space.heightType));
        sp_ptr->set_sensor_type(static_cast<hozon::perception::FreeSpaceOut::SensorType>(space.sensorType));

        for (auto p2d : space.freeSpacePointVRF) {
            hozon::common::Point3D* p2d_ptr = sp_ptr->add_freespace_point();
            p2d_ptr->set_x(p2d.x);
            p2d_ptr->set_y(p2d.y);
            p2d_ptr->set_z(p2d.z);
        }
        sp_ptr->set_time_creation(static_cast<double>(space.timeCreation.sec) + static_cast<double>(space.timeCreation.nsec) / 1e9);

        for (auto p2d : space.freeSpaceKeyPointVRF) {
            hozon::common::Point3D* p2d_ptr = sp_ptr->add_freespace_keypoint();
            p2d_ptr->set_x(p2d.x);
            p2d_ptr->set_y(p2d.y);
            p2d_ptr->set_z(p2d.z);
        }

        sp_ptr->set_islinkobjfusion(space.isLinkObjFusion);
    }

    for (auto space2d : mdc_data.Spaces2D) {
        hozon::perception::FreeSpace2DOut* sp2_ptr = proto_data.add_freespace_2d_out();
        sp2_ptr->set_freespace_seq(space2d.spaceSeq);
        sp2_ptr->set_sensor_name(space2d.sensorName);
        sp2_ptr->set_time_creation(static_cast<double>(space2d.timeCreation.sec) + static_cast<double>(space2d.timeCreation.nsec) / 1e9);
        for (auto info : space2d.points) {
            hozon::perception::FreeSpace2DOut::Point2DInfo* if_ptr = sp2_ptr->add_point2d_info();
            if_ptr->set_type(static_cast<hozon::perception::FreeSpace2DOut::Space2DType>(space2d.type));
            if_ptr->mutable_free_space_2d_point_vrf()->set_x(info.x);
            if_ptr->mutable_free_space_2d_point_vrf()->set_y(info.y);
        }
    }

    return proto_data;
}