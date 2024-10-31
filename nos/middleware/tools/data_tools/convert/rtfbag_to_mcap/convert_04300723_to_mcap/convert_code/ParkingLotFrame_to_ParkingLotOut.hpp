#pragma once
#include "hozon/parkinglot/impl_type_parkinglotframe.h"  // mdc 数据变量
#include "proto/perception/perception_parking_lot.pb.h"  // proto 数据变量

hozon::perception::ParkingLotOutArray ParkingLotFrameToParkingLotOut(hozon::parkinglot::ParkingLotFrame mdc_data) {
    hozon::perception::ParkingLotOutArray proto_data;

    // hozon.common.Header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);

    proto_data.set_locseq(mdc_data.locSeq);
    proto_data.set_opt_parking_seq(mdc_data.optSeq);
    proto_data.set_path_point_size(mdc_data.PathPointSize);

    for (auto lot : mdc_data.parkingLots) {
        hozon::perception::ParkingLotOut* lot_ptr = proto_data.add_parking_lots();
        lot_ptr->set_parking_seq(lot.seq);
        lot_ptr->set_type(static_cast<hozon::perception::ParkingLotOut::ParkType>(lot.type));
        lot_ptr->set_status(static_cast<hozon::perception::ParkingLotOut::ParkStatus>(lot.status));
        lot_ptr->set_sensor_type(static_cast<hozon::perception::ParkingLotOut::SenType>(lot.sensorType));
        lot_ptr->set_is_private_ps(lot.isPrivatePs);
        lot_ptr->set_time_creation(static_cast<double>(lot.timeCreation.sec) + static_cast<double>(lot.timeCreation.nsec) / 1e9);
        for (auto pts : lot.ptsVRF) {
            hozon::perception::PSPoint* pts_ptr = lot_ptr->add_pts_vrf();
            pts_ptr->mutable_point()->set_x(pts.point.x);
            pts_ptr->mutable_point()->set_y(pts.point.y);
            pts_ptr->mutable_point()->set_z(pts.point.z);

            pts_ptr->set_position(static_cast<hozon::perception::PSPoint::Position>(pts.position));
            pts_ptr->set_quality(static_cast<hozon::perception::PSPoint::Quality>(pts.quality));
        }
    }

    for (auto path : mdc_data.tracedPath) {
        hozon::perception::ParkingPathPoint* path_ptr = proto_data.add_traced_path();
        path_ptr->set_x(static_cast<double>(path.x));
        path_ptr->set_y(static_cast<double>(path.y));
        path_ptr->set_z(static_cast<double>(path.z));
        path_ptr->set_yaw(static_cast<double>(path.yaw));
        path_ptr->set_accumulate_s(static_cast<double>(path.accumulate_s));
        path_ptr->set_gear(static_cast<uint32_t>(path.gear));
    }

    return proto_data;
}