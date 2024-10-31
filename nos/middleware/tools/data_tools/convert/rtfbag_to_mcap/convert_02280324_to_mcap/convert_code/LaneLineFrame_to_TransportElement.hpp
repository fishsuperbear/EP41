#pragma once
#include "hozon/laneline/impl_type_lanelineframe.h"
#include "proto/perception/transport_element.pb.h"  // proto 数据变量

hozon::perception::LaneType ConvertClassTypeToLaneType(UInt8 mdc_enum) {
    hozon::perception::LaneType proto_enum;

    switch (mdc_enum) {
        case 0:
            proto_enum = hozon::perception::LaneType::SolidLine;
            break;
        case 1:
            proto_enum = hozon::perception::LaneType::DashedLine;
            break;
        case 2:
            proto_enum = hozon::perception::LaneType::ShortDashedLine;
            break;
        case 3:
            proto_enum = hozon::perception::LaneType::DoubleSolidLine;
            break;
        case 4:
            proto_enum = hozon::perception::LaneType::DoubleDashedLine;
            break;
        case 5:
            proto_enum = hozon::perception::LaneType::LeftSolidRightDashed;
            break;
        case 6:
            proto_enum = hozon::perception::LaneType::RightSolidLeftDashed;
            break;
        default:
            proto_enum = hozon::perception::LaneType::Other;
            break;
    }
    return proto_enum;
}

hozon::perception::Color ConvertColorType(UInt8 mdc_enum) {
    hozon::perception::Color proto_enum;
    switch (mdc_enum) {
        case 0:
            proto_enum = hozon::perception::Color::WHITE;
            break;
        case 1:
            proto_enum = hozon::perception::Color::YELLOW;
            break;
        case 4:
            proto_enum = hozon::perception::Color::GREEN;
            break;
        default:
            proto_enum = hozon::perception::Color::UNKNOWN;
            break;
    }
    return proto_enum;
}

hozon::perception::TransportElement LaneLineFrameToTransportElement(hozon::laneline::LaneLineFrame mdc_data) {
    hozon::perception::TransportElement proto_data;

    // hozon.common.Header header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);
    // repeated LaneInfo lane = 2;  // road_edge 也在这里边，用type区分
    for (auto lane_front_array_array : mdc_data.laneDetectionFrontOut) {
        for (auto item : lane_front_array_array) {
            hozon::perception::LaneInfo* lan_info = proto_data.add_lane();

            // int32 track_id
            lan_info->set_track_id(item.lanSeq);
            // LaneType lanetype = 2;            //车道线线型
            lan_info->set_lanetype(ConvertClassTypeToLaneType(item.cls));
            // LanePositionType lanepos
            // repeated hozon.common.Point3D
            for (auto point : item.pointVehicleCoord) {
                hozon::common::Point3D* point_3 = lan_info->add_points();
                point_3->set_x(point.x);
                point_3->set_y(point.y);
                point_3->set_z(point.z);
            }
            // optional LaneCubicSpline lane_param
            hozon::perception::LaneCubicCurve* lane_cubic_curve = lan_info->mutable_lane_param()->add_cubic_curve_set();
            lane_cubic_curve->set_start_point_x(item.laneFits.xStartVRF);
            lane_cubic_curve->set_end_point_x(item.laneFits.xEndVRF);

            lane_cubic_curve->set_c0(item.laneFits.coefficients.a);
            lane_cubic_curve->set_c1(item.laneFits.coefficients.b);
            lane_cubic_curve->set_c2(item.laneFits.coefficients.c);
            lane_cubic_curve->set_c3(item.laneFits.coefficients.d);

            // optional double confidence
            lan_info->set_confidence(item.geoConfidence);
            // optional LaneUseType use_type = 7;         //车道线生成来源
            // optional Color color
            lan_info->set_color(ConvertColorType(item.color));
        }
    }

    // repeated Arrow arror = 4;
    // optional StopLine stop_line = 5;
    // repeated ZebraCrossing zebra_crossing = 6;
    // repeated NoParkingZone no_parking_zone = 7;
    // repeated TurnWaitingZone turn_waiting_zone = 8;
    // repeated LightPole light_poles = 9;
    // repeated TrafficLight traffic_lights = 10;
    // repeated TrafficSign traffic_signs = 11;

    return proto_data;
}