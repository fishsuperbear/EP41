#pragma once
#include "hozon/sensors/impl_type_radartrackarrayframe.h"  // mdc 数据变量
#include "proto/soc/radar.pb.h"                            // proto 数据变量

hozon::soc::RadarTrackArrayFrame RadarFrameToRadarOut(hozon::sensors::RadarTrackArrayFrame mdc_data) {
    hozon::soc::RadarTrackArrayFrame proto_data;

    // hozon.common.Header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);

    proto_data.set_sensor_id(static_cast<uint32_t>(mdc_data.sensorID));
    proto_data.set_radar_state(static_cast<uint32_t>(mdc_data.radarState));

    // RadarTrackData
    for (auto data : mdc_data.trackList) {
        hozon::soc::RadarTrackData* data_ptr = proto_data.add_track_list();
        data_ptr->set_id(data.id);
        data_ptr->mutable_position()->set_x(data.position.x);
        data_ptr->mutable_position()->set_y(data.position.y);
        data_ptr->mutable_position()->set_z(data.position.z);
        data_ptr->mutable_position()->mutable_rms()->set_x(data.position.rms.x);
        data_ptr->mutable_position()->mutable_rms()->set_y(data.position.rms.y);
        data_ptr->mutable_position()->mutable_rms()->set_z(data.position.rms.z);
        data_ptr->mutable_position()->mutable_quality()->set_x(data.position.quality.x);
        data_ptr->mutable_position()->mutable_quality()->set_y(data.position.quality.y);
        data_ptr->mutable_position()->mutable_quality()->set_z(data.position.quality.z);

        data_ptr->mutable_velocity()->set_x(data.velocity.x);
        data_ptr->mutable_velocity()->set_y(data.velocity.y);
        data_ptr->mutable_velocity()->set_z(data.velocity.z);
        data_ptr->mutable_velocity()->mutable_rms()->set_x(data.velocity.rms.x);
        data_ptr->mutable_velocity()->mutable_rms()->set_y(data.velocity.rms.y);
        data_ptr->mutable_velocity()->mutable_rms()->set_z(data.velocity.rms.z);
        data_ptr->mutable_velocity()->mutable_quality()->set_x(data.velocity.quality.x);
        data_ptr->mutable_velocity()->mutable_quality()->set_y(data.velocity.quality.y);
        data_ptr->mutable_velocity()->mutable_quality()->set_z(data.velocity.quality.z);

        data_ptr->mutable_acceleration()->set_x(data.acceleration.x);
        data_ptr->mutable_acceleration()->set_y(data.acceleration.y);
        data_ptr->mutable_acceleration()->set_z(data.acceleration.z);
        data_ptr->mutable_acceleration()->mutable_rms()->set_x(data.acceleration.rms.x);
        data_ptr->mutable_acceleration()->mutable_rms()->set_y(data.acceleration.rms.y);
        data_ptr->mutable_acceleration()->mutable_rms()->set_z(data.acceleration.rms.z);
        data_ptr->mutable_acceleration()->mutable_quality()->set_x(data.acceleration.quality.x);
        data_ptr->mutable_acceleration()->mutable_quality()->set_y(data.acceleration.quality.y);
        data_ptr->mutable_acceleration()->mutable_quality()->set_z(data.acceleration.quality.z);

        data_ptr->set_rcs(data.rcs);
        data_ptr->set_snr(data.snr);
        data_ptr->set_exist_probability(data.existProbability);
        data_ptr->set_mov_property(static_cast<uint32_t>(data.movProperty));
        data_ptr->set_track_type(static_cast<uint32_t>(data.trackType));
        data_ptr->set_obj_obstacle_prob(static_cast<uint32_t>(data.objObstacleProb));
        data_ptr->set_measstate(static_cast<uint32_t>(data.measState));

        data_ptr->mutable_size_lwh()->set_x(data.sizeLWH.x);
        data_ptr->mutable_size_lwh()->set_y(data.sizeLWH.y);
        data_ptr->mutable_size_lwh()->set_z(data.sizeLWH.z);

        data_ptr->set_orient_agl(data.orientAgl);
    }
    return proto_data;
}