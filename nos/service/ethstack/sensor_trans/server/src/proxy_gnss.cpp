#include "proxy_gnss.h"

#include <memory>

#include "logger.h"
#include "common.h"

namespace hozon {
namespace netaos {
namespace sensor {

ProxyGnss::ProxyGnss() : _gnss_pub_last_seq(0) {}

std::shared_ptr<hozon::soc::gnss::GnssInfo> ProxyGnss::Trans(ara::com::SamplePtr<::hozon::netaos::AlgGnssInfo const> data) {
    std::shared_ptr<hozon::soc::gnss::GnssInfo> gnss_proto = std::make_shared<hozon::soc::gnss::GnssInfo>();
    if (gnss_proto == nullptr) {
        SENSOR_LOG_ERROR << "gnss_proto Allocate got nullptr!";
        return nullptr;
    }
    // receive someip gnss struct data to gnss proto
    auto pb_header = gnss_proto->mutable_header();
    pb_header->set_seq(data->header.seq);
    pb_header->set_frame_id(std::string(data->header.frameId.begin(), data->header.frameId.end()));
    struct timespec time;
    if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
        SENSOR_LOG_ERROR << "clock_gettime fail ";
    }

    // data->header.stamp.sec = time.tv_sec;
    // data->header.stamp.nsec = time.tv_nsec;

    pb_header->set_publish_stamp(time.tv_sec + time.tv_nsec / (1000.0 * 1000.0 * 1000.0));
    pb_header->set_gnss_stamp(data->header.gnssStamp.sec + data->header.gnssStamp.nsec / (1000.0 * 1000.0 * 1000.0));
    pb_header->mutable_sensor_stamp()->set_gnss_stamp(HafTimeConverStamp(data->header.stamp));
    gnss_proto->set_gnss_stamp_sec(data->header.gnssStamp.sec);
    gnss_proto->set_gps_week(data->gpsWeek);
    gnss_proto->set_gps_sec(data->gpsSec);

    auto pb_gnss_pos = gnss_proto->mutable_gnss_pos();
    pb_gnss_pos->set_pos_type(data->gnss_pos.posType);
    pb_gnss_pos->set_latitude(data->gnss_pos.latitude);
    pb_gnss_pos->set_longitude(data->gnss_pos.longitude);
    pb_gnss_pos->set_undulation(data->gnss_pos.undulation);
    pb_gnss_pos->set_altitude(data->gnss_pos.altitude);
    pb_gnss_pos->set_lat_std(data->gnss_pos.latStd);
    pb_gnss_pos->set_lon_std(data->gnss_pos.lonStd);
    pb_gnss_pos->set_hgt_std(data->gnss_pos.hgtStd);
    pb_gnss_pos->set_svs(data->gnss_pos.svs);
    pb_gnss_pos->set_solnsvs(data->gnss_pos.solnSVs);
    pb_gnss_pos->set_diff_age(data->gnss_pos.diffAge);
    pb_gnss_pos->set_hdop(data->gnss_pos.hdop);
    pb_gnss_pos->set_vdop(data->gnss_pos.vdop);
    pb_gnss_pos->set_pdop(data->gnss_pos.pdop);
    pb_gnss_pos->set_gdop(data->gnss_pos.gdop);
    pb_gnss_pos->set_tdop(data->gnss_pos.tdop);

    auto pb_gnss_vel = gnss_proto->mutable_gnss_vel();
    pb_gnss_vel->set_sol_status(data->gnss_vel.solStatus);
    pb_gnss_vel->set_hor_spd(data->gnss_vel.horSpd);
    pb_gnss_vel->set_trk_gnd(data->gnss_vel.trkGnd);
    pb_gnss_vel->set_vel_x(data->gnss_vel.velX);
    pb_gnss_vel->set_vel_y(data->gnss_vel.velY);
    pb_gnss_vel->set_vel_z(data->gnss_vel.velZ);
    pb_gnss_vel->set_vel_x_std(data->gnss_vel.velXstd);
    pb_gnss_vel->set_vel_y_std(data->gnss_vel.velYstd);
    pb_gnss_vel->set_vel_z_std(data->gnss_vel.velZstd);

    auto pb_gnss_heading = gnss_proto->mutable_gnss_heading();
    pb_gnss_heading->set_svs(data->gnss_heading.svs);
    pb_gnss_heading->set_soln_svs(data->gnss_heading.solnSVs);
    pb_gnss_heading->set_pos_type(data->gnss_heading.posType);
    pb_gnss_heading->set_length(data->gnss_heading.length);
    pb_gnss_heading->set_heading(data->gnss_heading.heading);
    pb_gnss_heading->set_pitch(data->gnss_heading.pitch);
    pb_gnss_heading->set_hdg_std(data->gnss_heading.hdgStd);
    pb_gnss_heading->set_pitch_std(data->gnss_heading.pitchStd);

    if(_gnss_pub_last_seq && (gnss_proto->mutable_header()->seq() > _gnss_pub_last_seq) 
        && ((gnss_proto->mutable_header()->seq() - _gnss_pub_last_seq) != 1)) {
        SENSOR_LOG_WARN << "gnss info lost data. receive seq: " << gnss_proto->mutable_header()->seq() \
            << " last seq : "  << _gnss_pub_last_seq  \
            << " seq diff : " << (gnss_proto->mutable_header()->seq() - _gnss_pub_last_seq)
            << " interval : " << (gnss_proto->mutable_header()->publish_stamp() \
                - gnss_proto->mutable_header()->mutable_sensor_stamp()->gnss_stamp()) << " s";
    } else if ((gnss_proto->mutable_header()->publish_stamp() 
            - gnss_proto->mutable_header()->mutable_sensor_stamp()->gnss_stamp()) > 0.1f) {  // 100ms
        SENSOR_LOG_WARN << "chassis info link latency : " << (gnss_proto->mutable_header()->publish_stamp() \
                - gnss_proto->mutable_header()->mutable_sensor_stamp()->gnss_stamp()) << " s";
    }
    _gnss_pub_last_seq = gnss_proto->mutable_header()->seq();

    return gnss_proto;
}

}  // namespace sensor
}  // namespace netaos
}  // namespace hozon
