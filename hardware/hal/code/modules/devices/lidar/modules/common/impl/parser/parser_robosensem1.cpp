#include "lidar/modules/common/impl/parser/parser_base.h"

RobosenseM1Parser::RobosenseM1Parser()
{
}

RobosenseM1Parser::~RobosenseM1Parser()
{
}

bool RobosenseM1Parser::init(const LidarConfig &config)
{
    config_ = config;
    return true;
}

void RobosenseM1Parser::parse(const Scan &scan, hw_lidar_pointcloud_XYZIT *points)
{
    points_size_ = 0;
    int packets_per_frame = config_.packets_per_frame;
    for (int i = 0; i < packets_per_frame; i++)
    {
        unpack(scan.packets[i], points);
    }
}

void RobosenseM1Parser::unpack(const Scan::Packet &packet, hw_lidar_pointcloud_XYZIT *points)
{
    RobosenseM1UdpPacket robosense_m1_udp_packet;
    memset(&robosense_m1_udp_packet, 0, sizeof(RobosenseM1UdpPacket));
    memcpy(&robosense_m1_udp_packet, &packet.data[0], sizeof(RobosenseM1UdpPacket));

    int64_t timestamp_sec = ((int64_t)robosense_m1_udp_packet.header.timestamp[0] << 40) +
                            ((int64_t)robosense_m1_udp_packet.header.timestamp[1] << 32) +
                            (robosense_m1_udp_packet.header.timestamp[2] << 24) +
                            (robosense_m1_udp_packet.header.timestamp[3] << 16) +
                            (robosense_m1_udp_packet.header.timestamp[4] << 8) +
                            robosense_m1_udp_packet.header.timestamp[5];
    int timestamp_usec = (robosense_m1_udp_packet.header.timestamp[6] << 24) +
                         (robosense_m1_udp_packet.header.timestamp[7] << 16) +
                         (robosense_m1_udp_packet.header.timestamp[8] << 8) +
                         robosense_m1_udp_packet.header.timestamp[9];
    int64_t timestamp_udp = timestamp_sec * 1e9 + timestamp_usec * 1e3; // ns

    for (int i = 0; i < 25; i++) // every udp packet has 25 blocks
    {
        RobosenseM1UdpPacketDataBlock block = robosense_m1_udp_packet.block[i];
        int64_t timestamp_block = timestamp_udp + ((int)block.time_offset * 1e3);

        for (int j = 0; j < 5; j++) // every block has 5 channels
        {
            RobosenseM1UdpPacketDataBlockChannel channel = block.channel[j];

            float radius = ((channel.radius[0] << 8) + channel.radius[1]) * 0.005;                   // m
            float elevation = (((channel.elevation[0] << 8) + channel.elevation[1]) - 32768) * 0.01; // degree
            float azimuth = (((channel.azimuth[0] << 8) + channel.azimuth[1]) - 32768) * 0.01;       // degree

            float elevation_radian = elevation * M_PI / 180.0; // rad
            float azimuth_radian = azimuth * M_PI / 180.0;     // rad

            if (points_size_ >= config_.points_per_frame)
            {
                HW_LIDAR_LOG_ERR("lidar%d points size: %d > %d!\n", 
                                 config_.index, 
                                 points_size_, 
                                 config_.points_per_frame);
                continue;
            }
            points[points_size_].x = radius * cos(elevation_radian) * cos(azimuth_radian);
            points[points_size_].y = radius * cos(elevation_radian) * sin(azimuth_radian);
            points[points_size_].z = radius * sin(elevation_radian);
            points[points_size_].intensity = channel.intensity;
            points[points_size_].timestamp = timestamp_block;
            points_size_++;
        }
    }
}
