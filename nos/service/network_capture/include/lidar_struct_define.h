/*
  * Copyright (C) 2020 HAW Hamburg
  *
  * This file is subject to the terms and conditions of the GNU Lesser
  * General Public License v2.1. See the file LICENSE in the top level
  * directory for more details.
  */

#ifndef LIDAR_STRUCT_DEFINE_H
#define LIDAR_STRUCT_DEFINE_H
#pragma once

#include <stdint.h>
#include <string>
#include <array>
#include <math.h>
#include <vector>
namespace hozon {
namespace netaos {
namespace network_capture {

#pragma pack(push, 1)

/* ******************************************************************************
                Protocal structure definition for point cloud
******************************************************************************
*/

//Hesai
struct DataHeader{
  uint8_t laser_num;
  uint8_t block_num;
  uint8_t first_block_return;
  uint8_t dis_unit;
  uint8_t return_num;
  uint8_t flags;
};

struct ChannelData {
  uint16_t distance;
  uint8_t reflectivity;
  uint8_t confidence;
};

struct DataBlock {
  uint16_t azimuth;
  uint8_t fine_azimuth;
  ChannelData channel_data[128];
};

struct DataBody{
  DataBlock data_block[2];
  uint32_t crc_32;
};

struct DataTail{
  uint8_t reserver1[6];
  uint8_t high_temperature_flag;
  uint8_t reserver2[11];
  int16_t motor_speed;
  uint32_t timestamp;
  uint8_t return_mode;
  uint8_t factory_info;
  uint8_t data_time[6];
  uint32_t sequence;
  uint32_t crc_32;
};


struct PointCloudFrameDEA {
  uint8_t packet_header[6];     
  DataHeader data_header;
  DataBody data_body;
  DataTail data_tail;
  uint8_t internet_safe[32];
};


struct SubPointCloudFrameDEA {     
  DataHeader data_header;
  DataBody data_body;
  DataTail data_tail;
};

//MDC: honon::sensor::PointCloudFrame 结构体
struct CommonTime {
    uint64_t sec;
    uint64_t nsec;
};

struct CommonHeader {
    uint32_t seq;
    std::string frameId;
    CommonTime stamp;
    CommonTime gnssStamp;
    // LatencyInfo latencyInfo;
};
struct PointField {
    double x;
    double y;
    double z;
    uint32_t time;
    double distance;
    double pitch;
    double yaw;
    uint32_t intensity;
    uint32_t ring;
    uint32_t block;
};

using PointFields = std::vector<PointField>;

struct LidarSN {
    std::string ecuSerialNumber;
};

struct LidarEolCalibStatus {
    uint8_t calib_status;
    float rotationX;
    float rotationY;
    float rotationZ;
    float rotationW;
    float translationX;
    float translationY;
    float translationZ;
};

struct PointCloudFrame {
    CommonHeader header;
    uint32_t isBigEndian;
    uint32_t height;
    uint32_t width;
    uint32_t pointStep;
    uint32_t rowStep;
    uint32_t isDense;
    PointFields data;
    LidarSN lidarSN;
    LidarEolCalibStatus eolCalibStatus;
};

#pragma pack(pop)

#define ANGLE_TABLE_BITS_ALL 0x03ff

#define MAX_AZI_LEN (36000 * 256)
#define M_PI 3.14159265358979323846 /* pi */

#pragma pack(push, 1)

struct AngleTable {
    uint32_t set_bits = 0;
    double vangle[96] = {0};
    double hangle[1500] = {0};
    double vangle_offset[4][1500] = {{0}};
    double hangle_offset[4][96] = {{0}};
};

struct PandarATCorrectionsHeader {
    uint8_t delimiter[2];
    uint8_t version[2];
    uint8_t channel_number;
    uint8_t mirror_number;
    uint8_t frame_number;
    uint8_t frame_config[8];
    uint8_t resolution;
};

struct PointXYZ {
    double x;
    double y;
    double z;
    uint8_t reflectivity;
    uint8_t confidence;
};

struct PointCloud {
    PointXYZ point[128];
    float vertical_angle[128];
    float code_wheel_angle;
};

struct PandarATCorrections {
    PandarATCorrectionsHeader header;
    uint32_t start_frame[3];
    uint32_t end_frame[3];
    int32_t azimuth[128];
    int32_t elevation[128];
    int8_t azimuth_offset[23040];
    int8_t elevation_offset[23040];
    uint8_t SHA256[32];

    std::array<float, MAX_AZI_LEN> sin_map;         //#define MAX_AZI_LEN (36000 * 256)
    std::array<float, MAX_AZI_LEN> cos_map;
    PandarATCorrections() {
        for (int i = 0; i < MAX_AZI_LEN; ++i) {
            sin_map[i] = std::sin(2 * i * M_PI / MAX_AZI_LEN);
            cos_map[i] = std::cos(2 * i * M_PI / MAX_AZI_LEN);
        }
    }

    static const int STEP3 = 200 * 256;
    int8_t getAzimuthAdjustV3(uint8_t ch, uint32_t azi) const {
        unsigned int i = std::floor(1.f * azi / STEP3);
        unsigned int l = azi - i * STEP3;
        float k = 1.f * l / STEP3;
        return round((1 - k) * azimuth_offset[ch * 180 + i] +
                    k * azimuth_offset[ch * 180 + i + 1]);
    }
    int8_t getElevationAdjustV3(uint8_t ch, uint32_t azi) const {           //获取垂直角修正量
        unsigned int i = std::floor(1.f * azi / STEP3);                       //计算azi / STEP3 的整数部分   2°一步
        unsigned int l = azi - i * STEP3;                                     //计算azi / STEP3 的余数部分
        float k = 1.f * l / STEP3;
        return round((1 - k) * elevation_offset[ch * 180 + i] +
                    k * elevation_offset[ch * 180 + i + 1]);
    }
    
};

#endif
}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon
