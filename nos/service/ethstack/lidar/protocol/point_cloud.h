#ifndef LIDAR_PROTOCAL_POINT_CLOUD_H
#define LIDAR_PROTOCAL_POINT_CLOUD_H
#include<string.h>
#include<vector>

namespace hozon {
namespace ethstack {
namespace lidar {


constexpr uint16_t LOCAL_POINT_CLOUD_PORT = 2368;
constexpr uint16_t REMOTE_POINT_CLOUD_PORT = 58005;

constexpr uint16_t LOCAL_FAULT_MESSAGE_PORT = 2369;
constexpr uint16_t REMOTE_FAULT_MESSAGE_PORT = 58003;

const constexpr char* POINT_CLOUD_MULTICAST_ADDRESS   = ("239.255.0.1");

const constexpr char* LOCAL_DEV_IP                    = ("172.16.80.11");
const constexpr char* LIDAR_DEV_ADDRESS               = ("172.16.80.20");

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

}  // namespace lidar
}  // namespace ethstack
}  // namespace hozon
#endif  // LIDAR_PROTOCAL_UNPACK_POINT_CLOUD_FRAME_H