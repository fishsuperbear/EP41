#ifndef LIDAR_TYPES_H
#define LIDAR_TYPES_H

#include <vector>
#include "hw_lidar.h"
#include "lidar/modules/common/impl/utils/blocking_queue.h"

struct Scan;

typedef BlockingQueue<Scan> ScanCache;

#pragma pack(1)

struct LidarConfig
{
    int index;
    HW_LIDAR_MODEL lidar_model;
    int port;
    int packets_per_frame; // 10 frame per second
    int points_per_frame; // pointcloud number per frame
};

struct Scan
{
    struct Packet
    {
        uint8_t data[1210];
    };

    double timestamp;
    std::vector<Packet> packets;
};

struct PointcloudXYZIT
{
    float x;
    float y;
    float z;
    unsigned int intensity; // 0-255
    int64_t timestamp;      // ns
};

struct RobosenseM1UdpPacketDataHeader
{
    uint8_t pkt_header[4];     // 识别头为 0x55，0xaa，0x5a，0xa5
    uint16_t pkt_psn;          // 包序列号，表示包计数，循环计数，从每帧数据的起点的包计数为1，每帧数据的最后一个点的包计数为最大值
    uint16_t protocol_version; // 表示UDP通信协议的版本号
    uint8_t wave_mode;         // 回波模式标志位
    uint8_t time_sync_mode;    // 时间同步模式
    uint8_t timestamp[10];     // 高6Bytes为秒位，低4Bytes为微秒位
    uint8_t reserved[10];      // 预留
    uint8_t lidar_type;        // 雷达类型标志位，默认值0x10
    uint8_t mems_tmp;          // mems温度,Temp=mems_tmp-80;即原始值0代表-80度
};

struct RobosenseM1UdpPacketDataBlockChannel
{
    uint8_t radius[2];    // 极坐标系下，通道的径向点距离值，距离解析分辨率5mm
    uint8_t elevation[2]; // 极坐标系下，通道的点垂直夹角，分辨率0.01°
    uint8_t azimuth[2];   // 极坐标系下，通道的点水平夹角，分辨率0.01°
    uint8_t intensity;    // 点反射强度值，取值范围0~255
    uint16_t resev;
};

struct RobosenseM1UdpPacketDataBlock
{
    uint8_t time_offset; // 该组 Block 里面所有的点相对于包的timestamp的时间偏移量，该组点的时间等于timestamp+time_offset
    uint8_t return_seq;  // 回波序列。单回波模式下，此标志位恒定为0
    RobosenseM1UdpPacketDataBlockChannel channel[5];
};

struct RobosenseM1UdpPacket
{
    RobosenseM1UdpPacketDataHeader header;
    RobosenseM1UdpPacketDataBlock block[25];
    uint8_t tail[3]; // 预留
};

#pragma pack()

#endif // LIDAR_TYPES_H