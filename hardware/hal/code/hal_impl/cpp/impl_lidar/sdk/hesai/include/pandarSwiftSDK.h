/* -*- mode: C++ -*- */
/*
 *  Copyright (C) 2009, 2010 Austin Robot Technology, Jack O'Quin
 *  Copyright (C) 2011 Jesse Vera
 *  Copyright (C) 2012 Austin Robot Technology, Jack O'Quin
 *  Copyright (c) 2017 Hesai Photonics Technology, Yang Sheng
 *  Copyright (c) 2020 Hesai Photonics Technology, Lingwen Fang
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

/** @file

    This class Pandar128SDK raw Pandar128 3D LIDAR packets to PointCloud2.

*/

#ifndef _PANDAR_POINTCLOUD_PANDAR128SDK_H_
#define _PANDAR_POINTCLOUD_PANDAR128SDK_H_ 1

#include <pthread.h>
#include <semaphore.h>
#include <boost/atomic.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/thread.hpp>
#include <fstream>

#include "fault_message.h"
#include "laser_ts.h"
#include "pandarSwiftDriver.h"
#include "point_types.h"
#include "tcp_command_client.h"

#include "hal_lidar.hpp"

#ifndef CIRCLE
#define CIRCLE (36000)
#endif

#define PANDARSDK_TCP_COMMAND_PORT (9347)
#define LIDAR_DATA_TYPE "lidar"
#define LIDAR_ANGLE_SIZE_5 (5)
#define LIDAR_ANGLE_SIZE_3_75 (float(3.75))
#define LIDAR_ANGLE_SIZE_7_5 (float(7.5))
#define LIDAR_ANGLE_SIZE_12_5 (float(12.5))
#define LIDAR_ANGLE_SIZE_18_75 (float(18.75))
#define LIDAR_AZIMUTH_UNIT (256)
#define LIDAR_ANGLE_SIZE_15 (15)
#define LIDAR_ANGLE_SIZE_10 (10)
#define LIDAR_ANGLE_SIZE_18 (18)
#define LIDAR_ANGLE_SIZE_20 (20)
#define LIDAR_ANGLE_SIZE_40 (40)
#define LIDAR_RETURN_BLOCK_SIZE_1 (1)
#define LIDAR_RETURN_BLOCK_SIZE_2 (2)

#define GPS_PACKET_SIZE (512)
#define GPS_PACKET_FLAG_SIZE (2)
#define GPS_PACKET_YEAR_SIZE (2)
#define GPS_PACKET_MONTH_SIZE (2)
#define GPS_PACKET_DAY_SIZE (2)
#define GPS_PACKET_HOUR_SIZE (2)
#define GPS_PACKET_MINUTE_SIZE (2)
#define GPS_PACKET_SECOND_SIZE (2)
#define GPS_ITEM_NUM (7)

#define PANDAR128_LASER_NUM (128)
#define PANDAR64S_LASER_NUM (64)
#define PANDAR40S_LASER_NUM (40)
#define PANDAR80_LASER_NUM (80)
#define PANDAR128_BLOCK_NUM (2)
#define MAX_BLOCK_NUM (8)
#define PANDAR128_DISTANCE_UNIT (0.004)
#define PANDAR128_SOB_SIZE (2)
#define PANDAR128_VERSION_MAJOR_SIZE (1)
#define PANDAR128_VERSION_MINOR_SIZE (1)
#define PANDAR128_HEAD_RESERVED1_SIZE (2)
#define PANDAR128_LASER_NUM_SIZE (1)
#define PANDAR128_BLOCK_NUM_SIZE (1)
#define PANDAR128_ECHO_COUNT_SIZE (1)
#define PANDAR128_ECHO_NUM_SIZE (1)
#define PANDAR128_HEAD_RESERVED2_SIZE (2)
#define PANDAR128_HEAD_SIZE                                       \
  (PANDAR128_SOB_SIZE + PANDAR128_VERSION_MAJOR_SIZE +            \
   PANDAR128_VERSION_MINOR_SIZE + PANDAR128_HEAD_RESERVED1_SIZE + \
   PANDAR128_LASER_NUM_SIZE + PANDAR128_BLOCK_NUM_SIZE +          \
   PANDAR128_ECHO_COUNT_SIZE + PANDAR128_ECHO_NUM_SIZE +          \
   PANDAR128_HEAD_RESERVED2_SIZE)
#define PANDAR128_AZIMUTH_SIZE (2)
#define DISTANCE_SIZE (2)
#define INTENSITY_SIZE (1)
#define CONFIDENCE_SIZE (1)
#define PANDAR128_UNIT_WITHOUT_CONFIDENCE_SIZE (DISTANCE_SIZE + INTENSITY_SIZE)
#define PANDAR128_UNIT_WITH_CONFIDENCE_SIZE \
  (DISTANCE_SIZE + INTENSITY_SIZE + CONFIDENCE_SIZE)
#define PANDAR128_BLOCK_SIZE                                      \
  (PANDAR128_UNIT_WITHOUT_CONFIDENCE_SIZE * PANDAR128_LASER_NUM + \
   PANDAR128_AZIMUTH_SIZE)
#define PANDAR128_TAIL_RESERVED1_SIZE (3)
#define PANDAR128_TAIL_RESERVED2_SIZE (3)
#define PANDAR128_SHUTDOWN_FLAG_SIZE (1)
#define PANDAR128_TAIL_RESERVED3_SIZE (3)
#define PANDAR128_MOTOR_SPEED_SIZE (2)
#define PANDAR128_TS_SIZE (4)
#define PANDAR128_RETURN_MODE_SIZE (1)
#define PANDAR128_FACTORY_INFO (1)
#define PANDAR128_UTC_SIZE (6)
#define PANDAR128_TAIL_SIZE                                        \
  (PANDAR128_TAIL_RESERVED1_SIZE + PANDAR128_TAIL_RESERVED2_SIZE + \
   PANDAR128_SHUTDOWN_FLAG_SIZE + PANDAR128_TAIL_RESERVED3_SIZE +  \
   PANDAR128_MOTOR_SPEED_SIZE + PANDAR128_TS_SIZE +                \
   PANDAR128_RETURN_MODE_SIZE + PANDAR128_FACTORY_INFO + PANDAR128_UTC_SIZE)
// #define PANDAR128_PACKET_SIZE                                         \
//   (PANDAR128_HEAD_SIZE + PANDAR128_BLOCK_SIZE * PANDAR128_BLOCK_NUM + \
//    PANDAR128_TAIL_SIZE)
#define PANDAR128_SEQ_NUM_SIZE (4)
// #define PANDAR128_PACKET_SEQ_NUM_SIZE \
//   (PANDAR128_PACKET_SIZE + PANDAR128_SEQ_NUM_SIZE)
#define PANDAR128_WITHOUT_CONF_UNIT_SIZE (DISTANCE_SIZE + INTENSITY_SIZE)

#define TASKFLOW_STEP_SIZE (200)
#define PANDAR128_CRC_SIZE (4)
#define PANDAR128_FUNCTION_SAFETY_SIZE (17)

#define CIRCLE_ANGLE (36000)
#define MAX_AZI_LEN (36000 * 256)
#define MOTOR_SPEED_600 (600)
#define MOTOR_SPEED_150 (150)
#define MOTOR_SPEED_750 (750)
#define MOTOR_SPEED_500 (500)
#define MOTOR_SPEED_400 (400)
#define MOTOR_SPEED_200 (200)
#define MOTOR_SPEED_900 (900)
#define MOTOR_SPEED_300 (300)

/************************************* AT 128
 * *********************************************/
#define PANDAR_AT128_SOB_SIZE (2)
#define PANDAR_AT128_VERSION_MAJOR_SIZE (1)
#define PANDAR_AT128_VERSION_MINOR_SIZE (1)
#define PANDAR_AT128_HEAD_RESERVED1_SIZE (2)
#define PANDAR_AT128_LASER_NUM_SIZE (1)
#define PANDAR_AT128_BLOCK_NUM_SIZE (1)
#define PANDAR_AT128_DISTANCE_UNIT_SIZE (1)
#define PANDAR_AT128_ECHO_COUNT_SIZE (1)
#define PANDAR_AT128_ECHO_NUM_SIZE (1)
#define PANDAR_AT128_HEAD_RESERVED2_SIZE (1)
#define PANDAR_AT128_HEAD_SIZE                                          \
  (PANDAR_AT128_SOB_SIZE + PANDAR_AT128_VERSION_MAJOR_SIZE +            \
   PANDAR_AT128_VERSION_MINOR_SIZE + PANDAR_AT128_HEAD_RESERVED1_SIZE + \
   PANDAR_AT128_LASER_NUM_SIZE + PANDAR_AT128_BLOCK_NUM_SIZE +          \
   PANDAR_AT128_ECHO_COUNT_SIZE + PANDAR_AT128_ECHO_NUM_SIZE +          \
   PANDAR_AT128_HEAD_RESERVED2_SIZE + PANDAR_AT128_DISTANCE_UNIT_SIZE)
#define PANDAR_AT128_AZIMUTH_SIZE (2)
#define PANDAR_AT128_FINE_AZIMUTH_SIZE (1)
#define DISTANCE_SIZE (2)
#define INTENSITY_SIZE (1)
#define CONFIDENCE_SIZE (1)
#define PANDAR_AT128_UNIT_WITHOUT_CONFIDENCE_SIZE \
  (DISTANCE_SIZE + INTENSITY_SIZE)
#define PANDAR_AT128_UNIT_WITH_CONFIDENCE_SIZE \
  (DISTANCE_SIZE + INTENSITY_SIZE + CONFIDENCE_SIZE)
#define PANDAR_AT128_BLOCK_SIZE                                         \
  (PANDAR_AT128_UNIT_WITHOUT_CONFIDENCE_SIZE * PANDAR_AT128_LASER_NUM + \
   PANDAR_AT128_AZIMUTH_SIZE)
#define PANDAR_AT128_TAIL_RESERVED1_SIZE (3)
#define PANDAR_AT128_TAIL_RESERVED2_SIZE (3)
#define PANDAR_AT128_SHUTDOWN_FLAG_SIZE (1)
#define PANDAR_AT128_TAIL_RESERVED3_SIZE (3)
#define PANDAR_AT128_TAIL_RESERVED4_SIZE (8)
#define PANDAR_AT128_MOTOR_SPEED_SIZE (2)
#define PANDAR_AT128_TS_SIZE (4)
#define PANDAR_AT128_RETURN_MODE_SIZE (1)
#define PANDAR_AT128_FACTORY_INFO (1)
#define PANDAR_AT128_UTC_SIZE (6)
#define PANDAR_AT128_TAIL_SIZE                                           \
  (PANDAR_AT128_TAIL_RESERVED1_SIZE + PANDAR_AT128_TAIL_RESERVED2_SIZE + \
   PANDAR_AT128_SHUTDOWN_FLAG_SIZE + PANDAR_AT128_TAIL_RESERVED3_SIZE +  \
   PANDAR_AT128_MOTOR_SPEED_SIZE + PANDAR_AT128_TS_SIZE +                \
   PANDAR_AT128_RETURN_MODE_SIZE + PANDAR_AT128_FACTORY_INFO +           \
   PANDAR_AT128_UTC_SIZE)
#define PANDAR_AT128_PACKET_SIZE                                               \
  (PANDAR_AT128_HEAD_SIZE + PANDAR_AT128_BLOCK_SIZE * PANDAR_AT128_BLOCK_NUM + \
   PANDAR_AT128_TAIL_SIZE)
#define PANDAR_AT128_SEQ_NUM_SIZE (4)
#define PANDAR_AT128_PACKET_SEQ_NUM_SIZE \
  (PANDAR_AT128_PACKET_SIZE + PANDAR_AT128_SEQ_NUM_SIZE)
#define PANDAR_AT128_WITHOUT_CONF_UNIT_SIZE (DISTANCE_SIZE + INTENSITY_SIZE)
#define PANDAR_AT128_FRAME_ANGLE_SIZE (6250)
#define PANDAR_AT128_FRAME_BUFFER_SIZE (7500)
#define PANDAR_AT128_FRAME_ANGLE_INTERVAL_SIZE (5600)
#define PANDAR_AT128_EDGE_AZIMUTH_OFFSET (4500)
#define PANDAR_AT128_EDGE_AZIMUTH_SIZE (1200)
#define PANDAR_AT128_CRC_SIZE (4)
#define PANDAR_AT128_FUNCTION_SAFETY_SIZE (17)
#define PANDAR_AT128_SIGNATURE_SIZE (32)
#define MIN_POINT_NUM (30000)
#define MAX_POINT_NUM (360000)
#define TX_TDM_ID (25)
#define RX_TDM_ID (26)
#define MB_TDM_ID (27)
#define PB_TDM_ID (28)
#define PACKET_NUM_PER_FRAME (630)
/************************************* AT 128
 * *********************************************/

typedef struct __attribute__((__packed__)) Pandar128Unit_s {
  uint16_t u16Distance;
  uint8_t u8Intensity;
  // uint8_t  u8Confidence;
} Pandar128Unit;

typedef struct __attribute__((__packed__)) Pandar128Block_s {
  uint16_t fAzimuth;
  Pandar128Unit units[PANDAR128_LASER_NUM];
} Pandar128Block;

typedef struct Pandar128HeadVersion13_s {
  uint16_t u16Sob;
  uint8_t u8VersionMajor;
  uint8_t u8VersionMinor;
  uint8_t u8DistUnit;
  uint8_t u8Flags;
  uint8_t u8LaserNum;
  uint8_t u8BlockNum;
  uint8_t u8EchoCount;
  uint8_t u8EchoNum;
  uint16_t u16Reserve1;
} Pandar128HeadVersion13;

typedef struct Pandar128HeadVersion14_s {
  uint16_t u16Sob;
  uint8_t u8VersionMajor;
  uint8_t u8VersionMinor;
  uint16_t u16Reserve1;
  uint8_t u8LaserNum;
  uint8_t u8BlockNum;
  uint8_t u8EchoCount;
  uint8_t u8DistUnit;
  uint8_t u8EchoNum;
  uint8_t u8Flags;
  inline bool hasSeqNum() const { return u8Flags & 1; }
  inline bool hasImu() const { return u8Flags & 2; }
  inline bool hasFunctionSafety() const { return u8Flags & 4; }
  inline bool hasSignature() const { return u8Flags & 8; }
  inline bool hasConfidence() const { return u8Flags & 0x10; }

} Pandar128HeadVersion14;

typedef struct Pandar128TailVersion13_s {
  uint8_t nReserved1[3];
  uint8_t nReserved2[3];
  uint8_t nShutdownFlag;
  uint8_t nReserved3[3];
  uint16_t nMotorSpeed;
  uint32_t nTimestamp;
  uint8_t nReturnMode;
  uint8_t nFactoryInfo;
  uint8_t nUTCTime[6];
  uint32_t nSeqNum;
} Pandar128TailVersion13;

typedef struct Pandar128TailVersion14_s {
  uint8_t nReserved1[3];
  uint8_t nReserved2[3];
  uint8_t nReserved3[3];
  uint16_t nAzimuthFlag;
  uint8_t nShutdownFlag;
  uint8_t nReturnMode;
  uint16_t nMotorSpeed;
  uint8_t nUTCTime[6];
  uint32_t nTimestamp;
  uint8_t nFactoryInfo;
  uint32_t nSeqNum;
} Pandar128TailVersion14;

typedef struct __attribute__((__packed__)) Pandar128PacketVersion13_t {
  Pandar128HeadVersion13 head;
  Pandar128Block blocks[PANDAR128_BLOCK_NUM];
  Pandar128TailVersion13 tail;
} Pandar128PacketVersion13;

struct PandarGPS_s {
  uint16_t flag;
  uint16_t year;
  uint16_t month;
  uint16_t day;
  uint16_t second;
  uint16_t minute;
  uint16_t hour;
  uint32_t fineTime;
};

/************************************* AT 128
 * *********************************************/
typedef struct PandarAT128Head_s {
  uint16_t u16Sob;
  uint8_t u8VersionMajor;
  uint8_t u8VersionMinor;
  uint16_t u16Reserve1;
  uint8_t u8LaserNum;
  uint8_t u8BlockNum;
  uint8_t u8EchoCount;
  uint8_t u8DistUnit;
  uint8_t u8EchoNum;
  uint8_t u8Flags;
  inline bool hasSeqNum() const { return u8Flags & 1; }
  inline bool hasImu() const { return u8Flags & 2; }
  inline bool hasFunctionSafety() const { return u8Flags & 4; }
  inline bool hasSignature() const { return u8Flags & 8; }
  inline bool hasConfidence() const { return u8Flags & 0x10; }

} PandarAT128Head;

typedef struct PandarAT128TailVersion41_s {
  uint8_t nReserved1[3];
  uint8_t nReserved2[3];
  uint8_t nShutdownFlag;
  uint8_t nReserved3[3];
  uint16_t nMotorSpeed;
  uint32_t nTimestamp;
  uint8_t nReturnMode;
  uint8_t nFactoryInfo;
  uint8_t nUTCTime[6];
} PandarAT128TailVersion41;

typedef struct PandarAT128TailVersion43_s {
  uint8_t nReserved1[3];
  uint8_t nReserved2[3];
  uint8_t nShutdownFlag;
  uint8_t nReserved3[3];
  uint8_t nReserved4[8];
  int16_t nMotorSpeed;
  uint32_t nTimestamp;
  uint8_t nReturnMode;
  uint8_t nFactoryInfo;
  uint8_t nUTCTime[6];
} PandarAT128TailVersion43;

struct PandarATCorrectionsHeader {
  uint8_t delimiter[2];
  uint8_t version[2];
  uint8_t channel_number;
  uint8_t mirror_number;
  uint8_t frame_number;
  uint8_t frame_config[8];
  uint8_t resolution;
};
static_assert(sizeof(PandarATCorrectionsHeader) == 16);
#pragma pack(pop)

struct PandarATFrameInfo {
  uint32_t start_frame[8];
  uint32_t end_frame[8];
  int32_t azimuth[128];
  int32_t elevation[128];
  std::array<float, MAX_AZI_LEN> sin_map;
  std::array<float, MAX_AZI_LEN> cos_map;
};

struct PandarATCorrections {
 public:
  PandarATCorrectionsHeader header;
  uint16_t start_frame[8];
  uint16_t end_frame[8];
  int16_t azimuth[128];
  int16_t elevation[128];
  int8_t azimuth_offset[36000];
  int8_t elevation_offset[36000];
  uint8_t SHA256[32];
  PandarATFrameInfo l;  // V1.5
  std::array<float, MAX_AZI_LEN> sin_map;
  std::array<float, MAX_AZI_LEN> cos_map;
  PandarATCorrections() {
    for (int i = 0; i < MAX_AZI_LEN; ++i) {
      sin_map[i] = std::sin(2 * i * M_PI / MAX_AZI_LEN);
      cos_map[i] = std::cos(2 * i * M_PI / MAX_AZI_LEN);
    }
  }
  static const int STEP = 200;
  int8_t getAzimuthAdjust(uint8_t ch, uint16_t azi) const {
    unsigned int i = std::floor(1.f * azi / STEP);
    unsigned int l = azi - i * STEP;
    float k = 1.f * l / STEP;
    return round((1 - k) * azimuth_offset[ch * 180 + i] +
                 k * azimuth_offset[ch * 180 + i + 1]);
  }
  int8_t getElevationAdjust(uint8_t ch, uint16_t azi) const {
    unsigned int i = std::floor(1.f * azi / STEP);
    unsigned int l = azi - i * STEP;
    float k = 1.f * l / STEP;
    return round((1 - k) * elevation_offset[ch * 180 + i] +
                 k * elevation_offset[ch * 180 + i + 1]);
  }
  static const int STEP3 = 200 * 256;
  int8_t getAzimuthAdjustV3(uint8_t ch, uint32_t azi) const {
    unsigned int i = std::floor(1.f * azi / STEP3);
    unsigned int l = azi - i * STEP3;
    float k = 1.f * l / STEP3;
    return round((1 - k) * azimuth_offset[ch * 180 + i] +
                 k * azimuth_offset[ch * 180 + i + 1]);
  }
  int8_t getElevationAdjustV3(uint8_t ch, uint32_t azi) const {
    unsigned int i = std::floor(1.f * azi / STEP3);
    unsigned int l = azi - i * STEP3;
    float k = 1.f * l / STEP3;
    return round((1 - k) * elevation_offset[ch * 180 + i] +
                 k * elevation_offset[ch * 180 + i + 1]);
  }
};

/************************************* AT 128
 * *********************************************/

typedef std::array<PandarPacket, 36000> PktArray;

typedef struct PacketsBuffer_s {
  PktArray m_buffers{};
  PktArray::iterator m_iterPush;
  PktArray::iterator m_iterTaskBegin;
  PktArray::iterator m_iterTaskEnd;
  int m_stepSize;
  bool m_startFlag;
  int m_pcapFlag;
  inline PacketsBuffer_s() {
    m_stepSize = TASKFLOW_STEP_SIZE;
    m_iterPush = m_buffers.begin();
    m_iterTaskBegin = m_buffers.begin();
    m_iterTaskEnd = m_iterTaskBegin + m_stepSize;
    m_startFlag = false;
    m_pcapFlag = 0;
  }

  inline int push_back(PandarPacket pkt) {
    if (!m_startFlag) {
      *(m_iterPush++) = pkt;
      m_startFlag = true;
      return 1;
    } else {
      // static bool lastOverflowed = false;
      // if(m_iterPush == m_iterTaskBegin) {
      // 	static uint32_t tmp = m_iterTaskBegin - m_buffers.begin();
      // 	if(m_iterTaskBegin - m_buffers.begin() != tmp) {
      // 		printf("buffer don't have space!,%d\n",m_iterTaskBegin -
      // m_buffers.begin()); 		tmp = m_iterTaskBegin - m_buffers.begin();
      // 	}
      // 	lastOverflowed = true;
      // 	return 0;
      // }
      // if(lastOverflowed) {
      // 	lastOverflowed = false;
      // 	printf("buffer recovered\n");
      // }
      *(m_iterPush++) = pkt;
      if (m_buffers.end() == m_iterPush) {
        m_iterPush = m_buffers.begin();
      }
      return 1;
    }
  }

  inline bool hasEnoughPackets() {
    // printf("%d %d %d\n",m_iterPush - m_buffers.begin(), m_iterTaskBegin -
    // m_buffers.begin(), m_iterTaskEnd - m_buffers.begin());
    return (m_iterPush > m_buffers.begin()) &&
            (((m_iterPush - m_pcapFlag) > m_iterTaskBegin &&
              (m_iterPush - m_pcapFlag) > m_iterTaskEnd) ||
             ((m_iterPush - m_pcapFlag) < m_iterTaskBegin &&
              (m_iterPush - m_pcapFlag) < m_iterTaskEnd));
  }
  inline bool empty() {
    // static int count = 0;
    // if((abs(m_iterPush - m_iterTaskBegin) <= 1 || abs(m_iterTaskEnd -
    // m_iterTaskBegin) <= 1)){
    //   if(count > 0){
    //     count = 0;
    //     return true;
    //   }
    //   else{
    //     count++;
    //     return false;
    //   }
    // }
    // return false;
    return (abs(m_iterPush - m_iterTaskBegin) <= m_pcapFlag ||
            abs(m_iterTaskEnd - m_iterTaskBegin) <= 1 );
  }

  inline PktArray::iterator getTaskBegin() { return m_iterTaskBegin; }
  inline PktArray::iterator getTaskEnd() { return m_iterTaskEnd; }
  inline void moveTaskEnd(PktArray::iterator iter) {
    m_iterTaskEnd = iter;
    // printf("push: %d, begin: %d, end:
    // %d\n",m_iterPush-m_buffers.begin(),m_iterTaskBegin-m_buffers.begin(),m_iterTaskEnd-m_buffers.begin());
  }
  inline void creatNewTask() {
    if (m_buffers.end() == m_iterTaskEnd) {
      m_iterTaskBegin = m_buffers.begin();
      m_iterTaskEnd = m_iterTaskBegin + m_stepSize;
    } else if ((m_buffers.end() - m_iterTaskEnd) < m_stepSize) {
      m_iterTaskBegin = m_iterTaskEnd;
      m_iterTaskEnd = m_buffers.end();
    } else {
      // printf("add step\n");
      m_iterTaskBegin = m_iterTaskEnd;
      m_iterTaskEnd = m_iterTaskBegin + m_stepSize;
    }
  }
} PacketsBuffer;

typedef hal::lidar::PointXYZIT PPoint;
typedef std::vector<PPoint> PPointCloud;
typedef struct RedundantPoint_s {
  int index;
  PPoint point;
} RedundantPoint;

class PandarSwiftSDK {
 public:
  /**
   * @brief Constructor
   * @param deviceipaddr  	  The ip of the device
   *        deviceipaddr  	  The ip of the host pc
   *        lidarport 		  The port number of lidar data
   *        gpsport   		  The port number of gps data
   *        frameid           The id of the point cloud data published to ROS
   *        correctionfile    The correction file path
   *        firtimeflie       The firtime flie path
   *        pcapfile          The pcap flie path
   *        pclcallback       The callback of PCL data structure
   *        rawcallback       The callback of raw data structure
   *        gpscallback       The callback of GPS structure
   *        certFile          Represents the path of the user's certificate
   *        privateKeyFile    Represents the path of the user's private key
   *        caFile            Represents the path of the root certificate
   *        start_angle       The start angle of every point cloud
   *                          should be <real angle> * 100.
   *        timezone          The timezone of local
   *        publishmode       The mode of publish
   *        datatype          The model of input data
   */
  PandarSwiftSDK(
      std::string deviceipaddr, std::string hostipaddr, uint16_t lidarport, uint16_t gpsport,
      std::string frameid, std::string correctionfile, std::string firtimeflie,
      std::string pcapfile,
      boost::function<void(boost::shared_ptr<PPointCloud>, double)> pclcallback,
      boost::function<void(PandarPacketsArray *)> rawcallback,
      boost::function<void(double)> gpscallback,
      boost::function<void(AT128FaultMessageInfo &)> faultmessagecallback,
      std::string certFile, std::string privateKeyFile, std::string caFile,
      int startangle, int timezone, int viewMode, std::string publishmode,
      std::string multicast_ip,
      std::map<std::string, int32_t> threadPriority = {},
      std::string datatype = LIDAR_DATA_TYPE);
  PandarSwiftSDK(
      std::string deviceipaddr, std::string hostipaddr, uint16_t lidarport, uint16_t gpsport,
      std::string frameid, std::string correctionfile, std::string firtimeflie,
      std::string pcapfile,
      boost::function<void(boost::shared_ptr<PPointCloud>, double, uint64_t, hal::lidar::ConfigInfo)> pclcallback,
      boost::function<void(PandarPacketsArray *)> rawcallback,
      boost::function<void(double)> gpscallback,
      boost::function<void(AT128FaultMessageInfo &)> faultmessagecallback,
      std::string certFile, std::string privateKeyFile, std::string caFile,
      int startangle, int timezone, int viewMode, std::string publishmode,
      std::string multicast_ip,
      hal::lidar::ConfigInfo config_info,
      std::map<std::string, int32_t> threadPriority = {},
      std::string datatype = LIDAR_DATA_TYPE);
  ~PandarSwiftSDK() {
    if (m_pTcpCommandClient) {
      delete m_pTcpCommandClient;
      m_pTcpCommandClient = NULL;
    }
  }
  static const char *m_sPtcsModeSetFilePath;
  std::map<std::string, int32_t> m_configMap;
  void driverReadThread();
  void publishRawDataThread();
  void processGps(PandarGPS *gpsMsg);
  void pushLiDARData(PandarPacket packet);
  void processFaultMessage(PandarPacket &packet);
  int processLiDARData();
  void publishPoints();
  void start();
  void stop();
  bool GetIsReadPcapOver();
  void SetIsReadPcapOver(bool enable);
  void setIsSocketTimeout(bool isSocketTimeout);
  bool getIsSocketTimeout();
  bool setStandbyLidarMode();
  void setTimeStampNum(int num);
  bool setNormalLidarMode();
  bool setLidarReturnMode(
      uint8_t mode);  // mode: 0-last return, 1-strongest return, 2-dual return
  bool getLidarReturnMode(uint8_t &mode);
  bool setLidarSpinRate(uint16_t spinRate);  // spinRate: 200 300 400 500
  bool getLidarSpinRate(uint16_t &spinRate);
  bool setLidarLensHeatSwitch(
      uint8_t heatSwitch);  // heatSwitch: 0-close, 1-open
  bool getLidarLensHeatSwitch(uint8_t &heatSwitch);
  bool setPtcsLidarMode();
  bool setPtcLidarMode();
  int getPtcsLidarMode();
  float getTxTemperature();
  float getRxTemperature();
  float getPbTemperature();
  float getMbTemperature();

 private:
  int parseData(Pandar128PacketVersion13 &pkt, const uint8_t *buf,
                const int len);
  void calcPointXYZIT(PandarPacket &pkt, int cursor);
  void doTaskFlow(int cursor);
  void loadOffsetFile(std::string file);
  bool loadCorrectionFile();
  int loadCorrectionString(char *correctionstring);
  int checkLiadaMode();
  void init();
  void changeAngleSize();
  void changeReturnBlockSize();
  void moveTaskEndToStartAngle();
  void checkClockwise(int16_t lidarmotorspeed);
  bool isNeedPublish();
  int calculatePointIndex(uint16_t azimuth, int blockId, int laserId,
                          int field);
  int calculatePointBufferSize();
  void SetEnvironmentVariableTZ();
  void updateMoniterInfo(int id, uint16_t data);

  pthread_mutex_t m_RedundantPointLock;
  boost::shared_ptr<PandarSwiftDriver> m_spPandarDriver;
  LasersTSOffset m_objLaserOffset;
  boost::function<void(boost::shared_ptr<PPointCloud> cld, double timestamp)>
      m_funcPclCallback;
  boost::function<void(boost::shared_ptr<PPointCloud> cld, double timestamp, uint64_t seq, hal::lidar::ConfigInfo config)> 
        m_funcPointcloudCallback;
  boost::function<void(double timestamp)> m_funcGpsCallback;
  boost::function<void(AT128FaultMessageInfo &faultMessage)>
      m_funcFaultMessageCallback;
  std::array<boost::shared_ptr<PPointCloud>, 2> m_OutMsgArray;
  boost::shared_ptr<PPointCloud> m_PublishMsgArray;
  std::vector<RedundantPoint> m_RedundantPointBuffer;
  PacketsBuffer m_PacketsBuffer;
  double m_dTimestamp;
  int m_iLidarRotationStartAngle;
  int m_iTimeZoneSecond;
  float m_fCosAllAngle[CIRCLE];
  float m_fSinAllAngle[CIRCLE];
  float m_fElevAngle[PANDAR128_LASER_NUM];
  float m_fHorizatalAzimuth[PANDAR128_LASER_NUM];
  std::string m_sFrameId;
  std::string m_sLidarFiretimeFile;
  std::string m_sLidarCorrectionFile;
  std::string m_sPublishmodel;
  boost::thread *m_driverReadThread;
  boost::thread *m_processLiDARDataThread;
  boost::thread *m_publishPointsThread;
  boost::thread *m_publishRawDataThread;
  int m_iWorkMode;
  int m_iReturnMode;
  int m_iMotorSpeed;
  int m_iLaserNum;
  float m_iAngleSize;  // 10->0.1degree,20->0.2degree
  int m_iReturnBlockSize;
  bool m_bPublishPointsFlag;
  int m_iPublishPointsIndex;
  void *m_pTcpCommandClient;
  std::string m_sDeviceIpAddr;
  std::string m_sPcapFile;
  std::string m_sSdkVersion;
  uint8_t m_u8UdpVersionMajor;
  uint8_t m_u8UdpVersionMinor;
  int m_iFirstAzimuthIndex;
  int m_iLastAzimuthIndex;
  bool m_bClockwise;
  PandarATCorrections m_PandarAT_corrections;
  int m_iViewMode;
  bool m_bIsSocketTimeout;
  int m_iField;
  int m_iEdgeAzimuthSize;
  std::string m_sDatatype;
  bool m_bIsReadPcapOver;
  double m_dAzimuthInterval;
  double m_dAzimuthRange;
  std::string m_sCaFilePath;
  std::string m_sCertFilePath;
  std::string m_sPrivateKeyFilePath;
  float m_fTxTemperature;
  float m_fRxTemperature;
  float m_fMbTemperature;
  float m_fPbTemperature;
  bool m_bGetCorrectionSuccess;
  int m_iGetCorrectionCount;
  int m_iWithoutDataWarningTime;
  uint32_t m_u32SequenceNum;
  int m_iLastPushIndex;
  uint32_t m_u32LastTaskEndAzimuth;
  bool m_bIsSwitchFrameFail;
  double m_dSystemTime;
  hal::lidar::ConfigInfo m_ConfigInfo;
};

#endif  // _PANDAR_POINTCLOUD_Pandar128SDK_H_
