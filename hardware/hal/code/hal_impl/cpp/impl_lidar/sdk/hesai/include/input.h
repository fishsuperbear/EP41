/* -*- mode: C++ -*-
 *
 *  Copyright (c) 2020 Hesai Photonics Technology, Lingwen Fang
 *
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

/** @file
 *
 *  Pandar128 3D LIDAR data input classes
 *
 *    These classes provide raw Pandar128 LIDAR input packets from
 *    either a live socket interface or a previously-saved PCAP dump
 *    file.
 *
 *  Classes:
 *
 *     pandar::Input -- base class for accessing the data
 *                      independently of its source
 *
 *     pandar::InputSocket -- derived class reads live data from the
 *                      device via a UDP socket
 *
 *     pandar::InputPCAP -- derived class provides a similar interface
 *                      from a PCAP dump file
 */

#ifndef __PANDAR_INPUT_H
#define __PANDAR_INPUT_H

#include <netinet/in.h>
#include <pcap.h>
#include <stdio.h>
#include <unistd.h>

#include <map>
#include <string>

#include "util.h"

#define ETHERNET_MTU (1500)
#define UDP_VERSION_MAJOR_1 (1)
#define UDP_VERSION_MAJOR_4 (4)
#define UDP_VERSION_MINOR_1 (1)
#define UDP_VERSION_MINOR_2 (2)
#define UDP_VERSION_MINOR_3 (3)
#define UDP_VERSION_MINOR_4 (4)
#define UDP_VERSION_1_3 "1.3"
#define UDP_VERSION_1_4 "1.4"
#define UDP_VERSION_4_1 "4.1"
#define UDP_VERSION_4_3 "4.3"
#define GPS_PACKET_SIZE (512)
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
#define PANDAR128_TAIL_RESERVED1_SIZE (3)
#define PANDAR128_TAIL_RESERVED2_SIZE (3)
#define PANDAR128_SHUTDOWN_FLAG_SIZE (1)
#define PANDAR128_TAIL_RESERVED3_SIZE (3)
#define PANDAR128_MOTOR_SPEED_SIZE (2)
#define PANDAR128_AZIMUTH_FLAG_SIZE (2)
#define PANDAR128_TS_SIZE (4)
#define PANDAR128_RETURN_MODE_SIZE (1)
#define PANDAR128_FACTORY_INFO (1)
#define PANDAR128_UTC_SIZE (6)
#define PANDAR128_CRC_SIZE (4)
#define PANDAR128_FUNCTION_SAFETY_SIZE (17)
#define PANDAR128_IMU_SIZE (22)
#define PANDAR128_SIGNATURE_SIZE (32)
#define PANDAR128_SEQ_NUM_SIZE (4)

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
#define PANDAR_AT128_CRC_SIZE (4)
#define PANDAR_AT128_FUNCTION_SAFETY_SIZE (17)
#define PANDAR_AT128_SIGNATURE_SIZE (32)
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
#define FAULT_MESSAGE_PCAKET_SIZE (99)
#define LOG_REPORT_PCAKET_SIZE (273)
/************************************* AT 128
 * *********************************************/

enum enumIndex {
  TIMESTAMP_INDEX,
  UTC_INDEX,
  SEQUENCE_NUMBER_INDEX,
  PACKET_SIZE,
};

static std::map<enumIndex, int> udpVersion13 = {
    {TIMESTAMP_INDEX, 796},
    {UTC_INDEX, 802},
    {SEQUENCE_NUMBER_INDEX, 817},
    {PACKET_SIZE, 812},
};

static std::map<enumIndex, int> udpVersion14 = {
    {TIMESTAMP_INDEX, 826},
    {UTC_INDEX, 820},
    {SEQUENCE_NUMBER_INDEX, 831},
    {PACKET_SIZE, 893},
};

enum PacketType {
  POINTCLOUD_PACKET,
  ERROR_PACKET,
  GPS_PACKET,
  PCAP_END_PACKET,
  FAULT_MESSAGE_PACKET,
  LOG_REPORT_PACKET,
};

typedef struct PandarPacket_s {
  double stamp;
  uint8_t data[ETHERNET_MTU];
  int32_t size;
} PandarPacket;

static uint16_t DATA_PORT_NUMBER = 2368;  // default data port
static uint16_t GPS_PORT_NUMBER = 10110;  // default gps port
/** @brief pandar input base class */
class Input {
 public:
  Input(std::string deviceipaddr, uint16_t lidarport);
  virtual ~Input() {}

  /** @brief Read one pandar packet.
   *
   * @param pkt points to pandarPacket message
   *
   * @returns 0 if successful,
   *          -1 if end of file
   *          > 0 if incomplete packet (is this possible?)
   */
  virtual PacketType getPacket(PandarPacket *pkt, bool &isTimeout,
                               bool &skipSleep) = 0;
  bool checkPacketSize(PandarPacket *pkt);
  void setUdpVersion(uint8_t major, uint8_t minor);
  std::string getUdpVersion();

 protected:
  uint16_t m_u16LidarPort;
  std::string m_sDeviceIpAddr;
  std::string m_sUdpVresion;
  bool m_bGetUdpVersion;
  int m_iTimestampIndex;
  int m_iUtcIindex;
  int m_iSequenceNumberIndex;
  int m_iPacketSize;
};

/** @brief Live pandar input from socket. */
class InputSocket : public Input {
 public:
  InputSocket(std::string deviceipaddr, std::string hostipaddr, uint16_t lidarport = DATA_PORT_NUMBER,
              uint16_t gpsport = GPS_PORT_NUMBER, std::string multicast_ip = "");
  virtual ~InputSocket();
  virtual PacketType getPacket(PandarPacket *pkt, bool &isTimeout,
                               bool &skipSleep);
  void calcPacketLoss(PandarPacket *pkt);

 private:
  int m_iSockfd;
  int m_iSockGpsfd;
  int m_iSocktNumber;
  uint32_t m_u32Sequencenum;
};

/** @brief pandar input from PCAP dump file.
 *
 * Dump files can be grabbed by libpcap, pandar's DSR software,
 * ethereal, wireshark, tcpdump, or the \ref vdump_command.
 */
class InputPCAP : public Input {
 public:
  InputPCAP(std::string deviceipaddr, uint16_t lidarport, std::string pcapfile);
  virtual ~InputPCAP();
  virtual PacketType getPacket(PandarPacket *pkt, bool &isTimeout,
                               bool &skipSleep);
  void sleep(const uint8_t *packet, bool &isTimeout);

 private:
  std::string m_sPcapFile;
  pcap_t *m_pcapt;
  bpf_program m_objPcapPacketFilter;
  char m_cErrorArray[PCAP_ERRBUF_SIZE];
  int m_iTimeGap;
  int64_t m_i64LastPktTimestamp;
  int m_iPktCount;
  int64_t m_i64LastTime;
  int64_t m_i64CurrentTime;
  int64_t m_i64PktTimestamp;
};

#endif  // __PANDAR_INPUT_H
