/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: socket can interface monitor[vehicle]
 */

#ifndef CAN_PARSER_INS_PVATB_H
#define CAN_PARSER_INS_PVATB_H

#include <map>
#include <mutex>
#include <string>

#include "can_parser.h"

namespace hozon {
namespace netaos {
namespace ins_pvatb {

struct CountData {
    uint32_t count;
    std::map<uint16_t, uint16_t> datamap;
    CountData() {
        count = 0;
        datamap.clear();
    }
    void Clear() {
        count = 0;
        datamap.clear();
    }
};

struct Time {
    uint32_t sec;
    uint32_t nsec;
};
struct GeometryPoit {
    double x;
    double y;
    double z;
};

struct ImuPose {
    GeometryPoit imuPosition;
    GeometryPoit eulerAngle;
};

// 结构体 : GnssHeadingInfo
// 功能描述 : GnssHeadingInfo原始定位信息结构体
/*******************************************************************************
 */
struct GnssHeadingInfo {
    uint8_t svs;
    uint8_t solnSVs;
    uint8_t posType;
    float length;
    float heading;
    float pitch;
    float hdgStd;
    float pitchStd;
};

// 结构体 : GnssPosInfo
// 功能描述 : GnssPosInfo
/*******************************************************************************
 */
struct GnssPosInfo {
    uint8_t posType;
    double latitude;
    double longitude;
    float undulation;
    float altitude;
    float latStd;
    float lonStd;
    float hgtStd;
    uint8_t svs;
    uint8_t solnSVs;
    float diffAge;
    float hdop;
    float vdop;
    float pdop;
    float gdop;
    float tdop;
};

// 结构体 : GnssVelInfo
// 功能描述 : GnssVelInfo
/*******************************************************************************
 */
struct GnssVelInfo {
    uint8_t solStatus;
    float horSpd;
    float trkGnd;
    GeometryPoit velocity;
    GeometryPoit velocityStd;
};

/* ******************************************************************************
    结构 名        :  ImuInfo
    功能描述        :  提供IMU数据信息
******************************************************************************
*/
struct ImuInfo {
    // uint32_t gpsWeek;
    // double gpsSec;
    GeometryPoit angularVelocity;
    GeometryPoit acceleration;
    GeometryPoit imuVBAngularVelocity;
    GeometryPoit imuVBLinearAcceleration;
    uint16_t imuStatus;
    float temperature;
    GeometryPoit gyroOffset;
    GeometryPoit gyoSF;
    GeometryPoit accelOffset;
    GeometryPoit accSF;
    GeometryPoit ins2antoffset;
    ImuPose imu2bodyosffet;
    float imuyaw;
};

/* ******************************************************************************
结构体 : InsInfo
功能描述 : Ins导远定位信息结构体
******************************************************************************
*/
struct InsInfo {
    // uint32_t gpsWeek;
    // double gpsSec;
    double latitude;
    double longitude;
    double altitude;
    GeometryPoit attitude;
    GeometryPoit linearVelocity;
    GeometryPoit augularVelocity;
    GeometryPoit linearAcceleration;
    float heading;
    GeometryPoit mountingError;
    GeometryPoit sdPosition;
    GeometryPoit sdAttitude;
    GeometryPoit sdVelocity;
    uint16_t sysStatus;
    uint16_t gpsStatus;
    uint16_t sensorUsed;
    float wheelVelocity;
    float odoSF;
};

struct GnssInfo {
    uint32_t gpsWeek;
    double gpsSec;
    GnssHeadingInfo gnss_heading_info;
    GnssVelInfo gnss_vel_info;
    GnssPosInfo gnss_pos_info;
};

struct InsDataInternal {
    GnssInfo gnss_info;
};

struct ImuInsDataInternal {
    uint32_t gpsWeek;
    double gpsSec;
    ImuInfo imub_info;
    InsInfo ins_info;
};
struct InsErrHandleParam {
    uint8_t  flag_bit;
    uint16_t fault_id;
    uint64_t report_time;
};
class CanParserInsPvatb : public hozon::netaos::canstack::CanParser {
   public:
    static CanParserInsPvatb* Instance();
    virtual ~CanParserInsPvatb() = default;
    virtual void Init();
    virtual void ParseCan(can_frame& receiveFrame);
    virtual void ParseCanfd(canfd_frame& receiveFrame);
    virtual void GetCanFilters(std::vector<can_filter>& filters);
    // InsDataInternal UtDebugGetData(); //ut
   private:
    CanParserInsPvatb();
    CountData recvDataCnt;
    static CanParserInsPvatb* mInstancePtr;
    uint8_t gnss_count;
    uint8_t imu_count;
    // uint8_t ins_count;
    uint8_t gnss_id385_count_;
    static std::mutex m_mtx1;
    std::map<uint8_t, InsErrHandleParam> ins_err_map_;
    
    void parse_ins_pvatb(long id, unsigned char* msg);
    void hcfd_monb(unsigned char* msg);
    void hcfd_mon_imub(unsigned char* msg);
    void hcfd_raw_gnss_pvat2b(unsigned char* msg);
    void hcfd_raw_gnss_pvat3b(unsigned char* msg);
    void hcfd_raw_gnss_pvatb(unsigned char* msg);
    void hcfd_raw_imub(unsigned char* msg);
    void hcfd_raw_imuvb(unsigned char* msg);
    void hcfd_ins_pvatb(unsigned char* msg);
    void hcfd_ins_pvat2b(unsigned char* msg);
    void hcfd_err_infob(unsigned char* msg);
    void hcfd_ins_pvat3b(unsigned char* msg);
    int doule_val_same(double val1, double val2);
    int32_t Int32ToSigned(uint32_t data, uint8_t highestindex);
    void PrintImuIns();
};
}  // namespace ins_pvatb
}  // namespace netaos
}  // namespace hozon
#endif  // CAN_PARSER_INS_PVATB_H
