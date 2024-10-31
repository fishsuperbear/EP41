#pragma once

#include <bits/stdint-uintn.h>
#include <bits/time.h>
#include <bits/types/struct_timespec.h>
#include "common.h"
#include "logger.h"

namespace hozon {
namespace netaos {
namespace sensor {

#define  CLOCK_VIRTUAL  12u

const uint32_t stringSize = 20;
const uint32_t trajectoryPointsLength = 60;
const uint32_t covarianceLength = 6 * 6;
const uint32_t keyPointVRFLength = 1;
const uint32_t laneDetectionFrontOutLength = 8;
const uint32_t laneDetectionFrontOutLength1 = 2;
const uint32_t laneDetectionRearOutLength = 8;
const uint32_t laneDetectionRearOutLength1 = 2;
const uint32_t cornersLength = 32;
const uint32_t sensorIDLength = 32;
const uint32_t fusionOutLength = 64;
const uint32_t Uint8Length = 1400;
const uint32_t HeaderLength = 4;
const uint32_t BodyLength = 1396;
const int32_t LaneIndexError = 255;

inline double GetRealTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);

    return static_cast<double>(time_now.tv_sec)
             + static_cast<double>(time_now.tv_nsec) / 1000 / 1000 / 1000;
}  

inline double GetAbslTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_VIRTUAL, &time_now);

    return static_cast<double>(time_now.tv_sec)
             + static_cast<double>(time_now.tv_nsec) / 1000 / 1000 / 1000;
}  

inline double HafTimeConverStamp(hozon::netaos::HafTime stamp) {
    return static_cast<double>(stamp.sec) 
            + static_cast<double>( stamp.nsec) / 1000 / 1000 / 1000;
}

inline uint32_t StampConverS(double time) {
    return static_cast<uint32_t>(time);
}

inline uint32_t StampConverNs(double time) {
    return static_cast<uint32_t>((time - static_cast<uint32_t>(time))  * 1000 * 1000 * 1000);
}

inline uint8_t SetUint8ByBit(bool first_bit, bool second_bit, bool third_bit,
                      bool fourth_bit, bool fifth_bit, bool sixth_bit,
                      bool seventh_bit, bool eighth_bit) {
  uint8_t bit8 = 0;
  return (first_bit | (second_bit << 1) | (third_bit << 2) | (fourth_bit << 3) |
          (fifth_bit << 4) | (sixth_bit << 5) | (seventh_bit << 6) |
          (third_bit << 7)) |
         bit8;
}

#define PRINTSENSORDATA(parameter) \
    SENSOR_LOG_DEBUG << #parameter" : " << parameter ;
    
#define PRINTSENSORDATAINFO(parameter) \
    SENSOR_LOG_INFO << #parameter" : " << parameter ;
}
}
}