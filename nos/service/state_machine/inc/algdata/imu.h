#ifndef SM_ALG_STRUCT_IMU_INFO_H__
#define SM_ALG_STRUCT_IMU_INFO_H__

#include <stdint.h>

struct AlgInsInfo {
    double latitude;   // 纬度 Unit: deg
    double longitude;  // 经度 Unit: deg
    double altitude;   // 海拔高度 Unit: meter
    uint16_t sysStatus;   // 组合状态
    uint16_t gpsStatus;   // 定位状态
};

struct  AlgImuIns {
    struct AlgInsInfo ins_info;
};

#endif // SM_ALG_STRUCT_IMU_INFO_H__
