#ifndef SM_ALG_STRUCT_EGOHMI_INFO_H__
#define SM_ALG_STRUCT_EGOHMI_INFO_H__

#include <stdint.h>

struct AlgEgoParkHmiInfo {
    uint8_t PA_ParkBarPercent;
    uint16_t NNS_distance;
};

/* ******************************************************************************
    结构 名        :  AlgEgoHmiFrame
    功能描述       :  规控模块写到底盘的HMI相关信息
****************************************************************************** */
struct AlgEgoHmiFrame {
    AlgEgoParkHmiInfo park_hmi_info;
};

#endif  // SM_ALG_STRUCT_EGOHMI_INFO_H__
