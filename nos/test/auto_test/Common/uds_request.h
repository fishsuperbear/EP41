/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: doip client test manager
 */
#ifndef UDS_REQUEST_COMMON_H
#define UDS_REQUEST_COMMON_H


#include <string>
#include <vector>
#include <linux/can.h>
#include "socketraw/socket.h"
#include "isotplib/isotp.h"


typedef enum DOIP_TA_TYPE
{
    DOIP_TA_TYPE_PHYSICAL       = 0x00,    /* Physical addressing[One-on-one diagnosis] */
    DOIP_TA_TYPE_FUNCTIONAL     = 0x01     /* Functional addressing[Batch diagnosis] */
} diag_ta_type_t;

typedef struct doip_payload {
    uint8_t version;
    uint8_t ver_inverse;
    uint16_t payload_type;
    uint32_t payload_len;
    uint16_t source_addr;
    uint16_t target_addr;
    uint8_t  *data;
    uint32_t data_length;// payload_len与date实际长度不一致时使用
} doip_payload_t;

typedef struct doip_request
{
    uint8_t tcp_ip_type;
    diag_ta_type_t ta_type;
    
    uint16_t source_addr;
    uint16_t target_addr;
    uint8_t  *data;
    uint32_t data_length;
} doip_request_t;






typedef struct DiagTestRequestInfo {
    diag_ta_type_t type;// 0: none, 1: physical, 2: functional
    uint16_t sourceAddress;
    uint16_t targetAddress;
    std::vector<uint8_t > requestData;
    std::vector<uint8_t > responseData;
    uint32_t timeoutMs; //ms
    bool isExactMatch;
    uint32_t retryCounts;
    uint32_t delayTimeMs; //ms
} DiagRequestInfo_t;

typedef struct DoipRequestInfo {
    std::string ip;
    uint32_t port;
    std::string ifname;
    uint16_t sourceAddress;
    uint16_t targetAddress;
    uint32_t canid;
    std::string canfname;
    
    uint32_t retryCounts;
    uint32_t delayTimeMs; //ms
    bool testerParent;
    std::vector<DiagRequestInfo_t > vecRequest;
    std::string describe;
    bool ignoreFail;//true失败后可以继续执行case，默认false
    bool ignoreKey;//true不在2704/06/12后面加key值
} DoipRequestInfo_t;



#endif

