/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: doip client socket
 */
#ifndef DIAG_CLIENT_CAN_H
#define DIAG_CLIENT_CAN_H

#include <string>
#include <vector>
#include <linux/can.h>
#include "socketraw/socket.h"
#include "isotplib/isotp.h"
#include "uds_request.h"

typedef enum DIAG_SERVER_ID_TYPE
{
    UDS_SERVICE_SESSION    = 0x10,
    UDS_SERVICE_RESET      = 0x11,
    UDS_SERVICE_CLEAR      = 0x14,
    UDS_SERVICE_READ       = 0x22,
    UDS_SERVICE_SECURY     = 0x27,
    UDS_SERVICE_WRITE      = 0x2E,
    UDS_SERVICE_CONTRAL    = 0x31,
    UDS_SERVICE_COMUNITE   = 0x85
    //.....
} diag_service_id_t;


typedef struct can_request
{
    uint32_t can_id;
    //uint8_t  source_address;          /* Extended Frame source address. */
    //uint8_t  target_address;          /* Extended Frame target address. */
    diag_ta_type_t ta_type;             /* Target address type. */
    uint8_t  *data;                     /* UDS data[The TP is responsible for releasing the memory] */
    uint32_t data_length;               /* UDS data length */
} can_request_data_t;

typedef struct can_response
{
    uint32_t can_id;
    uint8_t  *data;                     /* UDS data[The TP is responsible for releasing the memory] */
    uint32_t data_length;               /* UDS data length */
} can_response_data_t;





struct TestInfoCanTP {
    std::string request_canid;
    std::string response_canid;
    std::string type;
    
    struct TestItem {
        std::vector<std::string> request;
        std::vector<std::string> response;
        bool isExactMatch;
        int timeout;
        int retryCounts;
        int delayTime;
        std::string describe;
    };
    std::vector<TestItem> test;
};

class DiagResqestOnCan {
public:
    DiagResqestOnCan() {
        canid_ = 0x1234;
        socketApi_ = new SocketApi();
        socketApi_->CreateSocket(PF_CAN, SOCK_RAW, CAN_RAW);
        
        isotp_ = new IsoTP();
        isotp_->isotp_init_link(
            &g_link, canid_,
            isotpSendBuf_, sizeof(isotpSendBuf_), 
            isotpRecvBuf_, sizeof(isotpRecvBuf_),
            std::bind(&DiagResqestOnCan::isotp_user_debug, this, std::placeholders::_1),
            std::bind(&DiagResqestOnCan::isotp_user_send_can, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
            std::bind(&DiagResqestOnCan::isotp_user_get_ms, this)
        );
    }
    ~DiagResqestOnCan() {
        delete socketApi_;
        delete isotp_;
    }
    int32_t StartTestCanTP();
    
    int32_t SocketRawCanSend(uint32_t& canid, std::vector<uint8_t>& data);
    int32_t SocketRawCanRecv(uint32_t& canid, std::vector<uint8_t>& data);

private:
    int32_t ParseJsonFile(const std::string& filename, TestInfoCanTP& testInfo);
    int32_t ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data);
    bool CheckResponse(std::vector<uint8_t > srcData, std::vector<uint8_t > respData, bool isExactMatch);

#define ISOTP_BUFSIZE   4095
    void isotp_user_debug(const char* message);
    int32_t isotp_user_send_can(const uint32_t arbitration_id, const uint8_t* data, const uint8_t size);
    uint32_t isotp_user_get_ms(void);
    
    IsoTpLink g_link;
    uint8_t isotpRecvBuf_[ISOTP_BUFSIZE];
    uint8_t isotpSendBuf_[ISOTP_BUFSIZE];
    IsoTP *isotp_;


    uint32_t canid_;
    std::string ifname_;
    TestInfoCanTP testInfo_;
    SocketApi *socketApi_;
};



#endif
