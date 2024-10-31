/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: doip client test manager
 */
#ifndef UDS_REQUEST_FUNC_H
#define UDS_REQUEST_FUNC_H


#include <string>
#include <vector>
#include <linux/can.h>
#include "socketraw/socket.h"
#include "isotplib/isotp.h"
#include "uds_request.h"



class UdsResqestFuncOnCan {
public:
    UdsResqestFuncOnCan(){
        socketApi_ = new SocketApi();
        socketApi_->CreateSocket(PF_CAN, SOCK_RAW, CAN_RAW);
        seed_ = 0;

        canid_ = 0x1234;
        isotp_ = new IsoTP();
        isotp_->isotp_init_link(
            &g_link, canid_,
            isotpSendBuf_, sizeof(isotpSendBuf_), 
            isotpRecvBuf_, sizeof(isotpRecvBuf_),
            std::bind(&UdsResqestFuncOnCan::isotp_user_debug, this, std::placeholders::_1),
            std::bind(&UdsResqestFuncOnCan::isotp_user_send_can, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
            std::bind(&UdsResqestFuncOnCan::isotp_user_get_ms, this)
        );
    }
    ~UdsResqestFuncOnCan() {
        delete socketApi_;
        delete isotp_;
    }

    int32_t StartTestUdsOnCan();

private:
    int32_t ParseJsonFile(std::string file);
    int32_t ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data);
    bool CompareResponseData( std::vector<uint8_t > srcData, std::vector<uint8_t > respData, bool isExactMatch);
    
    
    SocketApi *socketApi_;
    uint32_t seed_;
    std::vector<DoipRequestInfo_t > udsRequestInfo_;


#define ISOTP_BUFSIZE   4095
    void isotp_user_debug(const char* message);
    int32_t isotp_user_send_can(const uint32_t arbitration_id, const uint8_t* data, const uint8_t size);
    uint32_t isotp_user_get_ms(void);
    IsoTpLink g_link;
    uint8_t isotpRecvBuf_[ISOTP_BUFSIZE];
    uint8_t isotpSendBuf_[ISOTP_BUFSIZE];
    IsoTP *isotp_;
    
    uint32_t canid_;
    std::string canfname_;
};


#endif

