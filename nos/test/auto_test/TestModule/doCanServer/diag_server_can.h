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


class DiagServerOnCan {
public:
    DiagServerOnCan() {
        canid_ = 0x1234;
        fname_ = "vcan0";
        socketApi_ = new SocketApi();
        socketApi_->CreateSocket(PF_CAN, SOCK_RAW, CAN_RAW);
        isotp_ = new IsoTP();
    }
    ~DiagServerOnCan() {
        delete socketApi_;
        delete isotp_;
    }
    int32_t StartTest();
private:
    void isotp_user_debug(const char* message);
    int32_t isotp_user_send_can(const uint32_t arbitration_id,
                         const uint8_t* data, const uint8_t size);
    uint32_t isotp_user_get_ms(void);

#define ISOTP_BUFSIZE   4095
    /* Alloc IsoTpLink statically in RAM */
    IsoTpLink g_link;
    /* Alloc send and receive buffer statically in RAM */
    uint8_t isotpRecvBuf_[ISOTP_BUFSIZE];
    uint8_t isotpSendBuf_[ISOTP_BUFSIZE];
    
    uint32_t canid_;
    std::string fname_;
    SocketApi *socketApi_;
    IsoTP *isotp_;
};



#endif
