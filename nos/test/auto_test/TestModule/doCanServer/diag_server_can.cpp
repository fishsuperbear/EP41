/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: doip client socket
 */
#include <iostream>
#include <fstream>
#include <algorithm>

#include "diag_server_can.h"
#include "isotplib/isotp.h"
#include "common.h"




#if 0
#include <stdarg.h>
static void isotp_user_debug(const char* message, ...)
{
    char buffer[1024];
    
    va_list args;
    va_start(args, message);
    vsnprintf(buffer, sizeof(buffer), message, args);
    va_end(args);
    
    printf("%s", buffer);
}
static int32_t isotp_user_send_can(const uint32_t arbitration_id,
                         const uint8_t* data, const uint8_t size)
{
    printfVecHex("send", (uint8_t *)data, size);

    int32_t ret;
    int32_t len = size <= 8 ? size : 8;
    ret = socketApi_->SocketCanSend(arbitration_id, (uint8_t *)data, len);

    return ret;
}
static uint32_t isotp_user_get_ms(void)
{
    struct timespec times = {0, 0};
    long time;

    clock_gettime(CLOCK_MONOTONIC, &times);
    time = times.tv_sec * 1000 + times.tv_nsec / 1000000;
    return time;
}
#endif


void
DiagServerOnCan::isotp_user_debug(const char* message)
{
    INFO_LOG << message;
}

int32_t
DiagServerOnCan::isotp_user_send_can(const uint32_t arbitration_id,
                        const uint8_t* data, const uint8_t size)
{
    printfVecHex("send", (uint8_t *)data, size);

    int32_t ret;
    int32_t len = size <= 8 ? size : 8;
    ret = socketApi_->SocketCanSend(arbitration_id, (uint8_t *)data, len);
    if (ret < 0) {
        return ISOTP_RET_ERROR;
    }
    return ISOTP_RET_OK;
}

uint32_t
DiagServerOnCan::isotp_user_get_ms(void)
{
    struct timespec times = {0, 0};
    long time;

    clock_gettime(CLOCK_MONOTONIC, &times);
    time = times.tv_sec * 1000 + times.tv_nsec / 1000000;
    return time;
}


static std::vector<std::vector<uint8_t>> udsRequest {
    {0x10,0x01},
    {0x10,0x02},
    {0x10,0x03},
    {0x22,0xf1,0x88},
    {0x22,0xf1,0x89},
    {0x22,0xf1,0x87},
    {0x22,0xf1,0x8a},
    {0x22,0xf1,0x80,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f}
};
static std::vector<std::vector<uint8_t>> udsResponse {
    {0x50,0x01},
    {0x50,0x02},
    {0x50,0x03},
    {0x62,0xf1,0x88,0x31,0x31,0x31,0x31,0x31,0x31,0x31,0x31,0x31,0x31,0x31},
    {0x62,0xf1,0x89,0x32,0x31,0x31,0x31,0x31,0x31,0x31,0x31,0x31,0x31},
    {0x62,0xf1,0x87,0x33,0x33,0x33,0x33,0x33},
    {0x62,0xf1,0x8a,0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34,0x34},
    {0x62,0xf1,0x80,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}
};
//sudo modprobe vcan
//sudo ip link add dev vcan0 type vcan
//sudo ip link set up vcan0

int32_t
DiagServerOnCan::StartTest()
{
    int32_t ret;
    uint32_t canid = 0;
    std::vector<uint8_t> data(8, 0);

    isotp_->isotp_init_link(
        &g_link, canid_,
        isotpSendBuf_, sizeof(isotpSendBuf_), 
        isotpRecvBuf_, sizeof(isotpRecvBuf_),
        std::bind(&DiagServerOnCan::isotp_user_debug, this, std::placeholders::_1),
        std::bind(&DiagServerOnCan::isotp_user_send_can, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
        std::bind(&DiagServerOnCan::isotp_user_get_ms, this)
    );
    socketApi_->SocketCanConfig(fname_.c_str(), true);

    while(1)
    {
        uint32_t timeout = 0x7fffffff;
        if(g_link.send_status == ISOTP_SEND_STATUS_INPROGRESS) {
            // timeout = isotp_user_get_ms();
            // timeout = g_link.send_timer_st > timeout? g_link.send_timer_st - timeout:0;

            timeout = g_link.send_st_min;
            if(g_link.send_sn == 1) {
                timeout = 100;
            }
            INFO_LOG << "timeout " << timeout;
        }
        ret = socketApi_->SocketCanRecv(&canid, &data[0], 8, timeout);
        if (ret > 0) {
            printfVecHex("recv", &data[0], 8);
            isotp_->isotp_on_can_message(&g_link, &data[0], 8);
        }

        if (g_link.receive_status == ISOTP_RECEIVE_STATUS_FULL) {
            uint32_t i;
            uint16_t out_size;
            uint8_t payload[100];
            isotp_->isotp_receive(&g_link, payload, sizeof(payload), &out_size);
            std::vector<uint8_t> vec_recv(payload, payload + out_size);

            printfVecHex("recv succ", g_link.receive_buffer, g_link.receive_size);
            for (i = 0; i < udsRequest.size(); i++) {
                if (vec_recv.size() >= udsRequest[i].size() && 
                    std::equal(udsRequest[i].begin(), udsRequest[i].begin() + udsRequest[i].size(), vec_recv.begin())) {
                    isotp_->isotp_send(&g_link, &udsResponse[i][0], udsResponse[i].size());
                    break;
                }
            }
            if (i >= udsRequest.size()) {
                std::vector<uint8_t> response{0x7F, g_link.receive_buffer[0], 0xFF};
                isotp_->isotp_send(&g_link, &response[0], response.size());
            }
        }
        isotp_->isotp_poll(&g_link);
    }

    return 0;
}
#if 1
//1. compile: g++ diag_server_can.cpp ../Common/isotplib/isotp.cpp ../Common/socketraw/socket.cpp -I../Common -o docan_server
//2. run: docan_server
//3. modify udsRequest and udsResponse add New Services
int main() {
    DiagServerOnCan requestSocket;
    requestSocket.StartTest();
    return 0;
}
#endif

