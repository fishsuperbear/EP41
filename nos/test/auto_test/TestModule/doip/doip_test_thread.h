/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: doip client socket
 */
#ifndef DOIP_REQUEST_SOCKET_H
#define DOIP_REQUEST_SOCKET_H


//
#include <string>
#include <vector>

#include "socketraw/socket.h"
#include "socketraw/socketudp.h"
#include "uds_request.h"
#include "doip_request_test.h"

class DoipResqestSocket {
public:
    DoipResqestSocket() {
        ip_ = "127.0.0.1";
        port_ = 13400;
        ifName_ = "enp0s31f6";
        socketApi_ = new SocketApi();
        socketApiUdp_ = new SocketApiUdp();
        result = 0;
    }
    ~DoipResqestSocket() {
        delete socketApi_;
        delete socketApiUdp_;
    }
    int32_t TestDoipThread(TestInfo info, int32_t threadId);
    int32_t result;

private:
    int32_t Ipv4TcpDoipPayloadSend(doip_payload_t *doip_payload, bool haveAddr, bool haveRand, int32_t protocol);
    int32_t Ipv4TcpDoipPayloadRecv(doip_payload_t *response, uint32_t timeoutMs, int32_t protocol);
    int32_t ParseJsonFile(const std::string& filename, TestInfo& testInfo);
    bool CheckResponse(TestItem::DoipResponse *jsonResp, doip_payload_t *response, bool haveAddr);
    int32_t ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data);

    // int32_t Ipv4TcpRequest(doip_request_t *requestInfo);
    // int32_t Ipv4TcpResponse(doip_payload_t *response);
    // int32_t Ipv4TcpRecvAndCheck(doip_message_type_t msg_type);
    // int32_t Ipv4TcpActivate(doip_request_t *requestInfo);
    
    SocketApi *socketApi_;
    SocketApiUdp *socketApiUdp_;
    std::string ip_;
    int32_t port_;
    std::string ifName_;
};


#endif

