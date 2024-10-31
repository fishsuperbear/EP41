/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: doip client test manager
 */
#ifndef UDS_REQUEST_FUNC_ONIP_H
#define UDS_REQUEST_FUNC_ONIP_H


#include <string>
#include <vector>
#include <linux/can.h>
#include "socketraw/socket.h"
#include "isotplib/isotp.h"
#include "uds_request.h"



class UdsResqestFuncOnIP {
public:
    UdsResqestFuncOnIP(){
        socketApi_ = new SocketApi();
        active_status_ = false;
        seed_ = 0;
        ip_ = "10.4.53.60";
        port_ = 13400;
        ifName_ = "enp0s31f6";
    }
    ~UdsResqestFuncOnIP() {
        delete socketApi_;
    }

    int32_t StartTestUdsOnIP();

private:
    int32_t ParseJsonFile(std::string file);
    int32_t ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data);

    int32_t Ipv4TcpDoipPayloadSend(doip_payload_t *doip_payload);
    int32_t Ipv4TcpDoipPayloadRecv(doip_payload_t *response, uint32_t timeoutMs);
    int32_t Ipv4TcpDoipActivate(doip_request_t *requestInfo, uint32_t timeoutMs);
    int32_t Ipv4TcpDoipRequestUds(doip_request_t *requestInfo, doip_request_t *response, uint32_t timeoutMs);

    int32_t DoipRequestUDSdata(DiagRequestInfo_t& requestInfo, bool ignoreKey);
    bool CompareResponseData( std::vector<uint8_t > srcData, std::vector<uint8_t > respData, bool isExactMatch);
    
    
    SocketApi *socketApi_;
    std::string ip_;
    int32_t port_;
    std::string ifName_;


    std::vector<DoipRequestInfo_t > doipRequestInfo_;
    bool active_status_;
    uint32_t seed_;
};


#endif

