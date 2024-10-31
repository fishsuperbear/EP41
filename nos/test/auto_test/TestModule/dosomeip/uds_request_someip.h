/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: doip client socket
 */
#ifndef DIAG_REQUEST_SOMEIF_H
#define DIAG_REQUEST_SOMEIF_H

#include <string>
#include <uds_request.h>




class UdsResqestSomeip {
public:
    UdsResqestSomeip() {
        ip_ = "10.4.53.60";
        port_ = 13400;
        ifName_ = "enp0s31f6";
    }
    int32_t SomeIPActivate(doip_payload_t *requestInfo);
    int32_t SomeIPRequest(doip_payload_t *requestInfo);
    int32_t SomeIPResponse(doip_payload_t *msg_type);
    
    int32_t StartTestUdsSomeIP();//测试使用

private:
    std::string ip_;
    int32_t port_;
    std::string ifName_;
};

#endif

