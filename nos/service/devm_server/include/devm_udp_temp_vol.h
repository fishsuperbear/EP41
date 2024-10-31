/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: udp get temperature
 */

#pragma once

#include <sys/un.h>
#include <netinet/in.h>
#include <stdint.h>
#include <unistd.h>

#include "device_info.h"

namespace hozon {
namespace netaos {
namespace devm_server {


class DevmSocketTempAndVol {
public:
    DevmSocketTempAndVol() : fd_(-1), type_(0) {}
    ~DevmSocketTempAndVol() {}
    int32_t GetFd() {return fd_;}
    void SetFd(int32_t fd) {fd_ = fd;}
    void CloseFd() {close(fd_);}
public:
    int32_t fd_;
    int32_t type_;
    //doip_event_source_t* source;
};


class DevmUdpTempAndVol : DevmSocketTempAndVol {
public:
    DevmUdpTempAndVol();
    ~DevmUdpTempAndVol();
    void Run();
    void SetStopFlag();

private:
    int32_t WriteVersionToCfg();
    int32_t GetIp(const char *ifname, char *ip, int32_t iplen);
    struct sockaddr_in addr_{};
    std::string ifname_{};
    int32_t port_{};
    TemperatureData temperature_{};
    VoltageData voltage_{};

    bool stop_flag_{false};
};


}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon


