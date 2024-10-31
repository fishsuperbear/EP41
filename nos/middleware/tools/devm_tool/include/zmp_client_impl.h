/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: zmq devm client
 */
#pragma once

#include "zmq_ipc/manager/zmq_ipc_client.h"

using namespace hozon::netaos::zmqipc;
namespace hozon {
namespace netaos {
namespace tools {

class ZmqToolClient {
public:
    ZmqToolClient();
    ~ZmqToolClient(){};
    void Init();
    void DeInit();
    int32_t ReadDidInfo(const std::string& request, std::string& reply);
    int32_t CpusInfo(std::string& reply);
    int32_t DeviceStatus(std::string& reply);
    int32_t DeviceInfo(std::string& reply);

private:
    std::shared_ptr<ZmqIpcClient> client_;
};

}  // namespace tools
}  // namespace netaos
}  // namespace hozon

