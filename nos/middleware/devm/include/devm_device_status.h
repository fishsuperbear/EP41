/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: devm
 * Description: devm_cpu_info.cpp
 * Created on: Nov 21, 2023
 * Author: yanlongxiang
 *
 */
#pragma once

#include "devm_define.h"
#include "zmq_ipc/manager/zmq_ipc_client.h"

using namespace hozon::netaos::zmqipc;

namespace hozon {
namespace netaos {
namespace devm {


class DevmClientDeviceStatus {
public:
    DevmClientDeviceStatus();
    ~DevmClientDeviceStatus();
    int32_t SendRequestToServer(Devicestatus& resp);

private:
    std::shared_ptr<ZmqIpcClient> client_{};
};

}  // namespace devm
}  // namespace netaos
}  // namespace hozon

