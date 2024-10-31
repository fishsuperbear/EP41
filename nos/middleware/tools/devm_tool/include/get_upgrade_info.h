
/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: zmq devm upgrade client
 */
#pragma once
#include <iostream>
#include <vector>
#include <memory>

#include "zmq_ipc/manager/zmq_ipc_client.h"

using namespace hozon::netaos::zmqipc;
namespace hozon {
namespace netaos {
namespace tools {

class UpgradeInfoZmq {
public:
    UpgradeInfoZmq(std::vector<std::string> arguments);
    ~UpgradeInfoZmq();
    int32_t StartGetUpgradeInfo();
private:
    void PrintUsage();
    void PrintCurTime();
    int32_t upgrade_status();
    int32_t upgrade_precheck();
    int32_t upgrade_progress();
    int32_t upgrade_update(const std::string package_path, bool precheck, int32_t ecu_mode, bool skip_version);
    int32_t upgrade_version();
    int32_t upgrade_finish();
    int32_t upgrade_result();
    int32_t upgrade_partition();
    int32_t upgrade_switch_slot();
    std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr);

    std::vector<std::string> arguments_;
    std::shared_ptr<ZmqIpcClient> client_;
};

}
}
}
