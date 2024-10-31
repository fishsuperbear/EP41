
/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: ota api definition
 */

#pragma once

#include <iostream>
#include "zmq_ipc/manager/zmq_ipc_client.h"

namespace hozon {
namespace netaos {
namespace otaapi {

enum State {
    NORMAL_IDLE             = 0x01,
    OTA_PRE_UPDATE          = 0x02,
    OTA_UPDATING            = 0x03,
    OTA_UPDATED             = 0x04,
    OTA_ACTIVING            = 0x05,
    OTA_ACTIVED             = 0x06,
    OTA_UPDATE_FAILED       = 0x07,
};


using namespace hozon::netaos::zmqipc;

class OTAApiZmq {
public:
    OTAApiZmq();
    ~OTAApiZmq();

    void ota_api_init();
    void ota_api_deinit();
    std::string ota_get_version();
    int32_t ota_precheck();
    uint8_t ota_progress();
    int32_t ota_start_update(std::string package_path);
    int32_t ota_get_update_status();

private:
    int32_t GetState(std::string state);

    std::shared_ptr<ZmqIpcClient> client_status_;
    std::shared_ptr<ZmqIpcClient> client_precheck_;
    std::shared_ptr<ZmqIpcClient> client_progress_;
    std::shared_ptr<ZmqIpcClient> client_update_;
    std::shared_ptr<ZmqIpcClient> client_version_;

    uint8_t progress_;
};

}
}
}

