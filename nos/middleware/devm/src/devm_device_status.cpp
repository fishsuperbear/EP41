/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: devm
 * Description: devm_device_status.cpp
 * Created on: Nov 21, 2023
 * Author: yanlongxiang
 *
 */
#include "devm_logger.h"
#include "devm/include/devm_device_status.h"
#include "zmq_ipc/proto/devm_tool.pb.h"

namespace hozon {
namespace netaos {
namespace devm {

DevmClientDeviceStatus::DevmClientDeviceStatus(){
}

DevmClientDeviceStatus::~DevmClientDeviceStatus() {
}

int32_t DevmClientDeviceStatus::SendRequestToServer(Devicestatus& resp) {
    DEVM_LOG_INFO << "DevmClientDeviceStatus SendRequestToServer enter!";

    client_ = std::make_shared<ZmqIpcClient>();
    client_->Init("tcp://localhost:11122");
    std::string reply{};
    DevmReq req_data{};
    
    req_data.set_req_type("device_status");
    int32_t res = client_->Request(req_data.SerializeAsString(), reply, 2000);
    if (res < 0) {
        DEVM_LOG_INFO << "Devicestatus Request failed. failedCode: " << res;
        client_->Deinit();
        return res;
    }

    DevmDeviceStatus resp_data{};
    resp_data.ParseFromString(reply);
    resp.soc_status = resp_data.soc_status();
    resp.mcu_status = resp_data.mcu_status();
    // resp.camera_status = resp_data.camera_status();
    for (const auto& it : resp_data.camera_status()) {
        resp.camera_status.insert({it.first, it.second});
    }
    for (const auto& it : resp_data.lidar_status()) {
        resp.lidar_status.insert({it.first, it.second});
    }
    for (const auto& it : resp_data.radar_status()) {
        resp.radar_status.insert({it.first, it.second});
    }
    for (const auto& it : resp_data.imu_status()) {
        resp.imu_status.insert({it.first, it.second});
    }
    for (const auto& it : resp_data.uss_status()) {
        resp.uss_status.insert({it.first, it.second});
    }

    client_->Deinit();
    return res;
}



}  // namespace devm
}  // namespace netaos
}  // namespace hozon
