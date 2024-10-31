/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: devm
 * Description: devm_device_info.cpp
 * Created on: Oct 31, 2023
 * Author: yanlongxiang
 *
 */
#include "devm_logger.h"
#include "devm/include/devm_device_info.h"
#include "devm/include/devm_device_info_zmq.h"
#include "zmq_ipc/proto/devm_tool.pb.h"

namespace hozon {
namespace netaos {
namespace devm {

DevmClientDeviceInfoZmq::DevmClientDeviceInfoZmq(){
}

DevmClientDeviceInfoZmq::~DevmClientDeviceInfoZmq() {
}

int32_t
DevmClientDeviceInfoZmq::SendRequestToServer(DeviceInfo& resp) {
    DEVM_LOG_INFO << "DevmClientDeviceInfoZmq SendRequestToServer enter!";

    client_ = std::make_shared<ZmqIpcClient>();
    client_->Init("tcp://localhost:11122");
    std::string reply{};
    DevmReq req_data{};

    req_data.set_req_type("device_info");
    int32_t res = client_->Request(req_data.SerializeAsString(), reply, 2000);
    if (res < 0) {
        DEVM_LOG_WARN << "DeviceInfo Request failed. failedCode: " << res;
        client_->Deinit();
        return res;
    }

    DevmDeviceInfo resp_ver{};
    resp_ver.ParseFromString(reply);
    resp.soc_version = resp_ver.soc_version();
    resp.mcu_version = resp_ver.mcu_version();
    resp.swt_version = resp_ver.swt_version();
    resp.dsv_version = resp_ver.dsv_version();
    resp.uss_version = resp_ver.uss_version();
    resp.soc_type = resp_ver.soc_type();
    resp.mcu_type = resp_ver.mcu_type();
    resp.switch_type = resp_ver.switch_type();
    for (const auto& it : resp_ver.sensor_version()) {
        resp.sensor_version.insert({it.first, it.second});
    }
    client_->Deinit();
    return res;
}

bool
DevmClientDeviceInfoZmq::GetTemperature(struct TemperatureData& temp) {
    DEVM_LOG_INFO << "DevmClientDeviceInfoZmq::GetTemperature";

    client_ = std::make_shared<ZmqIpcClient>();
    client_->Init("tcp://localhost:11122");
    std::string reply{};
    DevmReq req_data{};

    req_data.set_req_type("devm_temperature");
    int32_t res = client_->Request(req_data.SerializeAsString(), reply, 2000);
    if (res < 0) {
        DEVM_LOG_WARN << "Temperature Request failed. failedCode: " << res;
        client_->Deinit();
        return res;
    }

    DevmTemperature resp_temp{};
    resp_temp.ParseFromString(reply);
    temp.temp_soc = resp_temp.soc_temp();
    temp.temp_mcu = resp_temp.mcu_temp();
    temp.temp_ext0 = resp_temp.ext0_temp();
    temp.temp_ext1 = resp_temp.ext1_temp();

    client_->Deinit();
    return true;
}

bool
DevmClientDeviceInfoZmq::GetVoltage(struct VoltageData& voltage) {
    DEVM_LOG_INFO << "DevmClientDeviceInfoZmq::GetVoltage";

    client_ = std::make_shared<ZmqIpcClient>();
    client_->Init("tcp://localhost:11122");
    std::string reply{};
    DevmReq req_data{};

    req_data.set_req_type("devm_voltage");
    int32_t res = client_->Request(req_data.SerializeAsString(), reply, 2000);
    if (res < 0) {
        DEVM_LOG_WARN << "Voltage Request failed. failedCode: " << res;
        client_->Deinit();
        return res;
    }

    DevmVoltage resp_vol{};
    resp_vol.ParseFromString(reply);
    voltage.kl15 = resp_vol.kl15_vol();
    voltage.kl30 = resp_vol.kl30_vol();

    client_->Deinit();
    return true;
}

}  // namespace devm
}  // namespace netaos
}  // namespace hozon
