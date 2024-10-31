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
#include "zmq_ipc/proto/devm_tool.pb.h"

namespace hozon {
namespace netaos {
namespace devm {

DevmClientDeviceInfo::DevmClientDeviceInfo()
    : cfg_mgr_(ConfigParam::Instance()) {

    cfg_mgr_->Init(2000);
}

DevmClientDeviceInfo::~DevmClientDeviceInfo() {
    cfg_mgr_->DeInit();
}

int32_t
DevmClientDeviceInfo::SendRequestToServer(DeviceInfo& resp) {
    DEVM_LOG_INFO << "DevmClientDeviceInfo SendRequestToServer enter!";

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
    resp.soc_type = resp_ver.soc_type();
    resp.mcu_type = resp_ver.mcu_type();
    resp.switch_type = resp_ver.switch_type();

    client_->Deinit();
    return res;
}

bool
DevmClientDeviceInfo::GetVinNumber(std::string &vin_number) {
    DEVM_LOG_INFO << "DevmClientDeviceInfo::ReadVinData";
    if (nullptr == cfg_mgr_) {
        DEVM_LOG_ERROR << "DevmClientDeviceInfo::ReadVinData cfg_mgr_ is nullptr.";
        return false;
    }

    std::string data;
    cfg_mgr_->GetParam<std::string>("dids/F190", data);

    if (0 == data.size()) {
        DEVM_LOG_WARN << "DevmClientDeviceInfo::ReadVinData get data is empty.";
        return false;
    }

    if (data.size() != VIN_DATA_LENGTH) {
        DEVM_LOG_ERROR << "DevmClientDeviceInfo::ReadVinData get data size error. "
                 << "correct vin_data_.size: " << VIN_DATA_LENGTH << " cfg get data.size: " << data.size();
        return false;
    }

    vin_number = data;

    return true;
}

bool
DevmClientDeviceInfo::GetEcuSerialNumber(std::string& sn_number) {
    DEVM_LOG_INFO << "DevmClientDeviceInfo::GetEcuSerialNumber";
    if (nullptr == cfg_mgr_) {
        DEVM_LOG_ERROR << "DevmClientDeviceInfo::GetEcuSerialNumber cfg_mgr_ is nullptr.";
        return false;
    }

    std::string data;
    cfg_mgr_->GetParam<std::string>("dids/F18C", data);

    if (0 == data.size()) {
        DEVM_LOG_WARN << "DevmClientDeviceInfo::GetEcuSerialNumber get data is empty.";
        return false;
    }

    if (data.size() != ECU_SERIAL_NUMBER_LENGTH) {
        DEVM_LOG_ERROR << "DevmClientDeviceInfo::GetEcuSerialNumber get data size error. "
                 << "correct tester_sn_data_.size: " << ECU_SERIAL_NUMBER_LENGTH << " cfg get data.size: " << data.size();
        return false;
    }

    sn_number = data;

    return true;
}

bool
DevmClientDeviceInfo::GetEcuParterNum(std::string& ecu_parter_num) {
    DEVM_LOG_INFO << "DevmClientDeviceInfo::GetEcuParterNum";
    if (nullptr == cfg_mgr_) {
        DEVM_LOG_ERROR << "DevmClientDeviceInfo::GetEcuParterNum cfg_mgr_ is nullptr.";
        return false;
    }

    std::string data;
    cfg_mgr_->GetParam<std::string>("dids/F1D0", data);

    if (0 == data.size()) {
        DEVM_LOG_WARN << "DevmClientDeviceInfo::GetEcuParterNum get data is empty.";
        return false;
    }

    if (data.size() != ECU_PART_NUM_LENGTH) {
        DEVM_LOG_ERROR << "DevmClientDeviceInfo::GetEcuParterNum get data size error. "
                 << "correct tester_sn_data_.size: " << ECU_PART_NUM_LENGTH << " cfg get data.size: " << data.size();
        return false;
    }

    ecu_parter_num = data;

    return true;
}

bool
DevmClientDeviceInfo::GetVehicleCfg(std::string& ecu_vehicle_cfg) {
    DEVM_LOG_INFO << "DevmClientDeviceInfo::GetVehicleCfg";
    if (nullptr == cfg_mgr_) {
        DEVM_LOG_ERROR << "DevmClientDeviceInfo::GetVehicleCfg cfg_mgr_ is nullptr.";
        return false;
    }

    std::string data;
    cfg_mgr_->GetParam<std::string>("dids/F170", data);

    if (0 == data.size()) {
        DEVM_LOG_WARN << "DevmClientDeviceInfo::GetVehicleCfg get data is empty.";
        return false;
    }

    if (data.size() != VEHICLE_CFG_WORD_DATA_LENGTH) {
        DEVM_LOG_ERROR << "DevmClientDeviceInfo::GetVehicleCfg get data size error. "
                 << "correct tester_sn_data_.size: " << VEHICLE_CFG_WORD_DATA_LENGTH << " cfg get data.size: " << data.size();
        return false;
    }

    ecu_vehicle_cfg = data;

    return true;
}

bool
DevmClientDeviceInfo::GetTemperature(struct TemperatureData& temp) {
    DEVM_LOG_INFO << "DevmClientDeviceInfo::GetTemperature";

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
DevmClientDeviceInfo::GetVoltage(struct VoltageData& voltage) {
    DEVM_LOG_INFO << "DevmClientDeviceInfo::GetVoltage";

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
