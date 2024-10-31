/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: devm
 * Description: devm_client_impl.cpp
 * Created on: Sep 11, 2023
 * Author: jiangxiang
 *
 */
#include "devm_logger.h"
#include "devm_client.h"
#include "devm/include/devm_device_info.h"
#include "devm/include/common/devm_logger.h"

namespace hozon {
namespace netaos {
namespace devm {

DevmClient::DevmClient() {
    DevmClientLogger::GetInstance().CreateLogger("DEVMC");
    devm_impl_did_info_ = std::make_shared<DevmClientDidInfo>();
    devm_impl_cpu_info_ = std::make_shared<DevmClientCpuInfo>();
    devm_impl_device_info_ = std::make_shared<DevmClientDeviceInfo>();
    devm_impl_device_status_ = std::make_shared<DevmClientDeviceStatus>();
}

DevmClient::~DevmClient() {
}

void DevmClient::Init() {
}

void DevmClient::DeInit() {
}

std::string DevmClient::ReadDidInfo(std::string did) {
    return devm_impl_did_info_->ReadDidInfo(did);
}

CpuData DevmClient::GetCpuInfo() {
    CpuData temp{};
    devm_impl_cpu_info_->SendRequestToServer(temp);
    return temp;
}

DeviceInfo DevmClient::GetDeviceInfo() {
    DeviceInfo temp{};
    devm_impl_device_info_->SendRequestToServer(temp);
    return temp;
}

Devicestatus DevmClient::GetDeviceStatus() {
    Devicestatus temp{};
    devm_impl_device_status_->SendRequestToServer(temp);
    return temp;
}

}  // namespace devm
}  // namespace netaos
}  // namespace hozon
