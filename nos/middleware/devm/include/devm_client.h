/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: devm
 * Description: devm_client_impl.cpp
 * Created on:
 * Author: jiangxiang
 *
 */
#pragma once

#include <memory>
#include <string>
#include "devm_define.h"
#include "devm_did_info.h"
#include "devm_cpu_info.h"
#include "devm_device_info.h"
#include "devm_device_status.h"

namespace hozon {
namespace netaos {
namespace devm {


class DevmClient {
public:
    DevmClient();
    ~DevmClient();

    void Init();
    void DeInit();

    std::string ReadDidInfo(std::string did);
    CpuData GetCpuInfo();
    DeviceInfo GetDeviceInfo();
    Devicestatus GetDeviceStatus();

private:
    std::shared_ptr<DevmClientDidInfo> devm_impl_did_info_;
    std::shared_ptr<DevmClientCpuInfo> devm_impl_cpu_info_;
    std::shared_ptr<DevmClientDeviceInfo> devm_impl_device_info_;
    std::shared_ptr<DevmClientDeviceStatus> devm_impl_device_status_;
};

}  // namespace devm
}  // namespace netaos
}  // namespace hozon

