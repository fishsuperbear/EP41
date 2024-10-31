/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: devm
 * Description: devm_cpu_info.cpp
 * Created on: Nov 21, 2023
 * Author: yanlongxiang
 *
 */
#include "devm_logger.h"
#include "devm/include/devm_cpu_info.h"
#include "zmq_ipc/proto/devm_tool.pb.h"

namespace hozon {
namespace netaos {
namespace devm {

DevmClientCpuInfo::DevmClientCpuInfo(){
}

DevmClientCpuInfo::~DevmClientCpuInfo() {
}

int32_t DevmClientCpuInfo::SendRequestToServer(CpuData& resp) {
    DEVM_LOG_INFO << "DevmClientCpuInfo SendRequestToServer enter!";

    client_ = std::make_shared<ZmqIpcClient>();
    client_->Init("tcp://localhost:11122");
    std::string reply{};
    DevmReq req_data{};
    
    req_data.set_req_type("cpu_info");
    int32_t res = client_->Request(req_data.SerializeAsString(), reply, 2000);
    if (res < 0) {
        DEVM_LOG_INFO << "CpuData Request failed. failedCode: " << res;
        client_->Deinit();
        return res;
    }

    DevmCpuInfo resp_data{};
    resp_data.ParseFromString(reply);
    resp.architecture = resp_data.architecture();
    resp.cpus = resp_data.cpus();
    resp.online_cpus = resp_data.online_cpus();
    resp.offline_cpus = resp_data.offline_cpus();
    resp.model_name = resp_data.model_name();
    resp.cpu_max_mhz = resp_data.cpu_max_mhz();
    resp.cpu_min_mhz = resp_data.cpu_min_mhz();
    resp.l1d_catch = resp_data.l1d_catch();
    resp.l1i_catch = resp_data.l1i_catch();
    resp.l2_catch = resp_data.l2_catch();
    resp.l3_catch = resp_data.l3_catch();
    resp.temp_cpu = resp_data.temp_cpu();
    resp.temp_soc0 = resp_data.temp_soc0();
    resp.temp_soc1 = resp_data.temp_soc1();
    resp.temp_soc2 = resp_data.temp_soc2();
    for (const auto& it : resp_data.cpus_usage()) {
        resp.cpus_usage.push_back(it);
    }
    for (const auto& it : resp_data.cpu_binding()) {
        resp.cpu_binding.insert(it);
    }

    client_->Deinit();
    return res;
}


}  // namespace devm
}  // namespace netaos
}  // namespace hozon
