/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: zmq devm client
 */
#include "zmp_client_impl.h"
#include "devm_tool_logger.h"
#include "zmq_ipc/proto/devm_tool.pb.h"

namespace hozon {
namespace netaos {
namespace tools {

ZmqToolClient::ZmqToolClient()
{
    client_ = std::make_shared<ZmqIpcClient>();
}

void
ZmqToolClient::Init()
{
    DEVMTOOL_DEBUG << "ZmqToolClient::Init";
    client_->Init("tcp://localhost:11122");
}

void
ZmqToolClient::DeInit()
{
    DEVMTOOL_DEBUG << "ZmqToolClient::DeInit";
    client_->Deinit();
}

int32_t 
ZmqToolClient::ReadDidInfo(const std::string& dids, std::string& reply)
{
    DevmReq req{};
    req.set_req_type("did_info");
    req.set_data_value(dids.c_str());
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        std::cout << "Error: zmq request ret err." << std::endl;
        DEVMTOOL_ERROR << "zmq request ret err.";
        client_->Deinit();
        return -1;
    }

    std::string type = req.req_type();
    if (type == "did_info") {
        DevmDidInfo didinfo{};
        didinfo.ParseFromString(reply);
        DEVMTOOL_INFO << "read did " << didinfo.did() << " " << didinfo.data_value();
    }

    return 0;
}

int32_t 
ZmqToolClient::DeviceInfo(std::string& value)
{
    DevmReq req{};
    req.set_req_type("device_info");
    req.set_data_value("");
    std::string reply{};
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        std::cout << "Error: zmq request ret err." << std::endl;
        DEVMTOOL_ERROR << "zmq request ret err.";
        client_->Deinit();
        return -1;
    }

    std::string type = req.req_type();
    if (type == "device_info") {
        DevmDeviceInfo devinfo{};
        devinfo.ParseFromString(reply);
        DEVMTOOL_INFO << "dev-info " << devinfo.mcu_version();
        value = devinfo.mcu_version();
    }
    return 0;
}

int32_t 
ZmqToolClient::DeviceStatus(std::string& reply)
{
    DevmReq req{};
    req.set_req_type("device_status");
    req.set_data_value("");
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        std::cout << "Error: zmq request ret err." << std::endl;
        DEVMTOOL_ERROR << "zmq request ret err.";
        client_->Deinit();
        return -1;
    }

    std::string type = req.req_type();
    if (type == "device_status") {
        DevmDeviceStatus devstatus{};
        devstatus.ParseFromString(reply);
        DEVMTOOL_INFO << "dev-status " << devstatus.soc_status() << " " << devstatus.mcu_status();
    }
    return 0;
}

int32_t 
ZmqToolClient::CpusInfo(std::string& reply)
{
    DevmReq req{};
    req.set_req_type("cpu_info");
    req.set_data_value("");
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        std::cout << "Error: zmq request ret err." << std::endl;
        DEVMTOOL_ERROR << "zmq request ret err.";
        client_->Deinit();
        return -1;
    }

    std::string type = req.req_type();
    if (type == "cpu_info") {
        DevmCpuInfo cpuinfo{};
        cpuinfo.ParseFromString(reply);
        DEVMTOOL_INFO << "cpu-info " << cpuinfo.architecture();
    }
    return 0;
}


}  // namespace tools
}  // namespace netaos
}  // namespace hozon
