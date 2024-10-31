/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: zmq devm server
 */
#pragma once

#include <string>
#include <map>
#include <queue>

#include "zmq_ipc/manager/zmq_ipc_server.h"
#include "zmq_ipc/proto/log_server.pb.h"

using namespace hozon::netaos::zmqipc;
namespace hozon {
namespace netaos {
namespace devm_server {

const std::string compress_log_service_name = "tcp://*:11122";

class DevmServerImplZmq final : public ZmqIpcServer
{

public:
    DevmServerImplZmq();
    virtual ~DevmServerImplZmq(){};
    int32_t Init();
    int32_t DeInit();
    virtual int32_t Process(const std::string& request, std::string& reply);

private:
    // void CompressFile(const std::string& filePath, const std::string& file_base_name, const std::string& filename, const std::string& basename);
    // bool rename_file_(const spdlog::filename_t &src_filename, const spdlog::filename_t &target_filename);
private:
   std::mutex mtx_;
};

}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon
