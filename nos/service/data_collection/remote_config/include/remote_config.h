/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: remote_config.h
 * @Date: 2023/12/13
 * @Author: shenda
 * @Desc: --
 */

#pragma once

#include <signal.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "utils/include/dc_logger.hpp"

// #include "tsp_comm.h"
// #include "zmq/zmq.hpp"

// #include "zmq_ipc/manager/zmq_ipc_server.h"

// std::atomic_int count{0};

// using tspCommon = hozon::netaos::https::TspComm;

// class DcConfigServerImpl final : public hozon::netaos::zmqipc::ZmqIpcServer {

//    public:
//     DcConfigServerImpl() : hozon::netaos::zmqipc::ZmqIpcServer() {}

//     virtual ~DcConfigServerImpl(){};

//     virtual int32_t Process(const std::string& request, std::string& reply) {
//         std::future<tspCommon::TspResponse> ret_remotecfg = tspCommon::GetInstance().RequestRemoteConfig();
//         auto ret_request = ret_remotecfg.get();
//         std::cout << "result_code:" << ret_request.result_code << " remoteconfig:" << ret_request.response << std::endl;
//         reply = "test" + request;
//         std::cout << "Server------recv:" << request << std::endl;
//         //        DEBUG_LOG("Server recv request: %s, reply: %s", request.c_str(), reply.c_str());
//         return reply.size();
//     };

//    private:
// };

// std::atomic_bool stopFlag_ = false;

// void SigHandler(int signum) {
//     std::cout << "sigHandler signum: " << signum << std::endl;
//     stopFlag_ = true;
// }
