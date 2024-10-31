/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: state_server_impl_zmq.h
 * Created on: Nov 16, 2023
 * Author: yanlongxiang
 * 
 */

#pragma once

#include <mutex>
#include <map>
#include "sm/include/sm_types.h"
#include "sm/include/sm_logger.h"
#include "zmq_ipc/manager/zmq_ipc_server.h"
#include "zmq_ipc/proto/sm.pb.h"

using namespace hozon::netaos::log;

namespace hozon {
namespace netaos {
namespace sm {

const std::string stat_service_name = "tcp://*:11150";
class StateServerImplZmq {
public:

    class StatZmqServer final : public hozon::netaos::zmqipc::ZmqIpcServer
    {
        public:
        StatZmqServer(StateServerImplZmq *server) : state_server(server) {}
        virtual ~StatZmqServer() {}
        virtual int32_t Process(const std::string& request, std::string& reply) {
            std::lock_guard<std::mutex> lck(mtx_);
            return state_server->RequestProcess(request, reply);
        }

        private:
        std::mutex mtx_;
        StateServerImplZmq *state_server;
    };


    StateServerImplZmq() : state_zmq_server(this){};
    ~StateServerImplZmq(){};
    int32_t Start();
    void Stop();
    int32_t RequestProcess(const std::string& request, std::string& reply);

private:
    std::string GetProcessName();
    bool PreAndPostProcess(const ProcessMode mod, const std::string& src_mode, const std::string& tar_mode, std::map<std::string, std::string>& map);

    StatZmqServer state_zmq_server;
    std::map<std::pair<std::string,std::string>,std::map<std::string, std::string>> PreProcessFuncMap;
    std::map<std::pair<std::string,std::string>,std::map<std::string, std::string>> PostProcessFuncMap;
    std::mutex preprocess_funcmap_mutex;
    std::mutex postprocess_funcmap_mutex;

};
} // namespace sm
} // namespace netaos
} // namespace hozon
