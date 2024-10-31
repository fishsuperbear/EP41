/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_server_impl_zmq.cpp
 * Created on: Nov 22, 2023
 * Author: xumengjun 
 * 
 */

#ifndef EXEC_SERVER_IMPL_ZMQ_H
#define EXEC_SERVER_IMPL_ZMQ_H
#include <map>
#include "cm/include/method.h"
#include "idl/generated/emPubSubTypes.h"
#include "em/include/logger.h"
#include "zmq_ipc/manager/zmq_ipc_server.h"

using namespace std;	
using namespace hozon::netaos::cm;

namespace hozon {
namespace netaos{
namespace em{

class ExecServerImplZmq {
public:

	class ExecCMServer final : public hozon::netaos::zmqipc::ZmqIpcServer {
		public:
		ExecCMServer(ExecServerImplZmq* server) : exec_server(server){};
        ~ExecCMServer(){};
        virtual int32_t Process(const std::string& request, std::string& reply) {
            std::lock_guard<std::mutex> lck(mtx_);
            return exec_server->RequestProcess(request, reply);
        }
        private:
        std::mutex mtx_;
		ExecServerImplZmq * exec_server;
	};

    ExecServerImplZmq(): exec_zmq_server(this) {}
    ~ExecServerImplZmq(){};
    int32_t Start();
    void Stop();
    int32_t RequestProcess(const std::string& request, std::string& reply);

private:
	std::string GetProcessName();

private:
	ExecCMServer exec_zmq_server;
    const std::string em_service_name_ = "tcp://*:11151";
};
} // namespace em
} // namespace netaos
} // namespace hozon
#endif
