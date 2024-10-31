/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: dc_config_server.cpp
 * @Date: 2023/12/11
 * @Author: cheng
 * @Desc: --
 */
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>

#include "log/include/default_logger.h"

#include "zmq/zmq.hpp"
#include "tsp_comm.h"

#include "zmq_ipc/manager/zmq_ipc_server.h"

std::atomic_int count{0};

using tspCommon = hozon::netaos::https::TspComm;
class DcConfigServerImpl final : public hozon::netaos::zmqipc::ZmqIpcServer
{
    
   public:
    DcConfigServerImpl()
        : hozon::netaos::zmqipc::ZmqIpcServer()
    {
    }

    virtual ~DcConfigServerImpl(){};

    virtual int32_t Process(const std::string& request, std::string& reply)
    {
        std::future<tspCommon::TspResponse> ret_remotecfg = tspCommon::GetInstance().RequestRemoteConfig();
        auto ret_request = ret_remotecfg.get();
        std::cout << "result_code:" << ret_request.result_code << " remoteconfig:" << ret_request.response << std::endl;
        reply = "test"+request;
        std::cout<<"Server------recv:"<<request<<std::endl;
//        DEBUG_LOG("Server recv request: %s, reply: %s", request.c_str(), reply.c_str());
        return reply.size();
    };

   private:

};

std::atomic_bool stopFlag_ = false;
void SigHandler(int signum)
{
    std::cout<<"sigHandler signum: "<<signum<<std::endl;
    stopFlag_ = true;
}
int main(int argc, char ** argv) {
    hozon::netaos::log::InitLogging("DCTEST", "NETAOS DC", hozon::netaos::log::LogLevel::kTrace,
                                    hozon::netaos::log::HZ_LOG2CONSOLE , "/opt/usr/log/soc_log/", 10, (20*1024*1024),true);
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);
    tspCommon::GetInstance().Init();

     std::string dc_config_service_name = "tcp://*:90332";
     std::unique_ptr<DcConfigServerImpl> server_ = std::make_unique<DcConfigServerImpl>();
     server_->Start(dc_config_service_name);

    while (!stopFlag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::cout<<"server->Stop before!~"<<std::endl;
     server_->Stop();
     tspCommon::GetInstance().Deinit();
     std::cout<<"server->Stop after!~"<<std::endl;
    return 0;
}