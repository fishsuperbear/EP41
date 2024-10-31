/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: dc_config_client.cpp
 * @Date: 2023/12/11
 * @Author: cheng
 * @Desc: --
 */
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>


#include "zmq_ipc/manager/zmq_ipc_client.h"

#include "zmq/zmq.hpp"
#include "log/include/default_logger.h"
std::atomic_bool stopFlag_ = false;
void SigHandler(int signum)
{
    DEBUG_LOG("sigHandler signum: %d", signum);
    stopFlag_ = true;
}
int main(int argc, char ** argv) {
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    std::atomic_int count{0};
    std::cout<<"client: "<<__LINE__<<std::endl;
    std::string dc_config_service_name = "tcp://localhost:90332";
     std::unique_ptr<hozon::netaos::zmqipc::ZmqIpcClient>   client_ = std::make_unique<hozon::netaos::zmqipc::ZmqIpcClient>();
     client_->Init(dc_config_service_name);
     while (!stopFlag_) {
         std::string res;
         client_->Request(std::to_string(++count),res,1000);
         std::cout<<count<<" --Receive: "<<res<<std::endl;
         std::this_thread::sleep_for(std::chrono::milliseconds(4000));
     }
     std::cout<<"befor Deinit"<<std::endl;
     client_->Deinit();
     std::cout<<"after Deinit"<<std::endl;
    return 0;
}
