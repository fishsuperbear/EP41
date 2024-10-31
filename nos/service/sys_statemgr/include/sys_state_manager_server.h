/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: ssm
 * Created on: Nov 22, 2023
 * Author: xumengjun
 */

#ifndef __SYS_STATE_MGR_SERVER_H__
#define __SYS_STATE_MGR_SERVER_H__

#include <mutex>
#include <string>

#include "zmq_ipc/manager/zmq_ipc_server.h"

namespace hozon {
namespace netaos {
namespace ssm {

class SysManager;
class SysStateMgrServer {
public:
    SysStateMgrServer(SysManager *sys_manager); 
    ~SysStateMgrServer();

public:
    int32_t Start();
    void Stop();

    int32_t RequestProcess(const std::string& request, std::string& reply);

public:
    class ZmqServer final : public hozon::netaos::zmqipc::ZmqIpcServer {
        public:
            ZmqServer(SysStateMgrServer *server) : sys_state_mgr_server_(server) {}
            virtual ~ZmqServer() {}
            virtual int32_t Process(const std::string &request, std::string &reply) {
                std::lock_guard<std::mutex> lck(mtx_);
                return sys_state_mgr_server_->RequestProcess(request, reply);
            }

        private:
            std::mutex mtx_;
            SysStateMgrServer *sys_state_mgr_server_;
    };
    
private:
    SysManager *sys_manager_;
    ZmqServer sys_stat_mgr_zmq_server_;

    const std::string sys_stat_mgr_service_name_ = "tcp://*:11155";
};

} // namespace ssm
} // namespace netaos
} // namespace hozon

#endif // __SYS_STATE_MGR_SERVER_H__

