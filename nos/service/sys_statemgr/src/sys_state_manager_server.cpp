/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: ssm
 * Created on: Nov 22, 2023
 * Author: xumengjun
 */

#include "sys_statemgr/include/sys_state_manager_server.h"
#include "sys_statemgr/include/sys_manager.h"
#include "sys_statemgr/include/logger.h"

namespace hozon {
namespace netaos {
namespace ssm {

SysStateMgrServer::SysStateMgrServer(SysManager *sys_manager) : sys_manager_(sys_manager), sys_stat_mgr_zmq_server_(this) {
}

SysStateMgrServer::~SysStateMgrServer() {
}

int32_t SysStateMgrServer::Start() {
    SSM_LOG_INFO << __func__;
    sys_stat_mgr_zmq_server_.Start(sys_stat_mgr_service_name_);
    return 0;
}

void SysStateMgrServer::Stop() {
    SSM_LOG_INFO << __func__;
    sys_stat_mgr_zmq_server_.Stop();
}

int32_t SysStateMgrServer::RequestProcess(const std::string &request, std::string &reply) {
    SSM_LOG_INFO << "RequestProcess: " << request;

    reply = "failed";
    if (request == "reboot_soc") {
        hozon::netaos::PowerModeEnum power_mode = hozon::netaos::PowerModeEnum::Restart;
        if (sys_manager_->McuStateService()->SocPowerModeRequest(power_mode) == 0) {
            reply = "success";
        }
    } else if (request == "reboot_orin") {
        hozon::netaos::PowerModeEnum power_mode = hozon::netaos::PowerModeEnum::Reset;
        if (sys_manager_->McuStateService()->SocPowerModeRequest(power_mode) == 0) {
            reply = "success";
        }
    }

    return 0;
}

} // namespace hozon
} // namespace netaos
} // namespace ssm
