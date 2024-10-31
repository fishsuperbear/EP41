/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: ssm
 * Created on: Aug 15, 2023
 * Author: aviroz
 */

#ifndef SYS_MANAGER_H
#define SYS_MANAGER_H

#include "sys_statemgr/include/dispatcher.h"
#include "sys_statemgr/include/state_manager.h"
#include "sys_statemgr/include/power_manager_svr.h"
#include "sys_statemgr/include/mcustate_service_cli.h"
#include "sys_statemgr/include/sys_state_manager_server.h"

namespace hozon {
namespace netaos {
namespace ssm {

class SysManager {
public:

    SysManager();
    virtual ~SysManager();

    int32_t Init();
    void DeInit();
    void Run();

    std::shared_ptr<McuStateServiceCli> McuStateService();

private:
    std::shared_ptr<Dispatcher> disptch;
    std::shared_ptr<StateManager> statmgr;
    std::shared_ptr<PowerManagerServiceSvr> spmsvr;
    std::shared_ptr<McuStateServiceCli> msscli;
    std::shared_ptr<SysStateMgrServer> sysstatemgrsvr;
};

}}}
#endif
