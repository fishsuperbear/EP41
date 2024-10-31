/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateTMTaskDelayTimer Header
 */
#ifndef INTERFACE_UPDATE_REQ_DISPATCHER_H_
#define INTERFACE_UPDATE_REQ_DISPATCHER_H_

#include <string>
#include "update_manager/common/data_def.h"

#ifdef BUILD_FOR_MDC
    #include "self_update_proxy.h"
#endif

#ifndef BUILD_FOR_MDC
    #include "self_update_proxy_default.h"
#endif


namespace hozon {
namespace netaos {
namespace update {



class InterfaceUpdateReqDispatcher {
public:
    InterfaceUpdateReqDispatcher();
    ~InterfaceUpdateReqDispatcher();

    int32_t Init();
    void Deinit();
    int32_t GetVersionInfo(std::string& mdcVersion);
    int32_t Update(const std::string& packageName);
    int32_t Activate();

    // kIdle = 0
    // kReady = 1
    // kBusy = 2
    // kActivating = 3
    // kActivated = 4
    // kRollingBack = 5（暂未支持）
    // kRolledBack = 6（暂未支持）
    // kCleaning_Up = 7
    // kVerifying = 8

    /* IDLE & READY can update */
    /* BUSY is updating */
    /* Activate() can be called after READY */
    int32_t Query(std::string& updateStatus);

    /* If the upgrade fails, still call finish() */
    /* If the current state is VERIFING after restarting, wait until ACTIVATED */
    /* If the current status is ACTIVATED, need to call finish() */
    int32_t Finish();

    int32_t GetActivationProgress(uint8_t& progress, std::string& message);

    int32_t GetUpdateProgress(uint8_t& progress, std::string& message);

private:
    std::shared_ptr<hozon::swm::SelfUpdateProxy> proxy_;

};


}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // INTERFACE_UPDATE_REQ_DISPATCHER_H_
