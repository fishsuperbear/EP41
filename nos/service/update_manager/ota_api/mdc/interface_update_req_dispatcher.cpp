/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: update interface req dispatcher
*/

#include "interface_update_req_dispatcher.h"
#include "update_manager/log/update_manager_logger.h"


namespace hozon {
namespace netaos {
namespace update {


InterfaceUpdateReqDispatcher::InterfaceUpdateReqDispatcher()
    : proxy_(nullptr)
{
}

InterfaceUpdateReqDispatcher::~InterfaceUpdateReqDispatcher()
{
}

int32_t
InterfaceUpdateReqDispatcher::Init()
{
    int32_t ret = -1;
#ifdef BUILD_FOR_MDC
    proxy_ = std::make_shared<hozon::swm::SelfUpdateProxy>();
#endif
    if (nullptr != proxy_) {
        ret = proxy_->Init();
    }
    return ret;
}

void
InterfaceUpdateReqDispatcher::Deinit()
{
    if (nullptr != proxy_) {
        proxy_->DeInit();
    }
}

int32_t
InterfaceUpdateReqDispatcher::GetVersionInfo(std::string& mdcVersion)
{
    UPDATE_LOG_D("Update interface GetVersion");
    int32_t ret = -1;
    if (nullptr == proxy_) {
        UPDATE_LOG_E("proxy_ is nullptr!");
        return ret;
    }
    return proxy_->GetVersion(mdcVersion);
}

int32_t
InterfaceUpdateReqDispatcher::Update(const std::string& packageName)
{
    UPDATE_LOG_D("Update interface Update packageName: %s", packageName.c_str());
    int32_t ret = -1;
    if (nullptr == proxy_) {
        UPDATE_LOG_E("proxy_ is nullptr!");
        return ret;
    }
    std::string package = packageName;
    return proxy_->Update(package);
}

int32_t
InterfaceUpdateReqDispatcher::Activate()
{
    UPDATE_LOG_D("Update interface Activate...");
    int32_t ret = -1;
    if (nullptr == proxy_) {
        UPDATE_LOG_E("proxy_ is nullptr!");
        return ret;
    }
    return proxy_->Activate();
}

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
int32_t
InterfaceUpdateReqDispatcher::Query(std::string& updateStatus)
{
    int32_t ret = -1;
    if (nullptr == proxy_) {
        UPDATE_LOG_E("proxy_ is nullptr!");
        return ret;
    }
    return proxy_->Query(updateStatus);
}

/* If the upgrade fails, still call finish() */
/* If the current state is VERIFING after restarting, wait until ACTIVATED */
/* If the current status is ACTIVATED, need to call finish() */
int32_t
InterfaceUpdateReqDispatcher::Finish()
{
    int32_t ret = -1;
    if (nullptr == proxy_) {
        UPDATE_LOG_E("proxy_ is nullptr!");
        return ret;
    }
    return proxy_->Finish();
}


int32_t
InterfaceUpdateReqDispatcher::GetActivationProgress(uint8_t& progress, std::string& message)
{
    int32_t ret = -1;
    if (nullptr == proxy_) {
        UPDATE_LOG_E("proxy_ is nullptr!");
        return ret;
    }

    int res = proxy_->GetActivationProgress(progress, message);
    if (res < 0) {
        return ret;
    }
    progress = static_cast<uint8_t>(progress);

    return 0;
}

int32_t
InterfaceUpdateReqDispatcher::GetUpdateProgress(uint8_t& progress, std::string& message)
{
    int32_t ret = -1;
    if (nullptr == proxy_) {
        UPDATE_LOG_E("proxy_ is nullptr!");
        return ret;
    }

    int res = proxy_->GetUpdateProgress(progress, message);
    if (res < 0) {
        return ret;
    }
    progress = static_cast<uint8_t>(progress);
    return 0;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon