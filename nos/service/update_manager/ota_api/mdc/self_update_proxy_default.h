/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Self updat proxy empty implementation

 */
#ifndef SELF_UPDATE_PROXY_DEFAULT_H
#define SELF_UPDATE_PROXY_DEFAULT_H

#include <iostream>
#include <unistd.h>
#include <memory>

namespace hozon {
namespace swm {

class SelfUpdateProxy {
public:
    SelfUpdateProxy() {}
    ~SelfUpdateProxy() {}
    int32_t Init() {return 0;}
    void DeInit() {}
    int32_t GetVersion(std::string& mdcVersion) {return 0;}
    int32_t Update(std::string& packageName) {return 0;}
    int32_t GetUpdateProgress(uint8_t& progress, std::string& message) {return 0;}
    int32_t Activate() {return 0;}
    int32_t GetActivationProgress(uint8_t& progress, std::string& message) {return 0;}

    /* IDLE & READY can update */
    /* BUSY is updating */
    /* Activate() can be called after READY */
    int32_t Query(std::string& updateStatus) {return 0;}

    /* If the upgrade fails, still call finish() */
    /* If the current state is VERIFING after restarting, wait until ACTIVATED */
    /* If the current status is ACTIVATED, need to call finish() */
    int32_t Finish() {return 0;}
};
    
}  // namespace swm
}  // namespace hozon
#endif  // SELF_UPDATE_PROXY_DEFAULT_H
