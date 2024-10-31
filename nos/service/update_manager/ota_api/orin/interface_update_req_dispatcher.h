/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateTMTaskDelayTimer Header
 */
#ifndef INTERFACE_UPDATE_REQ_DISPATCHER_H_
#define INTERFACE_UPDATE_REQ_DISPATCHER_H_

#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

class InterfaceUpdateReqDispatcher {
public:
    InterfaceUpdateReqDispatcher();
    ~InterfaceUpdateReqDispatcher();

    int32_t Init();
    void Deinit();
    int32_t Update(const std::string& packageName);
    int32_t Query(std::string& updateStatus, uint8_t& progress);
    int32_t QueryUpdateProgress(uint8_t& progress);
    int32_t QueryUpdateStatus(std::string& updateStatus);
    int32_t SwitchSlot();
    int32_t GetCurrentSlot(std::string& currentSlot);
    int32_t Reboot();
    int32_t GetVersionInfo(std::string& version);
};


}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // INTERFACE_UPDATE_REQ_DISPATCHER_H_
