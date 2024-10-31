/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: phm fault dispatcher
 */
#ifndef PHM_FAULT_DISPATCHER_H
#define PHM_FAULT_DISPATCHER_H

#include <mutex>
#include <functional>
#include <unordered_map>
#include "proxy.h"
#include "skeleton.h"
#include "phmPubSubTypes.h"
#include "phm_server/include/common/phm_server_def.h"
#include "phm_server/include/common/time_manager.h"
#include <signal.h>

namespace hozon {
namespace netaos {
namespace phm_server {

using namespace hozon::netaos::cm;

class PhmFaultMethodRecive;
class FaultDispatcher {

public:
    static FaultDispatcher* getInstance();

    void Init();
    void DeInit();

    void SendFault(const SendFaultPack& fault);
    void QuerySystemCheckFault(std::vector<SystemCheckFaultInfo>& faultList);
    void ReportFault(const Fault_t& fault);
    void SendInhibitType(const uint32_t type);

private:
    FaultDispatcher();

    void ReceiveFault();
    void SystemCheck();
    void SystemCheckCompletedCallback(void* data);
    void AddSystemCheckFault(const Fault_t& fault);

private:
    static std::mutex mtx_;
    static FaultDispatcher* instance_;

    std::shared_ptr<Proxy> proxy_;
    std::shared_ptr<Skeleton> skeleton_;
    std::shared_ptr<Skeleton> inhibit_skeleton_;

    // system check
    std::shared_ptr<TimerManager> time_mgr_;
    int system_check_timer_fd_;
    bool system_check_completed_;
    std::unordered_map<uint32_t, SystemCheckFaultInfo> system_check_fault_list_;
    std::shared_ptr<PhmFaultMethodRecive> m_spPhmFaultMethodRecive;
    uint32_t inhibitType_;
    std::unordered_map<uint32_t, uint8_t> fault_status_maps_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_FAULT_DISPATCHER_H
