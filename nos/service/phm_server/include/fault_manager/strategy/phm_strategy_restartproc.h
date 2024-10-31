/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: fault strategy restart process
 */

#ifndef PHM_STRATEGY_RESTARTPROC_H
#define PHM_STRATEGY_RESTARTPROC_H

#include "phm_server/include/common/time_manager.h"
#include "phm_server/include/common/phm_server_def.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_base.h"
#include "sm/include/state_client.h"
#include <unordered_set>


namespace hozon {
namespace netaos {
namespace phm_server {


class PhmStrategyRestartProc : public StrategyBase {

public:

    PhmStrategyRestartProc();
    virtual ~PhmStrategyRestartProc() {};
    void Act(const FaultInfo& faultData) override;
    void Init() override;
    void DeInit() override;
    void ResetRestartCount();

private:
    PhmStrategyRestartProc(const PhmStrategyRestartProc &);
    PhmStrategyRestartProc & operator = (const PhmStrategyRestartProc &);

    void RestartProcess(const std::string& procName);
    void ProcessMonitorTimeout(void * data);

private:
    std::unordered_map<std::string, int32_t> m_retry_map;
    std::unordered_set<std::string> m_restart_cache_set;
    std::unordered_set<std::string> m_restart_process_set;
    std::mutex m_mtx;
    uint32_t m_config_restart_state; // occur[1], recover[0]
    bool m_timer_start_flag;
    int m_process_monitor_fd;
    std::shared_ptr<TimerManager> time_mgr_;
    std::shared_ptr<hozon::netaos::sm::StateClient> m_state_client;
};


}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon

#endif  // PHM_STRATEGY_RESTARTPROC_H
