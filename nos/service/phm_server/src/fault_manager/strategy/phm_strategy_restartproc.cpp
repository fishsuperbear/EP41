/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: phm fault strategy restart process
*/

#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_restartproc.h"

namespace hozon {
namespace netaos {
namespace phm_server {

const unsigned int PROCESS_MONITOR_TIME = 3000; // 3s
const std::string PRIOR_PROCESS_CONFIG_SERVER = "hz_configServerProcess";

PhmStrategyRestartProc::PhmStrategyRestartProc()
: m_config_restart_state(0)
, m_timer_start_flag(false)
, m_process_monitor_fd(-1)
, time_mgr_(new TimerManager())
, m_state_client(new hozon::netaos::sm::StateClient())
{
}

void PhmStrategyRestartProc::Init()
{
    if (time_mgr_ != nullptr) {
        time_mgr_->Init();
    }
}

void PhmStrategyRestartProc::DeInit()
{
    if (time_mgr_ != nullptr) {
        PHMS_INFO << "PhmStrategyRestartProc::DeInit fd:" << m_process_monitor_fd;
        time_mgr_->StopFdTimer(m_process_monitor_fd);
        time_mgr_->DeInit();
    }
}

void
PhmStrategyRestartProc::ResetRestartCount()
{
    PHMS_INFO << "PhmStrategyRestartProc::ResetRestartCount";
    std::lock_guard<std::mutex> lck(m_mtx);
    m_retry_map.clear();
}

void
PhmStrategyRestartProc::Act(const FaultInfo& faultData)
{
    PHMS_INFO << "PhmStrategyRestartProc::Act";
    std::lock_guard<std::mutex> lck(m_mtx);
    const PhmConfigInfo& configInfo = PHMServerConfig::getInstance()->GetPhmConfigInfo();
    if ("on" != configInfo.RestartProcSwitch) {
        PHMS_INFO << "PhmStrategyRestartProc::Act sw is off";
        return;
    }

    // if (!HzSmService::Instance().GetFucGroupState()) {
    //     return;
    // }

    const std::string processName = PHMServerConfig::getInstance()->GetProcInfoByFaultKey(faultData.faultId, faultData.faultObj);
    if (processName.empty()) {
        return;
    }

    PHMS_DEBUG << "PhmStrategyRestartProc::Act process " << processName << " status " << faultData.faultStatus;
    if (processName == PRIOR_PROCESS_CONFIG_SERVER) {
        if (faultData.faultStatus == 1) {
            PHMS_DEBUG << "PhmStrategyRestartProc::Act config server terminated";
            // set config server fault occur
            m_config_restart_state = 1;

            // stop timer
            m_timer_start_flag = false;
            time_mgr_->StopFdTimer(m_process_monitor_fd);
            PHMS_DEBUG << "PhmStrategyRestartProc::Act config server terminated stop timer " << m_process_monitor_fd;

            // copy process_set to cache_set
            m_restart_cache_set.insert(m_restart_process_set.begin(), m_restart_process_set.end());
            m_restart_process_set.clear();

            // restart config
            RestartProcess(PRIOR_PROCESS_CONFIG_SERVER);
        }
        else {
            PHMS_DEBUG << "PhmStrategyRestartProc::Act config server startup";
            // set config server fault recover
            m_config_restart_state = 0;

            // restart cache_set process
            PHMS_DEBUG << "PhmStrategyRestartProc::Act m_restart_cache_set size " << m_restart_cache_set.size();
            for (auto& item : m_restart_cache_set) {
                RestartProcess(item);
            }
            m_restart_cache_set.clear();
        }

        return;
    }

    if (faultData.faultStatus == 0) {
        PHMS_DEBUG << "PhmStrategyRestartProc::Act recover and discard fault";
        return;
    }

    if (m_config_restart_state == 1) {
        PHMS_DEBUG << "PhmStrategyRestartProc::Act config server is restarting, restart " << processName << " waiting...";
        m_restart_cache_set.emplace(processName);
        return;
    }

    if (!m_timer_start_flag) {
        // start timer monitor config server crash
        PHMS_DEBUG << "PhmStrategyRestartProc::Act start timer to record process start";
        m_timer_start_flag = true;
        time_mgr_->StartFdTimer(m_process_monitor_fd, PROCESS_MONITOR_TIME,
            std::bind(&PhmStrategyRestartProc::ProcessMonitorTimeout, this, std::placeholders::_1), NULL, false);
        PHMS_DEBUG << "PhmStrategyRestartProc::Act start timer to record process end " << m_process_monitor_fd;
    }

    if ("nvs_producerProcess" == processName
        || "camera_vencProcess" == processName) {
        PHMS_INFO << "PhmStrategyRestartProc::Act restart 4 process";
        m_restart_process_set.emplace("nvs_producerProcess");
        m_restart_process_set.emplace("camera_vencProcess");
        m_restart_process_set.emplace("perceptionProcess");
        m_restart_process_set.emplace("fisheye_perceptionProcess");
        return;
    }

    m_restart_process_set.emplace(processName);
    return;
}

void
PhmStrategyRestartProc::ProcessMonitorTimeout(void * data)
{
    PHMS_DEBUG << "PhmStrategyRestartProc::ProcessMonitorTimeout";
    std::lock_guard<std::mutex> lck(m_mtx);
    for (auto& item : m_restart_process_set) {
        RestartProcess(item);
    }

    m_restart_process_set.clear();
    m_timer_start_flag = false;
    PHMS_INFO << "PhmStrategyRestartProc::ProcessMonitorTimeout fd:" << m_process_monitor_fd;
    time_mgr_->StopFdTimer(m_process_monitor_fd);
}

void
PhmStrategyRestartProc::RestartProcess(const std::string& procName)
{
    PHMS_DEBUG << "PhmStrategyRestartProc::RestartProcess process " << procName;
    if (procName != PRIOR_PROCESS_CONFIG_SERVER) {
        auto iter = m_retry_map.find(procName);
        if (iter == m_retry_map.end()) {
            uint8_t retryCount = PHMServerConfig::getInstance()->getProcRetryCountByName(procName);
            m_retry_map.insert({procName, retryCount});
        }
        else {
            m_retry_map.at(procName) = iter->second - 1;
        }

        if (m_retry_map.at(procName) <= 0) {
            PHMS_INFO << "StrategyRestartProc::RestartProcess restart over count, return";
            return;
        }

        PHMS_INFO << "PhmStrategyRestartProc::RestartProcess restart rest count: " << m_retry_map.at(procName);
    }

    m_state_client->ProcRestart(procName);
    return;
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
