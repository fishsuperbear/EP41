/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: ssm
 * Created on: July 14, 2023
 * Author: aviroz
 */

#ifndef STATE_MANAGER_H
#define STATE_MANAGER_H

#include <signal.h>
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <thread>
#include <shared_mutex>
#include "sm/include/state_client_zmq.h"
#include "cfg/include/config_param.h"

namespace hozon {
namespace netaos {
namespace ssm {


using namespace hozon::netaos::sm;
using namespace hozon::netaos::cfg;

class StateManager {
public:
    StateManager();
    virtual ~StateManager();

    int32_t Init();
    void DeInit();
    void Run();
    int32_t StopMode();
    int32_t SwitchMode(const std::string&);
    /* soc state */
    uint8_t GetSysState();
    uint8_t GetSMState();
    /* mcu state */
    uint8_t GetMcuL2State();
    uint8_t GetMcuL3tate();
    void SetMcuL2State(uint8_t);
    void SetMcuL3State(uint8_t);

private:
    /* soc state */
    void SetSysState(uint8_t, const std::string&);
    void SetSMState(uint8_t);
    /* cfg monitor */
    void OnRecvSMChangedCallback(const std::string&, const std::string&, const uint8_t&);
    void OnRecvModeChangedCallback(const std::string&, const std::string&, const std::string&);

    void SysStateSyncMonitor();

private:
    std::shared_ptr<StateClientZmq> m_sm_cli;
    mutable std::shared_timed_mutex m_mutex_sstate;
    uint8_t m_sys_state;
    mutable std::shared_timed_mutex m_mutex_mstate;
    uint8_t m_mach_state;
    mutable std::shared_timed_mutex m_mutex_l2state;
    uint8_t m_l2_state;
    mutable std::shared_timed_mutex m_mutex_l3state;
    uint8_t m_l3_state;
    sig_atomic_t m_stopFlag;
    std::thread m_stat_thr;
    std::thread m_sync_thr;
};

}}}
#endif
