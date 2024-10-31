/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: em
 * Created on: Nov 23, 2022
 * Author: shefei
 * 
 */

#ifndef PROCESS_H
#define PROCESS_H

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <shared_mutex>
#include <signal.h>
#include "em/include/define.h"

namespace hozon {
namespace netaos {
namespace em {

using namespace hozon::netaos::em;

class Process {
public: 

    struct Schedule {
       uint32_t policy;
       uint32_t priority;
       cpu_set_t affinity;
       uint32_t iscpuset;
    };

    Process();
    virtual ~Process();

    int32_t ParseManifest(const std::string& file);
    int32_t Start(bool retry = true);
    int32_t Restart();
    int32_t Terminate(bool async = true);

    ProcessState GetState() const;
    int32_t WaitState(ProcessState target_state, uint32_t timeout_ms);

    int32_t GetOrderOfMode(const std::string& mode, uint32_t* order);
    void SetExecState(ExecutionState state);

private:
    inline void RedirectLog();
    inline int32_t SetCpuUtility();
    std::string ConfigLoader(const std::string& fpath);
    int32_t ParseModeOrder(std::vector<std::string>*);
    void SetState(ProcessState state);
    pid_t GetProcPid() const;
    void SetProcPid(pid_t pid);
    void MonitorPid();
    void KeepAlive(uint32_t *times);
    char **ArgMalloc(uint32_t *size);
    char **EnvMalloc(uint32_t *size);
    void MalcFree(char **argv, uint32_t size);

public:
    std::string m_process_name;
    ProcessState m_proc_state;
    uint32_t m_enter_timeout;
    uint32_t m_exit_timeout;
    uint32_t m_restart_attempt_num;
    std::vector<ModeOrder> m_order_mode_vec;
    std::vector<std::string> m_exp_environments;

private:
    pid_t m_pid;
    Schedule m_sched;
    sig_atomic_t m_alive;
    ExecutionState m_exec_state;
    std::string m_exec_filepath;
    std::string m_envrion_name;

    std::string m_executable;
    uint32_t m_keep_alive_num;
    std::vector<uint32_t> shall_run_on;
    std::vector<std::string> m_exec_order_mode;
    std::string m_sched_policy;
    uint32_t m_sched_priority;
    std::vector<std::string> m_arguments;
    std::vector<std::string> m_environments;
    mutable std::shared_timed_mutex m_smutex_pid;
    mutable std::shared_timed_mutex m_smutex_exec_state;

};

}}}
#endif