/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: em
 * Created on: Nov 23, 2022
 * Author: shefei
 * 
 */

#include <sys/wait.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/select.h>
#include <sys/prctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <cstring>
#include <thread>
#include <algorithm>
#include "em/utils/cJSON.h"
#include "em/include/logger.h"
#include "em/include/process.h"
#include <iostream>

namespace hozon {
namespace netaos {
namespace em {

Process::Process() {
    m_pid = 0;
    m_proc_state = ProcessState::IDLE;
    m_exec_state = ExecutionState::kDefault;
    m_restart_attempt_num = 2;
    m_keep_alive_num = 0;

    m_sched.iscpuset = 0;
    CPU_ZERO(&m_sched.affinity);
    // for (std::size_t i = 0; i < DEF_MAX_CPU_CORE; ++i) {
    //     CPU_SET(i, &m_sched.affinity);
    // }
}

Process::~Process() {
}

int32_t Process::ParseManifest(const std::string& file) {
    LOG_INFO<<">>> "<<file;
    int32_t ret = 0;
    std::string filepath = file;

    std::string str_value = ConfigLoader(file);
    cJSON *key = nullptr;
    cJSON *item = nullptr;
    cJSON *arritem = nullptr;
    cJSON *root = cJSON_Parse(str_value.c_str());
    if (!root) { goto ERROR; }
    if (!cJSON_IsObject(root)) { goto ERROR; }

    key = cJSON_GetObjectItem(root, "process_name");
    if (!key) { goto ERROR;
    } else {
        m_process_name = key->valuestring;
    }

    key = cJSON_GetObjectItem(root, "executable");
    if (!key) { goto ERROR;
    } else {
        m_executable = key->valuestring;
    }

    key = cJSON_GetObjectItem(root, "restart_attempt_num");
    if (!key) { goto ERROR;
    } else {
        m_restart_attempt_num = std::atoi(key->valuestring);
        if (m_restart_attempt_num < 0 || m_restart_attempt_num > 3) {
            m_restart_attempt_num = 2;
            LOG_WARN <<"restart attempt num should be from 0 to 3.";
        }
    }

    key = cJSON_GetObjectItem(root, "keep_alive_num");
    if (!key) {
        LOG_DEBUG<<"keep_alive_num not found";//goto ERROR;
    } else {
        m_keep_alive_num = std::atoi(key->valuestring);
    }

    key = cJSON_GetObjectItem(root, "scheduling_policy");
    if (!key) { goto ERROR;
    } else {
        m_sched_policy = key->valuestring;
        if (m_sched_policy == "RR") {
            m_sched.policy = SCHED_RR;
        } else if (m_sched_policy == "FIFO") {
            m_sched.policy = SCHED_FIFO;
        } else if (m_sched_policy == "OTHER") {
            m_sched.policy = SCHED_OTHER;
        } else {
            m_sched.policy = SCHED_OTHER;
            LOG_WARN<<"Unsupported scheduler policy,using default policy.";
        }
    }

    key = cJSON_GetObjectItem(root, "scheduling_priority");
    if (!key) { goto ERROR;
    } else {
        m_sched_priority = std::atoi(key->valuestring);
        if (m_sched_priority < 0 || m_sched_priority > 99) {
            m_sched.priority = 0;
            LOG_WARN << "schedule priority should be from 0 to 99.";
        } else {
            m_sched.priority = m_sched_priority;
        }
    }

    item = cJSON_GetObjectItem(root, "shall_run_ons");
    if (!item) { goto ERROR;
    } else {
        uint32_t arrsize_i = cJSON_GetArraySize(item);
        if (arrsize_i > 0) {
            m_sched.iscpuset = 1;
            CPU_ZERO(&m_sched.affinity);
        }
        for (size_t i = 0; i < arrsize_i; i++) {
            arritem = cJSON_GetArrayItem(item, i);
            if (arritem) {
                shall_run_on.push_back(arritem->valueint);
                std::sort(shall_run_on.begin(),shall_run_on.end());
                for (auto item : shall_run_on) {
                    CPU_SET((uint32_t)item, &m_sched.affinity);
                }
            }
        }
    }

    item = cJSON_GetObjectItem(root, "mode_of_order");
    if (!item) { goto ERROR;
    } else {
        uint32_t arrsize_j = cJSON_GetArraySize(item);
        for (size_t j = 0; j < arrsize_j; j++) {
            arritem = cJSON_GetArrayItem(item, j);
            if (arritem) {
                m_exec_order_mode.push_back(arritem->valuestring);
            }
        }
    }

    item = cJSON_GetObjectItem(root, "arguments");
    if (!item) { goto ERROR;
    } else {
        uint32_t arrsize_k = cJSON_GetArraySize(item);
        for (size_t k=0; k<arrsize_k; k++) {
            arritem = cJSON_GetArrayItem(item, k);
            if (arritem) {
                m_arguments.push_back(arritem->valuestring);
            }
        }
    }

    item = cJSON_GetObjectItem(root, "environments");
    if (!item) { goto ERROR;
    } else {
        uint32_t arrsize_l = cJSON_GetArraySize(item);
        for (size_t l = 0; l < arrsize_l; l++) {
            arritem = cJSON_GetArrayItem(item, l);
            if (arritem) {
                m_environments.push_back(arritem->valuestring);
            }
        }
    }

    if (root) {
        cJSON_Delete(root);
    }

    /* set exec file path */
    filepath.erase(filepath.find_last_of("/etc/")-4,std::string::npos);
    m_exec_filepath += filepath + "/bin/" + m_executable;

    m_envrion_name = ENVRION_NAME + m_process_name;

    /* parse mode order */
    ret = ParseModeOrder(&m_exec_order_mode);

    return ret;

ERROR:
    LOG_ERROR<<"key or object not found";
    if (root) {
        cJSON_Delete(root);
    }
    return -1;
}

int32_t Process::Start(bool retry) {
    m_alive = 1;
    SetState(ProcessState::IDLE);
    SetExecState(ExecutionState::kDefault);
    std::thread([this](bool rtry) {
        std::string thr_name = "thr_" + m_executable;
        prctl(PR_SET_NAME, thr_name.c_str());

        int32_t ret = SetCpuUtility();
        if (0 != ret) {
            LOG_ERROR<<"set cpu policy fail, errcode:"<<ret;
        }

        SetProcPid(0);
        int32_t count = (rtry == true ? (m_restart_attempt_num + 1) : 1);

        uint32_t arg_size = 0;
        uint32_t env_size = 0;
        char ** argv = ArgMalloc(&arg_size);
        char ** envr = EnvMalloc(&env_size);

        while (count-- > 0) {
            pid_t pid = fork();
            SetProcPid(pid);
            if (GetProcPid() < (pid_t)0) {
                LOG_WARN<<"fork fail";
                continue;
            } else if (GetProcPid() == (pid_t)0) {
                SetProcPid(getpid());
                RedirectLog();
                if (execve(m_exec_filepath.c_str(), argv, envr) < 0) {
                    pause();
                }
                exit(EXIT_FAILURE);
            } else {
                SetState(ProcessState::STARTING);
                if (0 == WaitState(ProcessState::RUNNING,m_enter_timeout)) {
                    break;
                } else {
                    kill(GetProcPid(),SIGKILL);
                    int wstatus = 0;
                    waitpid(GetProcPid(), &wstatus, 0);
                    SetProcPid(-1);
                    SetState(ProcessState::IDLE);

                    if (count > 0) {
                        LOG_INFO<<"restart "<< m_process_name << " times:" << m_restart_attempt_num + 1 - count;
                        continue;
                    }
                }
                break;
            }
        }

        MalcFree(argv,arg_size);
        MalcFree(envr,env_size);

        if (GetProcPid() > 0) {
            MonitorPid();
        } else {
            LOG_ERROR<<"< start proc "<< m_process_name <<" fail >";
            SetState(ProcessState::ABORTED);
        }
        // KeepAlive(&m_keep_alive_num);
    },retry).detach();

    return 0;
}

char** Process::ArgMalloc(uint32_t *size) {
    *size = m_arguments.size() + 2;
    uint32_t idx = 0;
    char **argv = new char*[*size];

    argv[idx] = new char[m_exec_filepath.size()+1];
    memset(argv[idx],0x00,m_exec_filepath.size()+1);
    strncpy(argv[idx],m_exec_filepath.c_str(),m_exec_filepath.size());

    for (auto item : m_arguments) {
        argv[++idx] = new char[item.size()+1];
        memset(argv[idx],0x00,item.size()+1);
        strncpy(argv[idx],item.c_str(),item.size());
    }
    argv[++idx] = nullptr;
    return argv;
}

char** Process::EnvMalloc(uint32_t *size) {
    *size = m_exp_environments.size() + m_environments.size() + 2;
    uint32_t idx = 0;
    char **env = new char *[*size];
    env[idx] = new char[m_envrion_name.size()+1];
    memset(env[idx],0x00,m_envrion_name.size()+1);
    strncpy(env[idx],m_envrion_name.c_str(),m_envrion_name.size());

    for (auto item : m_exp_environments) {
        env[++idx] = new char[item.size()+1];
        memset(env[idx],0x00,item.size()+1);
        strncpy(env[idx],item.c_str(),item.size());
    }
    for (auto item : m_environments) {
        env[++idx] = new char[item.size()+1];
        memset(env[idx],0x00,item.size()+1);
        strncpy(env[idx],item.c_str(),item.size());
    }
    env[++idx] = nullptr;
    return env;
}

void Process::MalcFree(char **argv, uint32_t size) {
    if (argv) {
        for (uint32_t i = 0; i < size; i++) {
            delete []argv[i];
        }
        delete argv;
    }
}

void Process::MonitorPid() {
    int wstatus = 0;
    waitpid(GetProcPid(), &wstatus, 0);
    char str[40] = {0};
    if (WIFEXITED(wstatus)) {
        sprintf(str,"[%d] exit with code [%d]", GetProcPid(), WEXITSTATUS(wstatus));
        if (WEXITSTATUS(wstatus) == 0) {
            if (GetState() == ProcessState::TERMINATING || GetState() == ProcessState::RUNNING) {
                SetState(ProcessState::TERMINATED);
                LOG_INFO<<"procname:"<< m_process_name <<" exit";
            } else {
                SetState(ProcessState::ABORTED);
                LOG_INFO<<"procname:"<<m_process_name<<"["<<GetProcPid() <<"] aborted cur state "<<(int32_t)GetState();
            }
        } else {
            SetState(ProcessState::ABORTED);
            LOG_WARN<<"procname:"<< m_process_name <<str;
        }
    } else if (WIFSIGNALED(wstatus)) {
        sprintf(str,"[%d] recv sig [%d] to exit", GetProcPid(), WTERMSIG(wstatus));
        if (6 == WTERMSIG(wstatus) && GetState() == ProcessState::TERMINATING) {
            SetState(ProcessState::TERMINATED);
            LOG_INFO<<"procname:"<< m_process_name <<" exit.";
        } else {
            SetState(ProcessState::ABORTED);
            LOG_INFO<<"procname:"<< m_process_name <<str;
        }
    } else {
        SetState(ProcessState::ABORTED);
        LOG_WARN<<"procname:"<< m_process_name <<"["<<GetProcPid() <<"] unknow exit";
    }
    SetProcPid(-1);
}

int32_t Process::Restart() {
    int32_t ret = 0;
    if (GetProcPid() > 0) {
        Terminate(false);
        ret = WaitState(ProcessState::TERMINATED,m_exit_timeout);
    }
    Start(false);
    ret = WaitState(ProcessState::RUNNING,m_enter_timeout*(m_restart_attempt_num+1));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return ret;
}

int32_t Process::Terminate(bool async) {
    m_alive = 0;
    std::thread thrd([this]() {
        pid_t pid = GetProcPid();
        if (pid > 0) {
            kill(pid,SIGTERM);
            if (WaitState(ProcessState::TERMINATED,m_exit_timeout) != 0) {
                kill(pid,SIGKILL);
            }
        }
    });

    if (async) {
        thrd.detach();
    } else {
        if (thrd.joinable()) { thrd.join(); }
    }

    return 0;
}

void Process::KeepAlive(uint32_t *times) {
    if (m_alive && m_keep_alive_num > 0) {
        LOG_DEBUG<<"times:"<<*times;
        if ( (*times)-- > 0 ) {
            std::this_thread::sleep_for(std::chrono::milliseconds(PROC_KEEP_ALIVE_PERIOD));
            Start(false);
        }
    }
}

void Process::RedirectLog() {
    int fd = open("/dev/null", O_WRONLY);
    dup2(fd, 1);
}

int32_t Process::WaitState(ProcessState target_state, uint32_t timeout_ms) {
    int32_t ret = 0;
    int32_t count = timeout_ms / PROC_STATE_DETECT_PERIOD;

    pid_t pid = GetProcPid();

    switch (target_state) {
    case ProcessState::RUNNING:
    case ProcessState::TERMINATING:{
        for ( ; count >= 0; count--) {
            if (target_state == GetState() || ProcessState::ABORTED == GetState()) {
                LOG_INFO<<"procname:" << m_process_name <<",pid:"<<pid <<",source_state:"<< proc_state[(uint32_t)GetState()] <<",target_state:"<< proc_state[(uint32_t)target_state];
                ret = 0;  break;
            } else if (count <= 0) {
                LOG_INFO<<"procname:" << m_process_name <<",pid:"<<pid <<",source_state:"<< proc_state[(uint32_t)GetState()] <<",target_state:"<< proc_state[(uint32_t)target_state] <<" is timeout";
                ret = -1;
                break;
            } else if ((ProcessState::TERMINATING == target_state) && (pid < 0)) {
                ret = 0; break;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(PROC_STATE_DETECT_PERIOD));
            }
        }
    } break;

    case ProcessState::TERMINATED: {
        for ( ; count >= 0; count--) {
            if ((target_state == GetState() || ProcessState::ABORTED == GetState()) && (GetProcPid() < 0)) {
                LOG_INFO<<"procname:" << m_process_name <<",pid:"<<pid <<",source_state:"<< proc_state[(uint32_t)GetState()] <<",target_state:"<< proc_state[(uint32_t)target_state];
                ret = 0;  break;
            } else if (count <= 0) {
                LOG_INFO<<"procname:" << m_process_name <<",pid:"<<pid <<",source_state:"<< proc_state[(uint32_t)GetState()] <<",target_state:"<< proc_state[(uint32_t)target_state] <<" is timeout ...";
                ret = -1;
                break;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(PROC_STATE_DETECT_PERIOD));
            }
        }
    } break;

    default:
        break;
    }
    return ret;
}

ProcessState Process::GetState() const {
    std::shared_lock<std::shared_timed_mutex> rLock(m_smutex_exec_state);
    return m_proc_state;
}

void Process::SetState(ProcessState state) {
    std::lock_guard<std::shared_timed_mutex> lock(m_smutex_exec_state);
    if (state != m_proc_state) {
        m_proc_state = state;
    }
}

pid_t Process::GetProcPid() const {
    std::shared_lock<std::shared_timed_mutex> rLock(m_smutex_pid);
    return m_pid;
}

void Process::SetProcPid(pid_t pid) {
    std::lock_guard<std::shared_timed_mutex> lock(m_smutex_pid);
    if (pid != m_pid) {
        m_pid = pid;
    }
}

void Process::SetExecState(ExecutionState state) {
    std::lock_guard<std::shared_timed_mutex> lock(m_smutex_exec_state);
    if (state != m_exec_state) {
        m_exec_state = state;
        switch (state) {
        case ExecutionState::kRunning:
            m_proc_state = ProcessState::RUNNING;
            break;
        case ExecutionState::kTerminating:
            m_proc_state = ProcessState::TERMINATING;
            break;
        default:
            break;
        }
    }
}

int32_t Process::GetOrderOfMode(const std::string& mode, uint32_t* order) {
    int32_t ret = -1;
    if (m_order_mode_vec.size() > 0) {
        std::vector<ModeOrder>::iterator itr = m_order_mode_vec.begin();
        for ( ; itr != m_order_mode_vec.end(); ++itr) {
            ModeOrder morder = *itr;
            if (mode == morder.mode) {
                *order = morder.order;
                ret = 0;
                break;
            }
        }
    }
    return ret;
}

int32_t Process::ParseModeOrder(std::vector<std::string>* vec) {
    if (vec->size() > 0) {
        std::vector<std::string>::iterator itr = vec->begin();
        for ( ; itr != vec->end(); ++itr) {
            std::string str = *itr;
            str.erase(0, str.find_first_not_of(" "));
            str.erase(str.find_last_not_of(" ") + 1);

            size_t pos = str.find(".");
            if(pos != str.npos){
                std::string s_mod = str.substr(0, pos);
                std::string s_idx = str.substr(pos + 1, str.size());
                ModeOrder mododer;
                mododer.mode = s_mod;
                mododer.order = (uint32_t)std::stoi(s_idx);
                m_order_mode_vec.push_back(mododer);
            }
        }
        return 0;
    } else {
        LOG_ERROR<< "mode order list is empty.";
        return -1;
    }
}

int32_t Process::SetCpuUtility() {
    int32_t ret = 0;
    sched_param param;
    if (m_sched.policy == SCHED_OTHER) {
        param.sched_priority = 0;
    } else if ((m_sched.policy == SCHED_FIFO)|| (m_sched.policy == SCHED_RR)) {
        param.sched_priority = m_sched.priority;
    }
    if (sched_setscheduler(0, m_sched.policy, &param)) { ret += 1; }

    if (m_sched.iscpuset) {
        if (sched_setaffinity(0, sizeof(m_sched.affinity), &m_sched.affinity)) { ret += 2; }
    }

    return ret;
}

std::string Process::ConfigLoader(const std::string& fpath) {
    std::string output ="";
    if (fpath.empty()) {
        LOG_ERROR<<"empty file path";
        return output;
    }
    std::ifstream in(fpath.c_str(), std::ios::in | std::ios::binary);
    if (in.is_open()) {
        while (!in.eof()) {
            std::string line;
            getline(in, line, '\n');
            output += line;
        }
        in.close();
        return output;
    } else {
        LOG_ERROR<< "failed to open:"<<fpath;
        return output;
    }
}

}}}
