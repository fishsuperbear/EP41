/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: em
 * Created on: Nov 23, 2022
 * Author: shefei
 * 
 */

#ifndef PROCESS_MANAGEMENT_H
#define PROCESS_MANAGEMENT_H

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include "em/include/define.h"
#include "em/include/processcontainer.h"

namespace hozon {
namespace netaos {
namespace em {

using namespace std;
using namespace hozon::netaos::em;

class ExecManagement {
public:

    static std::shared_ptr<ExecManagement> Instance();
    virtual ~ExecManagement();
    
    int32_t Init();
    void DeInit();

public:
/** em external functional interfaces **/
    /* get em default startup mode */
    int32_t GetDefaultMode(std::string * mode);
    /* set em default startup mode */
    int32_t SetDefaultMode(const std::string& mode);
    /* get curr startup mode */
    std::string GetCurrMode();
    /* get curr sys running mode */
    std::string GetSysMode();
    /* startup specified mode - only use from OFF state */
    int32_t StartMode(const std::string& mode);
    /* stop current mode all processes */
    int32_t StopMode();
    /* from current mode switch specified mode */
    int32_t SwitchMode(const std::string& mode);
    /* restart specified process by name */
    int32_t ProcRestart(const std::string& procname);
    /* get all defined mode */
    int32_t GetModeList(std::vector<std::string> * vect);
    /* get specified process state in current mode  */
    std::shared_ptr<Process> GetProcess(const std::string& procname);
    /* get all processes state in current mode  */
    int32_t GetModeOfProcess(std::vector<ProcessInfo> * vect);
    /* get the detail info of all modes */
    int32_t GetModeListDetailInfo(std::unordered_map<std::string, std::vector<std::shared_ptr<Process>>> &mode_name_process_list_map);

private:
    ExecManagement();
    ExecManagement(const ExecManagement &);  
    ExecManagement & operator = (const ExecManagement &);

    void SetCurrMode(const std::string& mode);
    /* mode running state */
    void SetModeState(ModeState state);
    ModeState GetModeState();

    /* cur system mode */
    void SetSysMode(const std::string& state);

    void ShellExec(const std::string& cmd);
    /* dev & test func */
    void SetDevConfig(std::string* fpath);

private:
    static std::shared_ptr<ExecManagement> sm_emager;
    ProcessContainer *m_procont;
    ModeState m_mode_state;
    std::string m_curmode;  /* sys startup mode */
    std::string m_sysmode;  /* cur system mode */

    std::shared_timed_mutex m_smutex_mode;
    std::shared_timed_mutex m_smutex_state;
    std::shared_timed_mutex m_smutex_syste;
    std::string m_dev_app_dir;
    std::string m_dev_conf_dir;
    bool m_dev_mode_on;
};

}}}
#endif
