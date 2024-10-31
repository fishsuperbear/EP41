/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: em
 * Created on: Nov 23, 2022
 * Author: shefei
 * 
 */

#include "em/include/execmanagement.h"
#include "em/include/logger.h"
#include "em/utils/cJSON.h"
#include <algorithm>
#include <fstream>
#include <stdio.h>

namespace hozon {
namespace netaos {
namespace em {

std::shared_ptr<ExecManagement> ExecManagement::sm_emager(new ExecManagement());

ExecManagement::ExecManagement() {
    m_procont = new ProcessContainer();
    m_mode_state = ModeState::DEFAULT;
    m_dev_mode_on = false;
    m_sysmode = DEFAULT_MODE;
}

ExecManagement::~ExecManagement() {
    if (m_procont) {
        delete m_procont; m_procont = nullptr;
    }
}

int32_t ExecManagement::Init() {
    int32_t ret = 0;
    std::string proc_fpath = PROCESS_DIR_PATH;
    std::string conf_fpath = MACHINE_MANIFEST_FILE;

#ifdef EM_DEBUG_ON
    char* tmp_proc_dir = getenv(DEV_ENVRION_PROC_DIR);
    char* tmp_conf_dir = getenv(DEV_ENVRION_CONF_DIR);
    if (tmp_proc_dir && tmp_conf_dir) {
        m_dev_mode_on = true;
        m_dev_app_dir = tmp_proc_dir;
        m_dev_conf_dir = tmp_conf_dir;
        proc_fpath = m_dev_app_dir;
        conf_fpath = m_dev_conf_dir +"/"+ DEV_MACHINE_MANIFEST_FILE;
    }
#endif

    if (!m_procont) { ret = -1; return ret; }

    this->GetDefaultMode(&m_curmode);

    ret = m_procont->InitStartConfig(conf_fpath);
    if (ret != 0) { return ret; }

    if (m_curmode == "") {
        this->SetCurrMode(m_procont->m_defmode);
    }
    m_procont->Scan(proc_fpath);
    return ret;
}

void ExecManagement::DeInit() {
    LOG_INFO<<"deinit";
    if (m_dev_mode_on) {
        StopMode();
    }
}

std::shared_ptr<ExecManagement> ExecManagement::Instance() {
    return sm_emager;
}

int32_t ExecManagement::GetDefaultMode(std::string * mode) {
    std::string fpath = STARTUP_MANIFEST_FILE;
    if (m_dev_mode_on) {
        fpath = m_dev_conf_dir +"/"+ DEV_STARTUP_MANIFEST_FILE;
    }
    std::string str_value = m_procont->ConfigLoader(fpath);
    cJSON *key = nullptr;
    cJSON *root = cJSON_Parse(str_value.c_str());
    if (!root) { goto ERROR; }
    if (!cJSON_IsObject(root)) { goto ERROR; }

    key = cJSON_GetObjectItem(root, "startup_mode");
    if (!key) {
        goto ERROR;
    } else {
        *mode = key->valuestring;
        LOG_INFO<<"cur startup mode:"<<*mode;
    }

    if (root) { cJSON_Delete(root); }
    return 0;

ERROR:
    LOG_ERROR<<" key or object not found";
    if (root) { cJSON_Delete(root); }
    return -1;
}


int32_t ExecManagement::SetDefaultMode(const std::string& mode) {
    int32_t ret = 0;
    LOG_INFO<<"mode:"<<mode;
    cJSON * root = cJSON_CreateObject();
    cJSON_AddItemToObject(root,"startup_mode", cJSON_CreateString(mode.c_str()));
    char * buf = cJSON_Print(root);

    std::string fpath = STARTUP_MANIFEST_FILE;
    if (m_dev_mode_on) {
        fpath = m_dev_conf_dir +"/"+ DEV_STARTUP_MANIFEST_FILE;
    }

    uint32_t max_times = 3;
    std::ofstream fout;
    fout.open(fpath.c_str(), ios_base::out | ios_base::binary);
    while (!fout.is_open() && --max_times >0) {
        LOG_WARN<<"open file failed, retry "<<(3 - max_times);
	}
    if (!fout.is_open()) {
        ret = -1;
        LOG_ERROR<<"open file failed.";
    } else {
        fout << buf << flush;
        fout.close();
    }
    if (root) { cJSON_Delete(root); }
    return ret;
}

std::string ExecManagement::GetCurrMode() {
    std::shared_lock<std::shared_timed_mutex> rLock(m_smutex_mode);
    return m_curmode;
}

void ExecManagement::SetCurrMode(const std::string& mode) {
    std::lock_guard<std::shared_timed_mutex> lock(m_smutex_mode);
    if (m_curmode != mode) {
        m_curmode = mode;
    }
}

void ExecManagement::SetSysMode(const std::string& state) {
    std::lock_guard<shared_timed_mutex> lock(m_smutex_syste);
    if(m_sysmode != state){
        m_sysmode = state;
    }
}

std::string ExecManagement::GetSysMode() {
    std::shared_lock<std::shared_timed_mutex> rLock(m_smutex_syste);
    return m_sysmode;
}

int32_t ExecManagement::StartMode(const std::string& mode) {
    LOG_INFO<<"mode:"<< mode;
    SetModeState(ModeState::STARTING);
    int32_t ret = -1;
    if (m_procont) {
        ret = m_procont->StartProcGroup(mode);
        SetCurrMode(mode);
        SetSysMode(mode);
        // SetCurrMode(ret == 0 ? mode : ABNORMAL_MODE);
        LOG_INFO<<"sys curmode:"<<m_procont->m_curmode;
    } else {
        LOG_CRITICAL<<"nullptr";
    }
    SetModeState(ModeState::FINISHED);
    return ret;
}

int32_t ExecManagement::StopMode() {
    int32_t ret = -1;
    if (ModeState::FINISHED != GetModeState()) {
        ret = (int32_t)ResultCode::kRejected;
        return ret;
    }

    SetModeState(ModeState::STARTING);
    LOG_INFO<<"cur mode:"<<GetCurrMode();
    if (m_procont) {
        ret =  m_procont->StopProcGroup(GetCurrMode());
        SetCurrMode(OFF_MODE);
        SetSysMode(OFF_MODE);
        // SetCurrMode(ret == 0 ? OFF_MODE : ABNORMAL_MODE);
    } else {
        LOG_CRITICAL<<"nullptr";
    }
    SetModeState(ModeState::FINISHED);
    return ret;
}

int32_t ExecManagement::SwitchMode(const std::string& mode) {
    LOG_INFO<<"mode:"<< mode;
    int32_t ret = -1;
    if (ModeState::FINISHED != GetModeState()) {
        ret = (int32_t)ResultCode::kRejected;
        return ret;
    }
    std::vector<std::string> vect;
    if (0 == GetModeList(&vect)) {
        LOG_INFO<<"mode list size:"<<vect.size();
        std::vector<std::string>::iterator itr = std::find(vect.begin(),vect.end(),mode);
        if (itr == vect.end()) {
            ret = (int32_t)ResultCode::kInvalid;
            return ret;
        }
    }

    SetModeState(ModeState::STARTING);
    if (m_procont) {
        ret = m_procont->SwitchProcGroup(mode);
        SetCurrMode(mode);
        SetSysMode(mode);
        // SetCurrMode(ret == 0 ? mode : ABNORMAL_MODE);
    } else {
        LOG_CRITICAL<<"nullptr";
    }
    SetModeState(ModeState::FINISHED);
    return ret;
}

int32_t ExecManagement::GetModeList(std::vector<std::string> *vect) {
    if (!vect || !m_procont) {
        LOG_ERROR<<"mode list is empty";
        return -1;
    } else {
        vect->assign(m_procont->m_modes_vec.begin(), m_procont->m_modes_vec.end());
        return 0;
    }
}

int32_t ExecManagement::GetModeListDetailInfo(std::unordered_map<std::string, std::vector<std::shared_ptr<Process>>> &mode_name_process_list_map) {
    std::vector<std::string> mode_list;
    if (GetModeList(&mode_list) != 0) {
        return -1;
    }

    for (const auto &mode : mode_list) {
        m_procont->GetProcessGroupOfMode(mode, mode_name_process_list_map);
    }
    return 0;
}

std::shared_ptr<Process> ExecManagement::GetProcess(const std::string& procname) {
    if (m_procont) {
        return m_procont->GetProcess(procname);
    } else {
        LOG_CRITICAL<<"nullptr";
        return nullptr;
    }
}

int32_t ExecManagement::ProcRestart(const std::string& procname) {
    int32_t ret = 0;
    if (ModeState::FINISHED != GetModeState()) {
        ret = (int32_t)ResultCode::kRejected;
        return ret;
    }

    SetModeState(ModeState::STARTING);
    if (m_procont) {
        std::shared_ptr<Process> proc = GetProcess(procname);
        if (proc) {
            LOG_INFO<<"procname:"<<procname;
            proc->Restart();
        } else {
            if (procname == DESAY_UPDATE_SERVICE_NAME) {
                ShellExec(DESAY_UPDATE_SERVICE_SHELL);
            } else {
                LOG_ERROR<<"not found proc:"<<procname;
                ret = (int32_t)ResultCode::kInvalid;
            }
        }
    } else {
        LOG_CRITICAL<<"nullptr";
        ret = (int32_t)ResultCode::kFailed;
    }
    SetModeState(ModeState::FINISHED);

    return ret;
}

int32_t ExecManagement::GetModeOfProcess(std::vector<ProcessInfo> * vect){
    if (!vect || !m_procont) {
        return -1;
    } else if (m_procont->m_process_group.size() > 0) {
        for(auto proc : m_procont->m_process_group){
            ProcessInfo info;
            info.group = 0;
            info.procname = proc->m_process_name;
            info.procstate = proc->m_proc_state;

            uint32_t order = 0;
            proc->GetOrderOfMode(GetCurrMode(),&order);
            std::map<uint32_t,std::vector<std::shared_ptr<Process>>>::iterator mapItr = m_procont->m_cur_mode_proc_map.find(order);
            if (mapItr != m_procont->m_cur_mode_proc_map.end()) {
                std::vector<std::shared_ptr<Process>> vec= mapItr->second;
                for(auto itm : vec){
                    std::shared_ptr<Process> pro = itm;
                    if(info.procname == pro->m_process_name){
                        info.procstate = pro->m_proc_state;
                        info.group = order;
                        break;
                    }
                }
            }
            vect->push_back(info);
        }
    }
    return 0;
}

void ExecManagement::SetModeState(ModeState state){
    std::lock_guard<shared_timed_mutex> lock(m_smutex_state);
    if(m_mode_state != state){
        m_mode_state = state;
    }
}

ModeState ExecManagement::GetModeState(){
    std::shared_lock<std::shared_timed_mutex> rLock(m_smutex_state);
    return m_mode_state;
}

void ExecManagement::SetDevConfig(std::string* fpath){
    std::string res = *fpath;
    if(m_dev_app_dir.empty()){
        char buf[128]={0};
        FILE * fp = popen("pwd", "r");
        if(fp){
            char* ret = fgets(buf,sizeof(buf),fp);
            if(ret && buf[0] != '\0' && buf[0] != '\n'){
                res = buf;
                res.erase(res.find_last_of("/bin")-3,std::string::npos);
                res += *fpath;
                *fpath = res;
            }
            pclose(fp); fp = nullptr;
        }
    }else{
        *fpath = m_dev_app_dir + res;
    }
}

void ExecManagement::ShellExec(const std::string& cmd) {
    std::string res="";
    char buf[128]={0};
    FILE * fp = popen(cmd.c_str(), "r");
    if(fp){
        char* ret = fgets(buf,sizeof(buf),fp);
        if(ret && buf[0] != '\0' && buf[0] != '\n'){
            res = buf;
        }
        pclose(fp); fp = nullptr;
    }
    LOG_INFO << cmd << "execution finish";
}

}}}
