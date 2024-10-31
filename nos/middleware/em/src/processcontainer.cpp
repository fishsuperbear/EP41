/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: em
 * Created on: Nov 23, 2022
 * Author: shefei
 * 
 */

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fstream>
#include <dirent.h>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <thread>
#include "em/utils/cJSON.h"
#include "em/include/define.h"
#include "em/include/logger.h"
#include "em/include/processcontainer.h"

namespace hozon {
namespace netaos {
namespace em {

using namespace std;

ProcessContainer::ProcessContainer() {
    m_curmode = "";
    m_defmode = "";
    m_pro_enter_timeout = 0;
    m_pro_exit_timeout = 0;
    m_restart_attempt_switch = 0;
}

ProcessContainer::~ProcessContainer() {
}

void ProcessContainer::Scan(const std::string& path) {

    std::vector<std::string> filevec;
    /* scan config folder */
    this->TraversalFolder(path,&filevec);
    LOG_DEBUG<<" vect file size: "<< filevec.size();
    if (filevec.size() == 0) { return; }

    std::vector<std::string>::iterator it = filevec.begin();
    for( ; it != filevec.end(); ++it) {
        /* parse manifest */
        std::shared_ptr<Process> proc = std::make_shared<Process>();
        int32_t ret = proc->ParseManifest(*it);
        if (ret) {
            LOG_ERROR<<"parse manifest fail."<< *it;
        }
        /* set proc timeout */
        proc->m_enter_timeout = m_pro_enter_timeout;
        proc->m_exit_timeout = m_pro_exit_timeout;
        /* restart attempt switch */
        if (m_restart_attempt_switch == 0) {
            proc->m_restart_attempt_num = 0;
        }

        if (m_def_environments.size() > 0) {
            proc->m_exp_environments.insert(proc->m_exp_environments.end(), m_def_environments.begin(), m_def_environments.end());
        }

        /* filter proc mode define */
        for (auto & elem : proc->m_order_mode_vec) {
            m_modes_vec.push_back(elem.mode);
        }
        sort(m_modes_vec.begin(),m_modes_vec.end());
        m_modes_vec.erase(unique(m_modes_vec.begin(), m_modes_vec.end()), m_modes_vec.end());

        /* all proc info */
        m_process_group.push_back(proc);
    }
}

std::shared_ptr<Process> ProcessContainer::GetProcess(const std::string& process_name) {
    if (m_process_group.size()>0) {
        std::vector<std::shared_ptr<Process>>::iterator itr = m_process_group.begin();
        for ( ; itr != m_process_group.end(); ++itr) {
            std::shared_ptr<Process> ptr = *itr;
            if (ptr->m_process_name == process_name) {
                return (*itr);
            }
        }
    } else {
        LOG_ERROR<<process_name<< " not found.";
    }
    return nullptr;
}

int32_t ProcessContainer::GetProcessGroupOfMode(const std::string& mode_name,
            std::unordered_map<std::string, std::vector<std::shared_ptr<Process>>> &mode_name_process_list_map) {
    LOG_DEBUG<<"mode:"<<mode_name;
    if (m_process_group.size() > 0) {
        std::vector<std::shared_ptr<Process>>::iterator itr = m_process_group.begin();
        for ( ; itr != m_process_group.end(); ++itr) {
            std::shared_ptr<Process> ptr = *itr;
            uint32_t order = 0;
            ptr->GetOrderOfMode(mode_name,&order); /* get cur mode proc order */
            if (order > 0 && order <= 99) {
                mode_name_process_list_map[mode_name].emplace_back(ptr);
            } else if (order != 0) {
                LOG_WARN<<"order out of range. mode_name:" << mode_name << ", order:" << order;
            }
        }
        return 0;
    } else {
        LOG_ERROR<< "process group not found.";
        return -1;
    }
}


int32_t ProcessContainer::GetProcessGroupOfMode(const std::string& mode_name, ProcMapType* mode_proc_map) {
    LOG_DEBUG<<"mode:"<<mode_name;
    if (m_process_group.size() > 0) {
        std::vector<std::shared_ptr<Process>>::iterator itr = m_process_group.begin();
        for ( ; itr != m_process_group.end(); ++itr) {
            std::shared_ptr<Process> ptr = *itr;
            uint32_t order = 0;
            ptr->GetOrderOfMode(mode_name,&order); /* get cur mode proc order */
            if (order > 0 && order <= 99) {
                FormatModeOrderMap(order,ptr,mode_proc_map);
            } else if (order != 0) {
                LOG_WARN<<"order out of range.";
            }
        }
        return 0;
    } else {
        LOG_ERROR<< "process group not found.";
        return -1;
    }
}

void ProcessContainer::FormatModeOrderMap(uint32_t order, std::shared_ptr<Process> ptr, ProcMapType* mode_proc_map) {
    std::vector<std::shared_ptr<Process>> vect;
    if (ptr) {
        vect.push_back(ptr);
        std::map<uint32_t,std::vector<std::shared_ptr<Process>>>::iterator mapItr = mode_proc_map->find(order);
        if (mapItr != mode_proc_map->end()) {
            std::vector<std::shared_ptr<Process>> tmp_vec = mapItr->second;
            tmp_vec.insert(tmp_vec.end(), vect.begin(), vect.end());
            (*mode_proc_map)[order] = tmp_vec;
        } else {
            mode_proc_map->insert({order,vect});
        }
    } else {
        LOG_ERROR<<"null pointer";
    }
}

std::vector<uint32_t> ProcessContainer::GetModeOrderList(ProcMapType * map, Sortord order) {
    std::vector<uint32_t> key_vec;
    std::transform(map->begin(),map->end(),std::inserter(key_vec, key_vec.end()), \
        [](std::pair<uint32_t,std::vector<std::shared_ptr<Process>>> pair) { return pair.first; });

    std::sort(key_vec.begin(),key_vec.end(),[=](uint32_t a,uint32_t b){ \
        return ((order == Sortord::ASCEND) ? (a < b) : (a > b));});

    return key_vec;
}

int32_t ProcessContainer::StartProcGroup(const std::string& mode) {
    std::map<uint32_t,std::vector<std::shared_ptr<Process>>> tar_proc_group_map;
    if (0 != GetProcessGroupOfMode(mode,&tar_proc_group_map)) {
        LOG_ERROR<<"GetProcessGroupOfMode fail.";
        return -1;
    }

    m_curmode = mode;
    std::vector<int32_t> vec_res;
    std::vector<uint32_t> key_vec = GetModeOrderList(&tar_proc_group_map);
    if (key_vec.size() > 0) {
        std::vector<uint32_t>::iterator it = key_vec.begin();
        for ( ; it != key_vec.end(); it++) {
            std::map<uint32_t,std::vector<std::shared_ptr<Process>>>::iterator mapItr = tar_proc_group_map.find(*it);
            if (mapItr != tar_proc_group_map.end()) {
                std::vector<std::shared_ptr<Process>> veci= mapItr->second;
                std::cout<<std::endl;
                LOG_INFO<<"start proc group:"<<*it<<",size:"<<veci.size();
                /* create proc */
                for (uint32_t j=0; j<veci.size(); j++) {
                    std::shared_ptr<Process> ptr = veci[j];
                    if (ptr) {
                        ptr->Start();
                        LOG_DEBUG<<">>> start proc:"<<ptr->m_process_name <<",groupid:"<<*it;
                    }
                }
                /* save cur mode proc map */
                m_cur_mode_proc_map.insert({*it,veci});

                int32_t ret = ProcGroupStateMonitor(&veci, ProcessState::RUNNING);
                vec_res.push_back(ret);
                LOG_INFO<<"start proc group:"<<*it<<" finished,timeout:"<<ret;BR;
            } else {
                vec_res.push_back(0);
                LOG_INFO<<"group id:"<<*it<<" is empty.";
                continue;
            }
        }
    } else {
        LOG_WARN<<"ignore empty group map.";
    }

    if ((vec_res.size() > 0) && (std::count(vec_res.begin(),vec_res.end(),0) == (int)key_vec.size())) {
        return 0;
    } else {
        return -1;
    }
}

int32_t ProcessContainer::StopProcGroup(const std::string& mode) {
    //LOG_INFO<<"mode:"<<mode;
    if (mode != m_curmode) {
        LOG_WARN<< "cur mode is:"<<m_curmode;
    }

    m_curmode = OFF_MODE;
    std::vector<int32_t> vec_res;
    std::vector<uint32_t> key_vec = GetModeOrderList(&m_cur_mode_proc_map, Sortord::DESCEND);
    if (key_vec.size() > 0) {
        std::vector<uint32_t>::iterator it = key_vec.begin();
        for ( ; it != key_vec.end(); it++) {
            std::map<uint32_t,std::vector<std::shared_ptr<Process>>>::iterator mapItr = m_cur_mode_proc_map.find(*it);
            if (mapItr != m_cur_mode_proc_map.end()) {
                std::vector<std::shared_ptr<Process>> veci= mapItr->second;
                LOG_INFO<<"stop proc group:"<<*it<<",size:"<<veci.size();
                for (uint32_t j=0; j<veci.size(); j++) {
                    std::shared_ptr<Process> ptr = veci[j];
                    if (ptr) {
                        ptr->Terminate();
                        LOG_DEBUG<<">>> stop proc:"<<ptr->m_process_name << ",groupid:"<<*it;
                    }
                }
                int32_t ret = ProcGroupStateMonitor(&veci, ProcessState::TERMINATED);
                vec_res.push_back(ret);
                LOG_INFO<<"stop proc group:"<<*it<<" finished,timeout:"<<ret;BR;
            } else {
                LOG_INFO<<"map key:"<<*it<< " not found.";
                continue;
            }
        }
    } else {
        LOG_WARN<<"ignore empty group map.";
    }

    m_cur_mode_proc_map.clear();
    //if(std::count(vec_res.begin(),vec_res.end(),0) != (int)key_vec.size()){ LOG_WARN<< "proc(s) exit timeout."; }
    return 0;
}

int32_t ProcessContainer::StartProcGroup(ProcMapType* map) {
    if (!map || map->size() == 0) {
        LOG_WARN<< "ignore empty group map.";
        return 0;
    } else {
        std::vector<int32_t> vec_res;
        std::vector<uint32_t> key_vec = GetModeOrderList(map);
        std::vector<uint32_t>::iterator it = key_vec.begin();
        for ( ; it != key_vec.end(); it++) {
            std::map<uint32_t,std::vector<std::shared_ptr<Process>>>::iterator mapItr = map->find(*it);
            if (mapItr != map->end()) {
                std::vector<std::shared_ptr<Process>> veci= mapItr->second;
                LOG_INFO<<"start proc group:"<<*it<<" size:"<<veci.size();
                /* create proc */
                for (uint32_t j=0; j<veci.size(); j++) {
                    std::shared_ptr<Process> ptr = veci[j];
                    if (ptr) {
                        ptr->Start();
                        LOG_DEBUG<<">>> start proc:"<<ptr->m_process_name << " groupid:"<<*it;
                    }
                }
                int32_t ret = ProcGroupStateMonitor(&veci, ProcessState::RUNNING);
                vec_res.push_back(ret);
                LOG_INFO<<"start proc group:"<<*it <<" finished,timeout:"<<ret;BR;
            } else {
                vec_res.push_back(0);
                LOG_INFO<<"group id:"<<*it<<" is empty.";
                continue;
            }
        }
        if ((vec_res.size() > 0) && (std::count(vec_res.begin(),vec_res.end(),0) == (int)key_vec.size())) {
            return 0;
        } else {
            return -1;
        }
    }
}

int32_t ProcessContainer::StopProcGroup(ProcMapType* map) {
    if (!map || map->size() == 0) {
        LOG_INFO<< "ignore empty group map.";
        return 0;
    } else {
        std::vector<int32_t> vec_res;
        std::vector<uint32_t> key_vec = GetModeOrderList(map, Sortord::DESCEND);
        std::vector<uint32_t>::iterator it = key_vec.begin();
        for ( ; it != key_vec.end(); it++) {
            std::map<uint32_t,std::vector<std::shared_ptr<Process>>>::iterator mapItr = map->find(*it);
            if (mapItr != map->end()) {
                std::vector<std::shared_ptr<Process>> veci= mapItr->second;
                LOG_INFO<<"stop proc group:"<<*it<<",size:"<<veci.size();
                for (uint32_t j = 0; j < veci.size(); j++) {
                    std::shared_ptr<Process> ptr = veci[j];
                    if (ptr) {
                        ptr->Terminate();
                        LOG_DEBUG<<">>> stop proc:"<<ptr->m_process_name<<",groupid:"<<*it;
                    }
                }
                int32_t ret = ProcGroupStateMonitor(&veci, ProcessState::TERMINATED);
                vec_res.push_back(ret);
                LOG_INFO<<"stop proc group:"<<*it<<" finished,timeout:"<<ret;BR;
            } else {
                LOG_INFO<<"map key:"<<*it<<" not found.";
                continue;
            }
        }
        // if(std::count(vec_res.begin(),vec_res.end(),0) != (int)key_vec.size()){  LOG_WARN<< "proc(s) exit timeout."; }
        return 0;
    }
}

int32_t ProcessContainer::SwitchProcGroup(const std::string& mode) {
    int32_t ret = 0;
    if (0 == m_cur_mode_proc_map.size()) {
        LOG_INFO<<"curr mode proc list is empty.";
        ret = this->StartProcGroup(mode);
    } else {
        m_curmode = mode;

        std::map<uint32_t,std::vector<std::shared_ptr<Process>>> tar_proc_group_map;
        if (0 != GetProcessGroupOfMode(mode,&tar_proc_group_map)) {
            LOG_ERROR<<"get process group of mode fail.";
            ret = -1;
        } else {
            /* get stop and start proc list */
            std::map<uint32_t,std::vector<std::shared_ptr<Process>>> exit_pro_map;
            std::map<uint32_t,std::vector<std::shared_ptr<Process>>> start_pro_map;
            CompareMap(&m_cur_mode_proc_map, &tar_proc_group_map, &exit_pro_map, MapCompareType::DIFF_DEL);
            CompareMap(&m_cur_mode_proc_map, &tar_proc_group_map, &start_pro_map, MapCompareType::DIFF_ADD);
            CompareMap(&m_cur_mode_proc_map, &tar_proc_group_map, nullptr, MapCompareType::DIFF_CPY);

            ret = this->StopProcGroup(&exit_pro_map);
            LOG_INFO<<"stop proc finished, ret:"<<ret;BR;
            std::this_thread::sleep_for(std::chrono::milliseconds(GROUP_STATE_DETECT_PERIOD));
            ret = this->StartProcGroup(&start_pro_map);
            LOG_INFO<<"start proc finished, ret:"<<ret;BR;
        }
    }
    LOG_INFO<<"switch proc group finished, ret:"<<ret;
    return ret;
}

void ProcessContainer::CompareMap(ProcMapType *src_map, ProcMapType *tar_map, ProcMapType *diff_map, MapCompareType cmp_type) {
    switch (cmp_type) {
    case MapCompareType::DIFF_DEL:{
        if (!src_map || !tar_map || !diff_map) {
            LOG_ERROR<<"null map pointer.";
            return;
        }
        std::vector<uint32_t> key_vec = GetModeOrderList(src_map);
        std::vector<uint32_t>::iterator it = key_vec.begin();
        for ( ; it != key_vec.end(); it++) {

            std::map<uint32_t,std::vector<std::shared_ptr<Process>>>::iterator tarItr = tar_map->find(*it);
            if (tarItr == tar_map->end()) { /* cur map key not in target map */
                diff_map->insert({*it,(*src_map)[*it]});
            } else {
                std::vector<std::shared_ptr<Process>> del_vect;
                std::vector<std::shared_ptr<Process>> cur_vect = (*src_map)[*it];
                std::vector<std::shared_ptr<Process>> tar_vect = (*tar_map)[*it];
                for (auto ptr1 : cur_vect) {
                    bool isneed = false;
                    std::shared_ptr<Process> proc = ptr1;
                    for (auto ptr2 : tar_vect) {
                        // std::shared_ptr<Process> proc = ptr2;
                        if (ptr1->m_process_name == ptr2->m_process_name) {
                            isneed = true;
                            break;
                        }
                    }
                    if (!isneed) {
                        del_vect.push_back(proc);
                    }
                }
                if (del_vect.size() > 0) {
                    diff_map->insert({*it,del_vect});
                }
            }
        }
    }
        break;
    case MapCompareType::DIFF_ADD:{
        if (!src_map || !tar_map || !diff_map) {
            LOG_ERROR<<"null map pointer.";
            return;
        }
        std::vector<uint32_t> key_vec = GetModeOrderList(tar_map);
        std::vector<uint32_t>::iterator it = key_vec.begin();
        for( ; it != key_vec.end(); it++) {
            std::map<uint32_t,std::vector<std::shared_ptr<Process>>>::iterator tarItr = src_map->find(*it);
            if (tarItr == src_map->end()) { /* tar map key not in cur map */
                diff_map->insert({*it,(*tar_map)[*it]});
            } else {
                std::vector<std::shared_ptr<Process>> add_vect;
                std::vector<std::shared_ptr<Process>> cur_vect = (*src_map)[*it];
                std::vector<std::shared_ptr<Process>> tar_vect = (*tar_map)[*it];
                for (auto ptr1 : tar_vect) {
                    bool exist = false;
                    std::shared_ptr<Process> proc = ptr1;
                    for (auto ptr2 : cur_vect) {
                        /* exist && running */
                        if (ptr1->m_process_name == ptr2->m_process_name && ptr2->GetState() == ProcessState::RUNNING) {
                            exist = true;
                            break;
                        }
                    }
                    if (!exist) {
                        add_vect.push_back(proc);
                    }
                }
                if (add_vect.size() > 0) {
                    diff_map->insert({*it,add_vect});
                }
            }
        }
    }
        break;
    case MapCompareType::DIFF_CPY:{
        if (!src_map || !tar_map) {
            LOG_ERROR<<"null map pointer.";
            return;
        }
        if (src_map->size() > 0) {
            src_map->clear();
        }
        std::vector<uint32_t> key_vec = GetModeOrderList(tar_map);
        std::vector<uint32_t>::iterator it = key_vec.begin();
        for( ; it != key_vec.end(); it++) {
            std::vector<std::shared_ptr<Process>> vect = (*tar_map)[*it];
            src_map->insert({*it,vect});
        }
    }
        break;
    default:
        break;
    }
}

int32_t ProcessContainer::ProcGroupStateMonitor(std::vector<std::shared_ptr<Process>>* ptr, ProcessState state) {
    int32_t ret = 0;
    if (ptr) {
        int32_t count = ( m_pro_enter_timeout / GROUP_STATE_DETECT_PERIOD ) * PROC_RESTART_MAX_TIMES + 2;
        if (ProcessState::TERMINATED == state) {
            count = ( m_pro_exit_timeout / GROUP_STATE_DETECT_PERIOD ) * 2;
        }
        /* check process report status */
        while (count-- >= 0) {
            uint32_t flag = 0;
            ret = 0;
            for (uint32_t i = 0; i < ptr->size(); i++) {
                std::shared_ptr<Process> proc = ptr->at(i);
                //LOG_DEBUG<<"procname:"<<proc->m_process_name <<",getstate:" << (int)proc->GetState();
                switch (state) {
                case ProcessState::TERMINATED:
                case ProcessState::RUNNING:{
                    if (state == proc->GetState()) {
                        flag++;
                    } else if (ProcessState::ABORTED == proc->GetState()) {
                        flag++; ret++;
                        //LOG_INFO<<"flag:"<<flag<<",ret:"<<ret<<",count:"<<count;
                    }
                }
                    break;
                default:
                    break;
                }
            }

            if (flag >= ptr->size()) {
                LOG_DEBUG<<"< all proc have finished reporting state >";
                break;
            } else {
                //LOG_DEBUG<<"flag:"<<flag<<",count:"<<count;
                std::this_thread::sleep_for(std::chrono::milliseconds(GROUP_STATE_DETECT_PERIOD));
                if (count <= 0) {
                    LOG_WARN<<"< wait proc report state timeout >";
                    ret = -1;
                    break;
                }
            }
        }
    }
    return ret;
}

int32_t ProcessContainer::InitStartConfig(const std::string& path) {
    std::string str_value = this->ConfigLoader(path);
    cJSON *key = nullptr;
    cJSON *item = nullptr;
    cJSON *arritem = nullptr;
    cJSON *root = cJSON_Parse(str_value.c_str());
    if (!root) { goto ERROR; }
    if (!cJSON_IsObject(root)) { goto ERROR; }

    key = cJSON_GetObjectItem(root, "default_startup_mode");
    if (!key) {
        goto ERROR;
    } else {
        m_defmode = key->valuestring;
        LOG_INFO<<"def startup mode:"<<m_defmode;
    }

    key = cJSON_GetObjectItem(root, "default_app_enter_timeout");
    if (!key) {
        goto ERROR;
    } else {
        m_pro_enter_timeout = std::atoi(key->valuestring)*1000;
    }

    key = cJSON_GetObjectItem(root, "default_app_exit_timeout");
    if (!key) {
        goto ERROR;
    } else {
        m_pro_exit_timeout =std::atoi(key->valuestring)*1000;
    }

    key = cJSON_GetObjectItem(root, "restart_attempt_switch");
    if (!key) {
        goto ERROR;
    } else {
        m_restart_attempt_switch = std::atoi(key->valuestring);
    }

    item = cJSON_GetObjectItem(root, "default_environments");
    if (!item) {
        goto ERROR;
    } else {
        uint32_t arrsize = cJSON_GetArraySize(item);
        for (size_t i = 0; i < arrsize; i++) {
            arritem = cJSON_GetArrayItem(item, i);
            if (arritem) {
                m_def_environments.push_back(arritem->valuestring);
                LOG_INFO<<"def envs:"<<arritem->valuestring;
            }
        }
    }

    if (root) {
        cJSON_Delete(root);
    }
    return 0;

ERROR:
    LOG_ERROR<<"key or object not found.";
    if (root) {
        cJSON_Delete(root);
    }
    return -1;
}

void ProcessContainer::TraversalFolder(const std::string& path,std::vector<std::string>* files) {
    DIR *pDir;
    struct dirent *ent;
    char subpath[512];
    std::vector<std::string> folder;

    if (!(pDir = opendir(path.c_str()))) {
        LOG_ERROR<<"folder not exist > "<<path;
        return;
    }

    memset(subpath, 0, sizeof(subpath));
    while ((ent = readdir(pDir)) != NULL) {
        if (ent->d_type & DT_DIR) {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
                continue;
            } else {
                sprintf(subpath, "%s/%s", path.c_str(), ent->d_name);
                folder.push_back(subpath);
            }
        }
    }
    closedir(pDir);

    if (folder.size()>0) {
        for (auto item : folder) {
            memset(subpath, 0, sizeof(subpath));
            sprintf(subpath, "%s/%s/%s", item.c_str(), PROC_CONFIG_FOLDER_NAME, PROC_CONFIG_FILE_NAME);
            if (access(subpath, F_OK) == 0) {
                files->push_back(subpath);
            }
        }
    }
}

std::string ProcessContainer::ConfigLoader(const std::string& fpath) {
    std::string output ="";
    if (fpath.empty()) {
        LOG_ERROR<<"empty file path.";
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
        LOG_ERROR<<"failed to open:"<<fpath;
        return output;
    }
}

}}}
