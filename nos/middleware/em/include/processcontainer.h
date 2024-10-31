/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: em
 * Created on: Nov 23, 2022
 * Author: shefei
 * 
 */

#ifndef PROCESS_CONTAINER_H
#define PROCESS_CONTAINER_H

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include "em/include/define.h"
#include "em/include/process.h"

namespace hozon {
namespace netaos {
namespace em {

using namespace std;
using namespace hozon::netaos::em;

typedef std::map<uint32_t,std::vector<std::shared_ptr<Process>>> ProcMapType;

class ProcessContainer {
public:
    ProcessContainer();
    virtual ~ProcessContainer();
    void Scan(const std::string& path);
    std::shared_ptr<Process> GetProcess(const std::string& process_name);
    int32_t GetProcessGroupOfMode(const std::string& mode_name, ProcMapType* mode_proc_map);
    int32_t GetProcessGroupOfMode(const std::string& mode_name,
            std::unordered_map<std::string, std::vector<std::shared_ptr<Process>>> &mode_name_process_list_map);
 
    int32_t InitStartConfig(const std::string& path);
    std::string ConfigLoader(const std::string& fpath);
    int32_t StartProcGroup(const std::string& mode);
    int32_t StopProcGroup(const std::string& mode);
    int32_t SwitchProcGroup(const std::string& mode);

private:
    void TraversalFolder(const std::string& path,std::vector<std::string>* files);
    /* monitor proc group status */
    int32_t ProcGroupStateMonitor(std::vector<std::shared_ptr<Process>>* ptr, ProcessState state);
    void FormatModeOrderMap(uint32_t order, std::shared_ptr<Process> ptr, ProcMapType* mode_proc_map);
    void CompareMap(ProcMapType *src_map, ProcMapType *tar_map, ProcMapType *diff_map, MapCompareType cmp_type);

    int32_t StartProcGroup(ProcMapType* map);
    int32_t StopProcGroup(ProcMapType* map);

    std::vector<uint32_t> GetModeOrderList(ProcMapType * map, Sortord order = Sortord::ASCEND);

public:
    std::string m_curmode;
    std::string m_defmode;
    uint32_t m_pro_enter_timeout;
    uint32_t m_pro_exit_timeout;
    uint32_t m_restart_attempt_switch;
    std::vector<std::string> m_def_environments;
    /* all proc info */
    std::vector<std::shared_ptr<Process>> m_process_group;
    /* all modes info */
    std::vector<std::string> m_modes_vec;
    /* cur mode proc info */
    ProcMapType m_cur_mode_proc_map;
};

}}}
#endif
