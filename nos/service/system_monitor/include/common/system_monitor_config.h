/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: system monitor config
 */

#ifndef SYSTEM_MONITOR_CONFIG_H
#define SYSTEM_MONITOR_CONFIG_H

#include <mutex>
#include "system_monitor/include/common/system_monitor_def.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

class SystemMonitorConfig {
public:
    static SystemMonitorConfig* getInstance();

    void Init();
    void DeInit();

    void LoadSystemMonitorConfig();
    const SystemMonitorConfigInfo& GetSystemMonitorConfigInfo() {return system_monitor_config_info_;}
    const SystemMonitorTcpConfigInfo& GetSystemMonitorTcpConfigInfo() {return system_monitor_tcp_config_info_;}

    bool IsDiskMonitorPathList(const std::string& path);
    uint8_t GetPartitionAlarmValue(const std::string& path);
    std::unordered_map<std::string, SystemMonitorDiskMonitorPartitionInfo> GetDiskMonitorPathList() {return disk_monitor_path_list_;}
    std::vector<SystemMonitorDiskMonitorLogMoveListInfo> GetDiskMonitorLogMoveList(const SystemMonitorDiskMonitorLogMoveType type);
    std::vector<std::string> GetProcessMonitorNameList() {return process_monitor_name_list_;}
    std::vector<std::string> GetMnandHsMonitorUfsNodeList() {return mnand_hs_monitor_ufs_node_list_;}
    std::vector<std::string> GetMnandHsMonitorEmmcNodeList() {return mnand_hs_monitor_emmc_node_list_;}
    std::vector<std::string> GetNetworkMonitorNicList() {return network_monitor_nic_list_;}

    // file protect path
    void GetFileProtectPath(std::vector<SystemMonitorFileMonitorInfo>& pathList) {pathList = file_protect_path_list_;}

    // file monitor path
    void GetFileMonitorPath(std::vector<SystemMonitorFileMonitorInfo>& pathList) {pathList = file_monitor_path_list_;}


    // For Test
    void QueryPrintConfigData();

private:
    char* GetJsonAll(const char *fname);
    void ParseSystemMonitorConfigJson();

private:
    SystemMonitorConfig();
    SystemMonitorConfig(const SystemMonitorConfig &);
    SystemMonitorConfig & operator = (const SystemMonitorConfig &);

private:
    static std::mutex mtx_;
    static SystemMonitorConfig* instance_;

    // system monitor config info
    SystemMonitorConfigInfo system_monitor_config_info_;
    SystemMonitorTcpConfigInfo system_monitor_tcp_config_info_;
    std::unordered_map<std::string, SystemMonitorDiskMonitorPartitionInfo> disk_monitor_path_list_;
    std::vector<SystemMonitorDiskMonitorLogMoveListInfo> disk_monitor_soc_log_move_list_;
    std::vector<SystemMonitorDiskMonitorLogMoveListInfo> disk_monitor_mcu_log_move_list_;
    std::vector<std::string> process_monitor_name_list_;
    std::vector<std::string> mnand_hs_monitor_ufs_node_list_;
    std::vector<std::string> mnand_hs_monitor_emmc_node_list_;
    std::vector<SystemMonitorFileMonitorInfo> file_protect_path_list_;
    std::vector<SystemMonitorFileMonitorInfo> file_monitor_path_list_;
    std::vector<std::string> network_monitor_nic_list_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_CONFIG_H
