/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: phm config
 */

#ifndef PHM_CONFIG_H
#define PHM_CONFIG_H

#include "phm/fault_manager/include/fault_cluster_value.h"
#include <mutex>
#include <unordered_map>
#include <vector>

#include "yaml-cpp/yaml.h"
#include "phm/common/include/phm_common_def.h"
#include "phm/include/phm_def.h"
#include "phm/fault_manager/include/module_config.h"

namespace hozon {
namespace netaos {
namespace phm {

class PHMConfig {

public:
    static PHMConfig* getInstance();

    int32_t Init(const std::string& configPath, std::shared_ptr<ModuleConfig> cfg);
    void DeInit();

    void LoadPhmConfig();
    const PhmConfigInfo& GetPhmConfigInfo() {return g_phm_config_info_;}
    std::unordered_map<uint32_t, std::vector<FaultClusterItem>>& GetFaultClusterMap();
    bool GetFaultInfoByFault(const uint32_t fault, PhmFaultInfo& faultInfo);
    void UpdateFaultStatus(const uint32_t fault, const uint8_t faultStatus);
    void GetRegistCluster(std::vector<ClusterItem>& clusterItem, std::shared_ptr<ModuleConfig> cfg);

    // For Test
    void QueryPrintConfigData(std::shared_ptr<ModuleConfig> cfg);

private:
    char* GetJsonAll(const char *fname);
    int32_t ParseFaultJson();
    int32_t ParseFaultClusterJson();

    void LoadInitYamlConfig(const std::string& configPath, std::shared_ptr<ModuleConfig> cfg);
    void LoadData(const YAML::Node& node, const std::string& type, std::shared_ptr<ModuleConfig> cfg);
    void ParseRuleData(const YAML::Node& node, const std::string& type, std::shared_ptr<ModuleConfig> cfg);
    void UpdateClusterLevel(const uint32_t fault, const uint8_t faultStatus);

private:
    PHMConfig();
    PHMConfig(const PHMConfig &);
    PHMConfig & operator = (const PHMConfig &);

private:
    static PHMConfig* instance_;
    static std::mutex mtx_;

    // phm config info
    PhmConfigInfo g_phm_config_info_;
    // unordered_map<fault, FaultClusterItem>
    std::unordered_map<uint32_t, std::vector<FaultClusterItem>> g_phm_fault_cluster_level_map_;
    // unordered_map<faultId, PhmFaultInfo>
    std::unordered_map<uint32_t, PhmFaultInfo> g_phm_fault_info_map_;

    bool update_cluster_level_flag_;
    FaultClusterValue fault_cluster_value_;
};
}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_CONFIG_H
