/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: fm debounce base
 */

#ifndef FAULT_CONFIG_H
#define FAULT_CONFIG_H

#include <unordered_map>
#include <vector>
#include <set>

#include "yaml-cpp/yaml.h"
#include "phm/common/include/phm_common_def.h"
#include "phm/include/phm_def.h"
#include "phm/fault_manager/include/fault_cluster_value.h"

namespace hozon {
namespace netaos {
namespace phm {

class ModuleConfig {

public:
    ModuleConfig();
    ~ModuleConfig();

    std::string& GetModuleName();
    void SetModuleName(const std::string name);
    std::set<uint32_t>& GetRegisterFaultList();
    void SetRegisterFaultList(std::set<uint32_t>& faultItemList);
    std::set<uint32_t>& GetRegisterCombinationList();
    void SetRegisterCombinationList(std::set<uint32_t> CombinationList);
    std::set<std::string>& GetRegisterPostClusterList();
    void SetRegisterPostClusterList(std::set<std::string>& postClusterList);
    std::vector<uint32_t>& GetInhibitFaultList();
    void SetInhibitFaultList(std::vector<uint32_t>& inhibitFaultList);
    std::vector<PhmTask>& GetPhmTask();
    void PushPhmTask(PhmTask phmTask);
    bool GetUpdateClusterLevelFlag();
    void SetUpdateClusterLevelFlag(bool flag);
    bool GetInhibitAllFaultFlag();
    void SetInhibitAllFaultFlag(bool flag);

    bool IsRegistFault(const ReceiveFault_t& fault);
    bool IsInhibitFault(const uint32_t fault);

    // fault inhibit
    void InhibitFault(const std::vector<uint32_t>& faultKeys);
    void RecoverInhibitFault(const std::vector<uint32_t>& faultKeys);
    void InhibitAllFault();
    void RecoverInhibitAllFault();

private:

    bool IsRegistPostProcess(const uint32_t fault);
    bool IsRegistCombinationId(const uint32_t combinationId);
    bool IsRegistFaultId(const uint32_t faultKey);

    std::string register_app_name_;
    std::set<uint32_t> register_fault_list_;
    std::set<uint32_t> register_combination_list_;
    std::set<std::string> register_post_cluster_list_;
    std::vector<PhmTask> phm_tasks_;
    std::vector<uint32_t> inhibit_fault_list_;
    bool update_cluster_level_flag_;
    bool inhibit_all_fault_flag_;
};

}  // namespace phm
}  // namespace netaos
}  // namespace hozon

#endif // FAULT_CONFIG_H