/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: fm debounce count policy
*/

#include "phm/common/include/phm_logger.h"
#include "phm/fault_manager/include/module_config.h"
#include "phm/common/include/phm_config.h"
#include <algorithm>

namespace hozon {
namespace netaos {
namespace phm {

ModuleConfig::ModuleConfig()
: register_app_name_("")
, update_cluster_level_flag_(false)
, inhibit_all_fault_flag_(false)
{

}

ModuleConfig::~ModuleConfig()
{
    register_fault_list_.clear();
    register_combination_list_.clear();
    register_post_cluster_list_.clear();
    phm_tasks_.clear();
    inhibit_fault_list_.clear();
}

std::string&
ModuleConfig::GetModuleName()
{
    return register_app_name_;
}

void
ModuleConfig::SetModuleName(const std::string name)
{
    register_app_name_ = name;
}

std::set<uint32_t>&
ModuleConfig::GetRegisterFaultList()
{
    return register_fault_list_;
}

void
ModuleConfig::SetRegisterFaultList(std::set<uint32_t>& faultItemList)
{
    register_fault_list_.insert(faultItemList.begin(), faultItemList.end());
}

std::set<uint32_t>&
ModuleConfig::GetRegisterCombinationList()
{
    return register_combination_list_;
}

void
ModuleConfig::SetRegisterCombinationList(std::set<uint32_t> CombinationList)
{
    if (CombinationList.empty()) return;
    register_combination_list_.insert(CombinationList.begin(), CombinationList.end());
}

std::set<std::string>&
ModuleConfig::GetRegisterPostClusterList()
{
    return register_post_cluster_list_;
}

void
ModuleConfig::SetRegisterPostClusterList(std::set<std::string>& postClusterList)
{
    register_post_cluster_list_.insert(postClusterList.begin(), postClusterList.end());
    if (register_post_cluster_list_.size() > 0) {
        SetUpdateClusterLevelFlag(true);
    }
}

std::vector<uint32_t>&
ModuleConfig::GetInhibitFaultList()
{
    return inhibit_fault_list_;
}

void
ModuleConfig::SetInhibitFaultList(std::vector<uint32_t>& inhibitFaultList)
{
    inhibit_fault_list_.swap(inhibitFaultList);
}

std::vector<PhmTask>&
ModuleConfig::GetPhmTask()
{
    return phm_tasks_;
}

void
ModuleConfig::PushPhmTask(PhmTask phmTask)
{
    phm_tasks_.emplace_back(phmTask);
}

bool
ModuleConfig::GetUpdateClusterLevelFlag()
{
    return update_cluster_level_flag_;
}

void
ModuleConfig::SetUpdateClusterLevelFlag(bool flag)
{
    update_cluster_level_flag_ = flag;
}

bool
ModuleConfig::GetInhibitAllFaultFlag()
{
    return inhibit_all_fault_flag_;
}

void
ModuleConfig::SetInhibitAllFaultFlag(bool flag)
{
    inhibit_all_fault_flag_ = flag;
}

bool
ModuleConfig::IsRegistFault(const ReceiveFault_t& fault)
{
    PHM_TRACE << "ModuleConfig::IsRegistFault enter!";
    const uint32_t faultKey = fault.faultId * 100 + fault.faultObj;
    if (update_cluster_level_flag_ && IsRegistPostProcess(faultKey)) {
        return true;
    }

    if (IsRegistCombinationId(fault.faultCombinationId)) {
        return true;
    }

    if (IsRegistFaultId(faultKey)) {
        return true;
    }

    return false;
}

bool
ModuleConfig::IsRegistFaultId(const uint32_t faultKey)
{
    PHM_TRACE << "ModuleConfig::IsRegistFaultId enter! faultKey: " << faultKey
              << ",size: " << register_fault_list_.size();
    if (register_fault_list_.empty()) return false;

    auto itrFault = register_fault_list_.find(faultKey);
    if (itrFault == register_fault_list_.end()) {
        PHM_TRACE << "ModuleConfig::IsRegistFaultId enter! not find";
        return false;
    }

    PHM_TRACE << "ModuleConfig::IsRegistFaultId enter! find ok";
    return true;
}

bool
ModuleConfig::IsRegistPostProcess(const uint32_t fault)
{
    std::unordered_map<uint32_t, std::vector<FaultClusterItem>> cluster_map = PHMConfig::getInstance()->GetFaultClusterMap();
    PHM_DEBUG << "ModuleConfig::IsRegistPostProcess enter! fault: " << fault << ",size:" << cluster_map.size();
    if (cluster_map.empty()) return false;

    auto itrClusterItem = cluster_map.find(fault);
    if (itrClusterItem != cluster_map.end()) {
        auto iter = register_post_cluster_list_.find("all");
        if (iter != register_post_cluster_list_.end()) {
            return true;
        }

        for (auto& item : itrClusterItem->second) {
            auto iter = register_post_cluster_list_.find(item.clusterName);
            if (iter != register_post_cluster_list_.end()) {
                return true;
            }
        }
    }

    return false;
}

bool
ModuleConfig::IsRegistCombinationId(const uint32_t combinationId)
{
    PHM_DEBUG << "ModuleConfig::IsRegistCombinationId enter! combinationId: " << combinationId << ",size:" << register_combination_list_.size();
    if (register_combination_list_.empty()) return false;

    auto iter = register_combination_list_.find(combinationId);
    if (iter == register_combination_list_.end()) {
        return false;
    }

    return true;
}

bool
ModuleConfig::IsInhibitFault(const uint32_t fault)
{
    if (GetInhibitAllFaultFlag()) {
        return true;
    }

    auto itr = std::find(inhibit_fault_list_.begin(), inhibit_fault_list_.end(), fault);
    if (itr != inhibit_fault_list_.end()) {
        return true;
    }

    return false;
}

// fault inhibit
void
ModuleConfig::InhibitFault(const std::vector<uint32_t>& faultKeys)
{
    inhibit_fault_list_.assign(faultKeys.begin(), faultKeys.end());
}

void
ModuleConfig::RecoverInhibitFault(const std::vector<uint32_t>& faultKeys)
{
    for (auto& item : faultKeys) {
        auto itr = std::find(inhibit_fault_list_.begin(), inhibit_fault_list_.end(), item);
        if (itr != inhibit_fault_list_.end()) {
            inhibit_fault_list_.erase(itr);
        }
    }
}

void
ModuleConfig::InhibitAllFault()
{
    inhibit_all_fault_flag_ = true;
}

void
ModuleConfig::RecoverInhibitAllFault()
{
    inhibit_all_fault_flag_ = false;
    inhibit_fault_list_.clear();
}


}  // namespace phm
}  // namespace netaos
}  // namespace hozon
