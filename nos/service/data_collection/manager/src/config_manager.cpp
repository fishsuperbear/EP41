/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: config_manager.cpp
 * @Date: 2023/08/14
 * @Author: cheng
 * @Desc: --
 */


#include "manager/include/config_manager.h"

namespace hozon {
namespace netaos {
namespace dc {

std::vector<TriggerIdPriority_> ConfigManager::triggerIdPriorityDynamic_vec;
std::mutex ConfigManager::mtx_;
std::mutex ConfigManager::mtxForDynamic;
std::string ConfigManager::triggerLimitYamlFilePath = "/app/runtime_service/data_collection/conf/dc_trigger_limit.yaml";

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
