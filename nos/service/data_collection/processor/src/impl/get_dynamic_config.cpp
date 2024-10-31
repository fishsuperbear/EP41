/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: get_dynamic_config.cpp
 * @Date: 2023/12/5
 * @Author: kun
 * @Desc: --
 */

#include "processor/include/impl/get_dynamic_config.h"

#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>

#include "json/json.h"
#include "utils/include/trans_utils.h"
#include "https.h"
#include "log_moudle_init.h"
#include "utils/include/sign_utils.h"

namespace hozon {
namespace netaos {
namespace dc {

CollectFlag BasicTask::collectFlag;

GetDynamicConfig::GetDynamicConfig() {
    m_taskStatus.store(TaskStatus::INITIAL, std::memory_order::memory_order_release);
    m_stopFlag = false;
}

GetDynamicConfig::~GetDynamicConfig() {}

void GetDynamicConfig::onCondition(std::string type, char* data, Callback callback) {}

void GetDynamicConfig::configure(std::string type, YAML::Node& node) {
    m_cdnConfigFilePath = node["cdnConfigFilePath"].as<std::string>();
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void GetDynamicConfig::configure(std::string type, DataTrans& node) {
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void GetDynamicConfig::active() {

    if (m_stopFlag == true) {
        m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
        return;
    }
    m_taskStatus.store(TaskStatus::RUNNING, std::memory_order::memory_order_release);
    DC_SERVER_LOG_DEBUG << "get dynamic config: begin";
    std::lock_guard<std::mutex> lg(m_mtx);

    // 获取CDN配置
    if (m_cdnConfigFilePath.empty() || !PathUtils::isFileExist(m_cdnConfigFilePath)) {
        DC_SERVER_LOG_DEBUG << "get dynamic config: cdn json file not exist";
        return;
    }
    std::string triggerDynamicConfigStr;
    SignUtils::ReadFileWithLock(m_cdnConfigFilePath.c_str(), triggerDynamicConfigStr);
    std::string triggerDynamicConfigMD5 = SignUtils::getMd5(triggerDynamicConfigStr);
    if (m_oldTriggerDynamicConfigMD5 == triggerDynamicConfigMD5) {
        DC_SERVER_LOG_DEBUG << "get dynamic config: config no change";
        return;
    } else {
        m_oldTriggerDynamicConfigMD5 = triggerDynamicConfigMD5;
    }
    Json::Reader reader;
	Json::Value rootValue;
    reader.parse(triggerDynamicConfigStr, rootValue, false);
    if (rootValue["code"].asUInt64() != 200) {
        DC_SERVER_LOG_WARN << "get dynamic config: get config failed";
        m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
        return;
    }

    // 修改各数据collect flag
    BasicTask::collectFlag.logCollect = rootValue["data"]["strategyData"]["logCollect"].asBool();
    BasicTask::collectFlag.faultCollect = rootValue["data"]["strategyData"]["faultCollect"].asBool();
    BasicTask::collectFlag.calibrationCollect = rootValue["data"]["strategyData"]["calibrationCollect"].asBool();

    // 更新动态trigger配置
    YAML::Node config;
    Mapping_ dynamicCfgMapping;
    std::vector<std::string> cdnTrueTriVec;
    if (m_model.task.empty()) {
        try {
            YAML::Node root = YAML::LoadFile("/app/runtime_service/data_collection/conf/dc_mapping_dynamic.yaml");
            dynamicCfgMapping = root.as<Mapping_>();
        } catch (const YAML::Exception& e) {
            DC_SERVER_LOG_ERROR << e.what();
            m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
            return;
        }
        for (auto cfg : dynamicCfgMapping.mapping) {
            if (cfg.triggerId == "0") {
                m_model = cfg;
                break;
            }
        }
    }
    dynamicCfgMapping.mapping.clear();
    dynamicCfgMapping.mapping.push_back(m_model);
    if (rootValue["data"]["strategyData"]["strategies"].isArray()) {
        for (uint i = 0; i < rootValue["data"]["strategyData"]["strategies"].size(); i++) {
            TriggerIdPriority_ trigger;
            trigger.triggerId = rootValue["data"]["strategyData"]["strategies"][i]["triggerId"].asString();
            trigger.task = m_model.task;
            trigger.priority = rootValue["data"]["strategyData"]["strategies"][i]["priority"].asUInt64();
            dynamicCfgMapping.mapping.push_back(trigger);
        }
    } else {
        DC_SERVER_LOG_DEBUG << "get dynamic config: strategies is null";
    }
    ConfigManager::changeDynamicMapping(dynamicCfgMapping.mapping);
    createYaml(dynamicCfgMapping, "/app/runtime_service/data_collection/conf/dc_mapping_dynamic.yaml");

    // 更新limit
    if (rootValue["data"]["strategyData"]["limit"]["cycle"].isString() && rootValue["data"]["strategyData"]["limit"]["triggerCount"].isInt()) {
        std::string cycle = rootValue["data"]["strategyData"]["limit"]["cycle"].asString();
        int limit = rootValue["data"]["strategyData"]["limit"]["triggerCount"].asInt();
        ConfigManager::changeLimitAndCycle(cycle, limit);
    } else {
        DC_SERVER_LOG_DEBUG << "get dynamic config: end";
    }
    
    DC_SERVER_LOG_DEBUG << "get dynamic config: end";
}

void GetDynamicConfig::deactive() {
    m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
    m_stopFlag = true;
}

TaskStatus GetDynamicConfig::getStatus() {
    return m_taskStatus.load(std::memory_order::memory_order_acquire);
}

bool GetDynamicConfig::getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) {
    if (m_taskStatus.load(std::memory_order::memory_order_acquire) != TaskStatus::FINISHED) {
        return false;
    } else {
        return true;
    }
}

void GetDynamicConfig::pause() {}

void GetDynamicConfig::doWhenDestroy(const Callback& callback) {
    m_cb = callback;
}

void GetDynamicConfig::createYaml(Mapping_ mapping, std::string yamlFilePath) {
    std::string tempYamlFilePath = yamlFilePath + ".temp";
    std::ofstream output_file(tempYamlFilePath, std::ios::out | std::ios::binary);
    output_file << "mapping:";
    for (auto cfg : mapping.mapping) {
        output_file << "\n    - triggerId: ";
        output_file << cfg.triggerId;
        output_file << "\n      task: ";
        output_file << cfg.task;
        output_file << "\n      priority: ";
        output_file << cfg.priority;
    }
    output_file.close();

    std::string yamlFolderPath = PathUtils::getFolderName(yamlFilePath);
    std::string oldYamlFolderPath = PathUtils::getFilePath(yamlFolderPath, "old");
    PathUtils::createFoldersIfNotExists(oldYamlFolderPath);
    std::string oldYamlFileName = PathUtils::getFileName(yamlFilePath) + "_" + TransUtils::stringTransFileName("%Y%m%d-%H%M%S");
    PathUtils::renameFile(yamlFilePath, PathUtils::getFilePath(oldYamlFolderPath, oldYamlFileName));
    PathUtils::renameFile(tempYamlFilePath, yamlFilePath);
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
