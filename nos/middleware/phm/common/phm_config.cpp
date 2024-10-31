/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: config file
 */

#include <regex>
#include <iostream>
#include <unordered_set>

#include "json/json.h"
#include "phm/common/include/phm_logger.h"
#include "phm/common/include/phm_config.h"

namespace hozon {
namespace netaos {
namespace phm {

    const std::string APPNAME = "appName";
    const std::string CLUSTER = "cluster";
    const std::string FAULT = "fault";
    const std::string COMBINATION = "combination";
    const std::string CUSTOMCOMBINATION = "customCombination";
    const std::string PHMMONITOR = "phmMonitor";
    const std::string REGEX = "-";

    PHMConfig* PHMConfig::instance_ = nullptr;
    std::mutex PHMConfig::mtx_;

    const int MAX_LOAD_SIZE = 1024;

    #ifdef BUILD_FOR_MDC
        const std::string PHM_CONFIG_PATH = "/opt/usr/diag_update/mdc-llvm/conf/phm_config.json";
        const std::string PHM_FAULT_LIST_PATH = "/opt/usr/diag_update/mdc-llvm/conf/phm_fault_list.json";
        const std::string PHM_FAULT_CLUSTER_LIST_PATH = "/opt/usr/diag_update/mdc-llvm/conf/phm_fault_cluster_list.json";
    #elif BUILD_FOR_J5
        const std::string PHM_CONFIG_PATH = "/userdata/diag_update/j5/conf/phm_config.json";
        const std::string PHM_FAULT_LIST_PATH = "/userdata/diag_update/j5/conf/phm_fault_list.json";
        const std::string PHM_FAULT_CLUSTER_LIST_PATH = "/userdata/diag_update/j5/conf/phm_fault_cluster_list.json";
    #elif BUILD_FOR_ORIN
        const std::string PHM_CONFIG_PATH = "/app/conf/phm_cli_config.json";
        const std::string PHM_FAULT_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_list.json";
        const std::string PHM_FAULT_CLUSTER_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_cluster_list.json";
    #else
        const std::string PHM_CONFIG_PATH = "/app/conf/phm_cli_config.json";
        const std::string PHM_FAULT_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_list.json";
        const std::string PHM_FAULT_CLUSTER_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_cluster_list.json";
    #endif

    PHMConfig::PHMConfig(): update_cluster_level_flag_(false)
    {
    }

    PHMConfig*
    PHMConfig::getInstance()
    {
        if (nullptr == instance_) {
            std::lock_guard<std::mutex> lck(mtx_);
            if (nullptr == instance_) {
                instance_ = new PHMConfig();
            }
        }

        return instance_;
    }

    int32_t
    PHMConfig::Init(const std::string& configPath, std::shared_ptr<ModuleConfig> cfg)
    {
        PHM_INFO << "PHMConfig::Init enter! configPath: " << configPath.c_str();
        int32_t ret = 0;

        // phm_fault_cluster_list.json
        if (ParseFaultClusterJson() < 0) {
            PHM_ERROR << "PHMConfig::Init parser ParseFaultClusterJson failed";
            ret = -1;
        }
        PHM_INFO << "PHMConfig::Initg_phm_fault_cluster_level_map_.size: " << g_phm_fault_cluster_level_map_.size();

        // phm_fault_list.json
        if (ParseFaultJson() < 0) {
            PHM_ERROR << "PHMConfig::Init parser ParseFaultJson failed";
            ret = -1;
        }
        PHM_INFO << "PHMConfig::Init g_phm_fault_info_map_.size: " << g_phm_fault_info_map_.size();

        // config yaml file
        LoadInitYamlConfig(configPath, cfg);

        // QueryPrintConfigData(cfg);
        return ret;
    }

    void
    PHMConfig::DeInit()
    {
        PHM_INFO << "PHMConfig::DeInit enter!";
        g_phm_fault_cluster_level_map_.clear();
        g_phm_fault_info_map_.clear();

        if (instance_ != nullptr) {
            delete instance_;
            instance_ = nullptr;
        }
    }

    char*
    PHMConfig::GetJsonAll(const char *fname)
    {
        FILE *fp;
        char *str;
        char txt[MAX_LOAD_SIZE];
        int filesize;
        if ((fp = fopen(fname, "r")) == NULL) {
            return NULL;
        }

        fseek(fp, 0, SEEK_END);
        filesize = ftell(fp);
        str = (char*)malloc(filesize + 1);
        memset(str, 0, filesize);
        rewind(fp);
        while ((fgets(txt, MAX_LOAD_SIZE, fp)) != NULL) {
            strcat(str, txt);
        }

        fclose(fp);
        return str;
    }

    int32_t
    PHMConfig::ParseFaultClusterJson()
    {
        PHM_TRACE << "PHMConfig::ParseFaultClusterJson config path: " << PHM_FAULT_CLUSTER_LIST_PATH;
        if (g_phm_fault_cluster_level_map_.size() > 0) {
            return 0;
        }
        char* jsonstr = GetJsonAll(PHM_FAULT_CLUSTER_LIST_PATH.c_str());
        if (nullptr == jsonstr) {
            PHM_ERROR << "PHMConfig::ParseFaultClusterJson jsonstr is nullptr.";
            return -1;
        }

        Json::CharReaderBuilder readerBuilder;
        std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
        Json::Value  rootValue;
        JSONCPP_STRING errs;
        bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &rootValue, &errs);
        if (!res || !errs.empty()) {
            PHM_ERROR << "PHMConfig::ParseFaultClusterJson jsonreader error.";
            if (jsonstr != NULL) {
                free(jsonstr);
            }
            return -2;
        }

        Json::Value & faultClusterValue = rootValue["Faults"];
        std::vector<FaultClusterItem> faultCluster;
        for (uint32_t i = 0; i < faultClusterValue.size(); i++) {
            uint32_t faultKey = static_cast<uint32_t>(faultClusterValue[i]["FaultKey"].asUInt());
            faultCluster.clear();
            Json::Value & clusterValue = faultClusterValue[i]["Cluster"];
            FaultClusterItem clusterItem;
            for (uint32_t j = 0; j < clusterValue.size(); j++) {
                clusterItem.clusterName = clusterValue[j]["Name"].asString();
                clusterItem.bitPosition = static_cast<uint8_t>(clusterValue[j]["BitPosition"].asUInt());
                faultCluster.emplace_back(clusterItem);
            }

            g_phm_fault_cluster_level_map_.insert(std::make_pair(faultKey, faultCluster));
        }

        if (jsonstr != NULL) {
            free(jsonstr);
        }

        return 0;
    }

    int32_t
    PHMConfig::ParseFaultJson()
    {
        PHM_TRACE << "PHMConfig::ParseFaultJson config path: " << PHM_FAULT_LIST_PATH;
        if (g_phm_fault_info_map_.size() > 0) {
            return 0;
        }

        char* jsonstr = GetJsonAll(PHM_FAULT_LIST_PATH.c_str());
        if (nullptr == jsonstr) {
            PHM_ERROR << "PHMConfig::ParseFaultJson jsonstr is nullptr.";
            return -1;
        }

        Json::CharReaderBuilder readerBuilder;
        std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
        Json::Value  rootValue;
        JSONCPP_STRING errs;

        bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &rootValue, &errs);
        if (!res || !errs.empty()) {
            PHM_ERROR << "PHMConfig::ParseFaultJson jsonreader error.";
            if (jsonstr != NULL) {
                free(jsonstr);
            }
            return -2;
        }

        Json::Value & resultValue = rootValue["FaultList"];
        for (uint32_t i = 0; i < resultValue.size(); i++) {
            Json::Value clusterValue = resultValue[i]["FaultCluster"];
            uint32_t clusterId = static_cast<uint32_t>(clusterValue["ClusterID"].asUInt());
            std::string process = clusterValue["Process"].asString();

            Json::Value faultsValue = resultValue[i]["Faults"];
            for (uint32_t j = 0; j < faultsValue.size(); j++) {
                PhmFaultInfo faultInfo;
                faultInfo.faultClusterId = clusterId;
                faultInfo.faultProcess = process;
                uint32_t key = static_cast<uint32_t>(faultsValue[j]["Key"].asUInt());
                faultInfo.faultObj = static_cast<uint8_t>(key % 100);
                faultInfo.faultId = static_cast<uint32_t>((key - faultInfo.faultObj) / 100);
                faultInfo.faultLevel = static_cast<uint16_t>(faultsValue[j]["Level"].asUInt());
                std::string strDtc = faultsValue[j]["DtcCode"].asString();
                faultInfo.dtcCode = static_cast<uint32_t>(std::strtoul(strDtc.c_str(), 0, 0));
                faultInfo.faultDescribe = faultsValue[j]["Dscribe"].asString();
                faultInfo.faultAction.record = static_cast<uint8_t>(faultsValue[j]["Action"]["record"].asUInt());
                faultInfo.faultAction.analysis = static_cast<uint8_t>(faultsValue[j]["Action"]["analysis"].asUInt());
                faultInfo.faultAction.dtcMapping = static_cast<uint8_t>(faultsValue[j]["Action"]["dtcMapping"].asUInt());
                faultInfo.faultAction.strategy.notifyMcu = static_cast<uint8_t>(faultsValue[j]["Action"]["Strategy"]["notify_mcu"].asUInt());
                faultInfo.faultAction.strategy.notifyApp = static_cast<uint8_t>(faultsValue[j]["Action"]["Strategy"]["notify_app"].asUInt());
                faultInfo.faultAction.strategy.restartproc = static_cast<uint8_t>(faultsValue[j]["Action"]["Strategy"]["restartproc"].asUInt());

                faultInfo.faultDomain = "";
                faultInfo.faultOccurTime = 0;
                faultInfo.faultStatus = 0;

                g_phm_fault_info_map_.insert(std::make_pair(key, faultInfo));
            }
        }

        if (jsonstr != NULL) {
            free(jsonstr);
        }
        return 0;
    }

    std::unordered_map<uint32_t, std::vector<FaultClusterItem>>&
    PHMConfig::GetFaultClusterMap()
    {
        return g_phm_fault_cluster_level_map_;
    }

    bool
    PHMConfig::GetFaultInfoByFault(const uint32_t fault, PhmFaultInfo& faultInfo)
    {
        auto itr = g_phm_fault_info_map_.find(fault);
        if (itr == g_phm_fault_info_map_.end()) {
            PHM_WARN << "PHMConfig::GetFaultInfoByFault error fault: " << fault;
            return false;
        }

        faultInfo = itr->second;
        return true;
    }

    void
    PHMConfig::UpdateFaultStatus(const uint32_t fault, const uint8_t faultStatus)
    {
        PHM_DEBUG << "PHMConfig::UpdateFaultStatus fault: " << fault << " faultStatus: " << (int)faultStatus;
        std::lock_guard<std::mutex> lck(mtx_);
        auto itr = g_phm_fault_info_map_.find(fault);
        if (itr == g_phm_fault_info_map_.end()) {
            PHM_WARN << "PHMConfig::UpdateFaultStatus error fault: " << fault;
            return;
        }

        if (faultStatus != itr->second.faultStatus) {
            itr->second.faultStatus = faultStatus;

            // when init config yaml register cluster
            if (update_cluster_level_flag_) {
                UpdateClusterLevel(fault, faultStatus);
            }
        }
    }

    void
    PHMConfig::UpdateClusterLevel(const uint32_t fault, const uint8_t faultStatus)
    {
        PHM_DEBUG << "PHMConfig::UpdateClusterLevel fault: " << fault << " faultStatus: " << (int)faultStatus;
        auto itr = g_phm_fault_cluster_level_map_.find(fault);
        if (itr == g_phm_fault_cluster_level_map_.end()) {
            PHM_WARN << "PHMConfig::UpdateClusterLevel fault: " << fault << " no cluster.";
            return;
        }

        fault_cluster_value_.UpdateClusterData(fault, faultStatus, itr->second);
        return;
    }

    void
    PHMConfig::GetRegistCluster(std::vector<ClusterItem>& clusterItem, std::shared_ptr<ModuleConfig> cfg)
    {
        if (!(update_cluster_level_flag_)) {
            return;
        }
        clusterItem.clear();

        std::set<std::string>& register_post_cluster_list = cfg->GetRegisterPostClusterList();
        auto itr = register_post_cluster_list.find("all");
        if (itr != register_post_cluster_list.end()) {
            fault_cluster_value_.GetClusterValueData(clusterItem);
        }
        else {
            for (auto& clusterName : register_post_cluster_list) {
                ClusterItem cluster;
                fault_cluster_value_.GetClusterValueData(clusterName, cluster);
                clusterItem.emplace_back(cluster);
            }
        }
    }

    static
    std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr = REGEX)
    {
        std::regex re(regexStr);
        std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
        return {first, last};
    }

    void
    PHMConfig::LoadInitYamlConfig(const std::string& configPath, std::shared_ptr<ModuleConfig> cfg)
    {
        PHM_INFO << "PHMConfig::LoadInitYamlConfig configPath: " << configPath.c_str();
        if (configPath.empty()) {
            PHM_WARN << "PHMConfig::LoadInitYamlConfig configPath is empty!";
            return;
        }

        if (0 != access(configPath.c_str(), F_OK)) {
            PHM_ERROR << "PHMConfig::LoadInitYamlConfig file: " << configPath.c_str() << "not exist!";
            return;
        }

        YAML::Node config = YAML::LoadFile(configPath);
        if (!config) {
            PHM_ERROR << "PHMConfig::LoadInitYamlConfig open file node failed!";
            return;
        }

        if (config[APPNAME]) {
            cfg->SetModuleName(config[APPNAME].as<std::string>());
            PHM_INFO << "PHMConfig::LoadInitYamlConfig appName " << cfg->GetModuleName();
        }

        if (config[CLUSTER]) {
            std::set<std::string> register_post_cluster_list;
            for (auto item : config[CLUSTER]) {
                register_post_cluster_list.emplace(item.as<std::string>());
                PHM_INFO << "PHMConfig::LoadInitYamlConfig post process cluster_name " << item.as<std::string>();
            }
            cfg->SetRegisterPostClusterList(register_post_cluster_list);
        }

        LoadData(config, FAULT, cfg);
        LoadData(config, COMBINATION, cfg);

        if (config[PHMMONITOR]) {
            // PHM_INFO << "ruleMapping number: " << (int)config[PHMMONITOR].size();
            for (auto item : config[PHMMONITOR]) {
                // PHM_INFO << "ruleMapping id: " << item["monitorType"].as<string>();
                ParseRuleData(item, item["monitorType"].as<std::string>(), cfg);
            }
        }

        PHM_INFO << "PHMConfig::LoadInitYamlConfig register_fault_list_.size: " << cfg->GetRegisterFaultList().size()
            << " register_combination_list_.size: " << cfg->GetRegisterCombinationList().size()
            << " register_post_cluster_list_.size: " << cfg->GetRegisterPostClusterList().size()
            << " phm_tasks_.size: " << cfg->GetPhmTask().size();

        if (cfg->GetRegisterPostClusterList().size()) {
            update_cluster_level_flag_ = true;
        }

        PHM_DEBUG << "PHMConfig::LoadInitYamlConfig configPath ok";
    }

    void
    PHMConfig::LoadData(const YAML::Node& node, const std::string& type, std::shared_ptr<ModuleConfig> cfg)
    {
        PHM_INFO << "PHMConfig::LoadData enter type: " << type;
        if (!node) {
            PHM_ERROR << "PHMConfig::LoadData error node!";
            return;
        }

        if (node[type]) {
            // PHM_INFO << "PHMConfig::LoadData enter type: " << type;
            std::set<uint32_t> register_fault_list;
            std::set<uint32_t> register_combination_list;

            auto nodeIter = node[type].begin();
            for (; nodeIter != node[type].end(); nodeIter++) {
                // PHM_INFO << nodeIter->first.as<std::string>() << ":" << nodeIter->second.as<std::string>();
                auto firstVec = Split(nodeIter->first.as<std::string>());
                auto secondVec = Split(nodeIter->second.as<std::string>());

                if (FAULT == type) {
                    uint32_t faultID = 0, faultIDSize = 0, objID = 0, objIDSize = 0;
                    faultID = (uint32_t)std::stoi(firstVec[0]);
                    objID = (uint32_t)std::stoi(secondVec[0]);
                    if (firstVec.size() > 1) {
                        faultIDSize = (uint32_t)std::stoi(firstVec[1]) - (faultID % 10);
                    }

                    if (secondVec.size() > 1) {
                        objIDSize = (uint32_t)std::stoi(secondVec[1]) - objID;
                    }

                    // PHM_INFO << "faultID: "<< faultID << ", faultIDSize: " << faultIDSize << ", objID: " << objID << ", objIDSize: " << objIDSize;
                    for (uint i = 0; i <= faultIDSize; i++) {
                        for (uint j = 0; j <= objIDSize; j++) {
                            const uint32_t faultId = faultID + i;
                            const uint8_t faultObj = objID + j;
                            const uint32_t faultKey = faultId * 100 + faultObj;
                            // PHM_INFO << "faultKey: "<< faultKey;
                            register_fault_list.emplace(faultKey);
                        }
                    }
                    // PHM_INFO << "register_fault_list size: "<< register_fault_list.size();
                    continue;
                }

                if (COMBINATION == type) {
                    uint32_t clusterID, clusterIDSize = 0;
                    clusterID = (uint32_t)std::stoi(firstVec[0]);
                    if (firstVec.size() > 1) {
                        clusterIDSize = (uint32_t)std::stoi(firstVec[1]) - (clusterID % 10);
                    }

                    // PHM_INFO << "clusterID: "<< clusterID << ", clusterIDSize: " << clusterIDSize;
                    for (uint i = 0; i <= clusterIDSize; i++) {
                        uint32_t clusterId = clusterID + i;
                        register_combination_list.emplace(clusterId);
                    }

                    // PHM_INFO << "register_combination_list size: "<< register_combination_list.size();
                    continue;
                }
            }

            cfg->SetRegisterFaultList(register_fault_list);
            cfg->SetRegisterCombinationList(register_combination_list);
        }
    }

    void
    PHMConfig::LoadPhmConfig()
    {
        if (true == g_phm_config_info_.LoadFlag.load()) {
            return;
        }
        g_phm_config_info_.LoadFlag.store(true);

        do {
            char* jsonstr = GetJsonAll(PHM_CONFIG_PATH.c_str());
            if (nullptr == jsonstr) {
                std::cout << "PHMConfig::LoadPhmConfig[" << PHM_CONFIG_PATH << "] error jsonstr is nullptr." << std::endl;
                break;
            }

            Json::CharReaderBuilder readerBuilder;
            std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
            Json::Value  rootValue;
            JSONCPP_STRING errs;
            bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &rootValue, &errs);
            if (!res || !errs.empty()) {
                if (jsonstr != NULL) {
                    free(jsonstr);
                }
                break;
            }

            g_phm_config_info_.LogContextName = static_cast<std::string>(rootValue["LogContextName"].asString());
            g_phm_config_info_.LogLevel = static_cast<uint8_t>(rootValue["LogLevel"].asUInt());
            g_phm_config_info_.DebugSwitch = static_cast<std::string>(rootValue["DebugSwitch"].asString());
            g_phm_config_info_.SystemCheckTime = static_cast<uint32_t>(rootValue["SystemCheckTime"].asUInt());

            if (jsonstr != NULL) {
                free(jsonstr);
            }
        } while(false);

        PhmLogger::Instance().SetLogLevel(static_cast<int32_t>(g_phm_config_info_.LogLevel));
        PhmLogger::Instance().CreateLogger(g_phm_config_info_.LogContextName);
    }

    void
    PHMConfig::ParseRuleData(const YAML::Node& node, const std::string& type, std::shared_ptr<ModuleConfig> cfg)
    {
        PhmTask phm_task;
        std::string monitorType = node["monitorType"].as<std::string>();
        if (monitorType == "alive") {
            phm_task.monitorType = PHM_MONITOR_TYPE_ALIVE;
            phm_task.parameter.push_back(node["periodMs"].as<uint64_t>());
            phm_task.parameter.push_back(node["expectedIndication"].as<uint64_t>());
            phm_task.parameter.push_back(node["minMargin"].as<uint64_t>());
            phm_task.parameter.push_back(node["maxMargin"].as<uint64_t>());
        }
        else if (monitorType == "deadline") {
            phm_task.monitorType = PHM_MONITOR_TYPE_DEADLINE;
            phm_task.parameter.push_back(node["deadlineMinMs"].as<uint64_t>());
            phm_task.parameter.push_back(node["deadlineMaxMs"].as<uint64_t>());
        }
        else if (monitorType == "logic") {
            phm_task.monitorType = PHM_MONITOR_TYPE_LOGIC;
        }

        for (auto item : node["checkPointId"]) {
            phm_task.checkPointId.push_back(item.as<uint64_t>());
        }

        phm_task.faultId = node["faultId"].as<uint64_t>();
        phm_task.faultObj = node["faultObj"].as<uint64_t>();

        cfg->PushPhmTask(phm_task);
    }

    void
    PHMConfig::QueryPrintConfigData(std::shared_ptr<ModuleConfig> cfg)
    {
        /**************data print for test**************/
         PHM_DEBUG << "PHMConfig::LoadConfig print g_phm_config_info_"
                  << " LogContextName: " << g_phm_config_info_.LogContextName
                  << " LogLevel: " << static_cast<uint>(g_phm_config_info_.LogLevel)
                  << " DebugSwitch: " << g_phm_config_info_.DebugSwitch
                  << " SystemCheckTime: " << g_phm_config_info_.SystemCheckTime;

        int i = 0;
        for (auto& item : g_phm_fault_cluster_level_map_) {
            PHM_DEBUG << "PHMConfig::LoadConfig print g_phm_fault_cluster_level_map_[" << i << "] fault: " << item.first;
            for (auto& item1 : item.second) {
                 PHM_DEBUG << "PHMConfig::LoadConfig print g_phm_fault_cluster_level_map_[" << i << "]"
                           << " clusterName: " << item1.clusterName
                           << " bitPosition: " << (int)item1.bitPosition;
            }

            i++;
        }

        i = 0;
        for (auto& item : g_phm_fault_info_map_) {
             PHM_DEBUG << "PHMConfig::LoadConfig print g_phm_fault_info_map_[" << i << "]"
                      << " faultId: " << item.second.faultId
                      << " faultObj: " << static_cast<uint>(item.second.faultObj)
                      << " faultAction.record: " << static_cast<uint>(item.second.faultAction.record)
                      << " faultAction.analysis: " << static_cast<uint>(item.second.faultAction.analysis)
                      << " faultAction.dtcMapping: " << static_cast<uint>(item.second.faultAction.dtcMapping)
                      << " faultAction.strategy.notifyApp: " << static_cast<uint>(item.second.faultAction.strategy.notifyApp)
                      << " faultAction.strategy.notifyMcu: " << static_cast<uint>(item.second.faultAction.strategy.notifyMcu)
                      << " faultAction.strategy.restartproc: " << static_cast<uint>(item.second.faultAction.strategy.restartproc)
                      << " faultLevel: " << item.second.faultLevel
                      << " faultDomain: " << item.second.faultDomain
                      << " faultClusterId: " << item.second.faultClusterId
                      << " faultProcess: " << item.second.faultProcess
                      << " faultDescribe: " << item.second.faultDescribe
                      << " dtcCode: " << item.second.dtcCode
                      << " faultOccurTime: " << item.second.faultOccurTime
                      << " faultStatus: " << static_cast<uint>(item.second.faultStatus);
            i++;
        }

        i = 0;
        for (auto& item : cfg->GetRegisterFaultList()) {
            PHM_DEBUG << "PHMConfig::LoadConfig print register_fault_list_[" << i << "]" << " faultKey: " << item;
            i++;
        }

        i = 0;
        for (auto& item : cfg->GetRegisterCombinationList()) {
            PHM_DEBUG << "PHMConfig::LoadConfig print register_combination_list_[" << i << "]: combinationId:" << item;
            i++;
        }

        i = 0;
        for (auto& item : cfg->GetRegisterPostClusterList()) {
            PHM_DEBUG << "PHMConfig::LoadConfig print register_post_cluster_list_[" << i << "]: clusterName: " << item;
            i++;
        }

        i = 0;
        for (auto& item : cfg->GetPhmTask()) {
            PHM_DEBUG << "PHMConfig::LoadConfig print phm_tasks_[" << i << "]"
                     <<" monitorType: " << item.monitorType
                     << " faultId: " << item.faultId
                     << " faultObj: " << (int)item.faultObj;

            for (auto& item1 : item.checkPointId) {
                 PHM_DEBUG << "PHMConfig::LoadConfig print phm_tasks_[" << i << "]: checkPointId: " << item1;
            }

            i++;
        }

        /**************data print for test**************/
    }


}  // namespace phm
}  // namespace netaos
}  // namespace hozon
