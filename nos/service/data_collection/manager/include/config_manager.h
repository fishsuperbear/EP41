/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: config_manager.h
 * @Date: 2023/08/14
 * @Author: cheng
 * @Desc: --
 */

#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_MANAGER_CONFIG_MANAGER_H
#define MIDDLEWARE_TOOLS_DATA_COLLECT_MANAGER_CONFIG_MANAGER_H


#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include "basic/task.h"
#include "common/dc_consts.h"
#include "utils/include/dc_logger.hpp"
#include "utils/include/time_utils.h"
#include "utils/include/path_utils.h"
#include "utils/include/trans_utils.h"

namespace hozon {
namespace netaos {
namespace dc {

struct TriggerIdPriority_ {
    std::string triggerId;
    std::string task;
    int priority;
};
struct Mapping_ {
    std::vector<TriggerIdPriority_> mapping;
};

class ConfigManager {
   public:
    ConfigManager() {}

    ~ConfigManager() {}

    class Type {
       public:
        STATIC_CONST_CHAR_P Collections = "Collections";
        STATIC_CONST_CHAR_P Destination = "Destination";
        STATIC_CONST_CHAR_P Processor = "Processor";
        STATIC_CONST_CHAR_P LifyCycle = "LifyCycle";
        STATIC_CONST_CHAR_P TriggerTask = "BasicTask";
        STATIC_CONST_CHAR_P PresetTask = "PresetTask";

        static bool isValid(const std::string& target) {
            STATIC_CONST_CHAR_P aaa[] = {Collections, Destination, Processor, LifyCycle, TriggerTask, PresetTask};
            for (auto item : aaa) {
                if (target == item) {
                    return true;
                }
            }
            return false;
        }
    };

    int getRunningThreadNum(std::string type) {
        const int defaultRunningThreadNum = 5;
        if (!currentConfig_[k_0_dcConfig][type]["runningThreadNum"]) {
            return defaultRunningThreadNum;
        }
        try {
            return currentConfig_[k_0_dcConfig][type]["runningThreadNum"].as<int>();

        } catch (const YAML::Exception& e) {
            DC_SERVER_LOG_ERROR << e.what();
            return defaultRunningThreadNum;
        }
    }

    int getMaxThreadNum(std::string type) {
        const int defaultRunningThreadNum = 9;
        if (!currentConfig_[k_0_dcConfig][type]["maxThreadNum"]) {
            return defaultRunningThreadNum;
        }
        try {
            return currentConfig_[k_0_dcConfig][type]["maxThreadNum"].as<int>();
        } catch (const YAML::Exception& e) {
            DC_SERVER_LOG_ERROR << e.what();
            return defaultRunningThreadNum;
        }
    }

    inline const YAML::Node & cnode(const YAML::Node &n) {
        return n;
    }

    YAML::Node merge_nodes(YAML::Node a, YAML::Node b)
    {
        if (b.IsSequence() && a.IsSequence()) {
            YAML::Node subList;
            int num = a.size()>b.size()?a.size():b.size();
            for (std::size_t i = 0; i < num; i++) {
                subList.push_back(merge_nodes(a[i],b[i]));
            }
            return subList;
        }
        if (!b.IsMap()) {
            // If b is not a map, merge result is b, unless b is null
            return b.IsNull() ? a : b;
        }
        if (!a.IsMap()) {
            // If a is not a map, merge result is b
            return b;
        }
        if (!b.size()) {
            // If a is a map, and b is an empty map, return a
            return a;
        }
        // Create a new map 'c' with the same mappings as a, merged with b
        auto c = YAML::Node(YAML::NodeType::Map);
        for (auto n : a) {
            if (n.first.IsScalar()) {
                const std::string & key = n.first.Scalar();
                auto t = YAML::Node(cnode(b)[key]);
                if (t) {
                    c[n.first] = merge_nodes(n.second, t);
                    continue;
                }
            }
            c[n.first] = n.second;
        }
        // Add the mappings from 'b' not already in 'c'
        for (auto n : b) {
            if (!n.first.IsScalar() || !cnode(c)[n.first.Scalar()]) {
                c[n.first] = n.second;
            }
        }
        return c;
    }

    bool loadYaml(const std::string& yamlFilePath) {
        try {
            currentConfig_ = YAML::LoadFile(yamlFilePath);
        } catch (YAML::Exception& e) {
            DC_SERVER_LOG_ERROR <<"load yaml error:"<< e.what();
            return false;
        }
        return true;
    }

    std::string getTimerPlane() {
        std::string plane = "MP";
        if (!currentConfig_[k_0_dcConfig]["TimerCfg"]["plane"]) {
            return plane;
        }
        try {
            return currentConfig_[k_0_dcConfig]["TimerCfg"]["plane"].as<std::string>();
        } catch (const YAML::Exception& e) {
            DC_SERVER_LOG_ERROR <<"get timer plane error:"<< e.what();
            return plane;
        }
    }

    int32_t getTimerOffsetSecond() {
        int32_t offsetSecond = 0;
        if (!currentConfig_[k_0_dcConfig]["TimerCfg"]["offsetSec"]) {
            return offsetSecond;
        }
        try {
            return currentConfig_[k_0_dcConfig]["TimerCfg"]["offsetSecond"].as<int32_t>();
        } catch (const YAML::Exception& e) {
            DC_SERVER_LOG_ERROR <<"getTimerOffsetSecond error:"<< e.what();
            return offsetSecond;
        }
    }

    bool getCommonConfigure(const char* type, const std::string& name, YAML::Node& target) {
        std::lock_guard<std::mutex> lg(mtx_);
        if (!Type::isValid(type)) {
            DC_SERVER_LOG_ERROR << type << " is invalid in config manager.";
            return false;
        }
        if (!currentConfig_[k_0_dcConfig][type][name]) {
            return false;
        }
        if (currentConfig_[k_0_dcConfig][type][name]["<<"]) {
            // 原生的yaml-cpp 不支持merge, 这里简单实现一下功能。
            currentConfig_[k_0_dcConfig][type][name] = merge_nodes(currentConfig_[k_0_dcConfig][type][name]["<<"], currentConfig_[k_0_dcConfig][type][name]);
            currentConfig_[k_0_dcConfig][type][name].remove("<<");
        }
        target = currentConfig_[k_0_dcConfig][type][name];
        return true;
    }

    bool getLifeCycle(const std::string& name, YAML::Node& target) {
        if (!currentConfig_[k_0_dcConfig][Type::LifyCycle][name]) {
            return false;
        }
        target = currentConfig_[k_0_dcConfig][Type::LifyCycle][name];
        return true;
    }


    bool getPipeTaskFromName(const std::string& triggerTaskName, PipeLineTask& target) {
        if (!currentConfig_[k_0_dcConfig][Type::TriggerTask][triggerTaskName]) {
            return false;
        }
        target = currentConfig_[k_0_dcConfig][Type::TriggerTask][triggerTaskName].as<PipeLineTask>();
        return true;
    }

    bool getTriggerPipeTask(const std::string& triggerId, PipeLineTask& target) {
        auto it = std::find_if(triggerIdPriorityStatic_vec.begin(), triggerIdPriorityStatic_vec.end(), [&triggerId](const TriggerIdPriority_& item) -> bool {
            if (triggerId == item.triggerId) {
                return true;
            }
            return false;
        });
        if (it == triggerIdPriorityStatic_vec.end()) {
            std::lock_guard<std::mutex> lg(mtxForDynamic);
            it = std::find_if(triggerIdPriorityDynamic_vec.begin(), triggerIdPriorityDynamic_vec.end(), [&triggerId](const TriggerIdPriority_& item) -> bool {
                if (triggerId == item.triggerId) {
                    return true;
                }
                return false;
            });
            if (it == triggerIdPriorityDynamic_vec.end()) {
                DC_SERVER_LOG_ERROR << "cannot find task for trigger id " << triggerId;
                return false;
            }
        }
        auto triggerTaskName = it->task;
        DC_SERVER_LOG_INFO << "find task " << triggerTaskName << " for trigger id " << triggerId;
        if (currentConfig_[k_0_dcConfig][Type::TriggerTask][triggerTaskName]) {
            target = currentConfig_[k_0_dcConfig][Type::TriggerTask][triggerTaskName].as<PipeLineTask>();
        } else if (currentConfig_[k_0_dcConfig][Type::PresetTask][triggerTaskName]) {
            target = currentConfig_[k_0_dcConfig][Type::PresetTask][triggerTaskName].as<PipeLineTask>();
        } else {
            DC_SERVER_LOG_ERROR << "cannot find task " << triggerTaskName << " from config";
            return false;
        }
        try {
            std::lock_guard<std::mutex> lg(mtx_);
            YAML::Node root = YAML::LoadFile(triggerLimitYamlFilePath);
            int date = root["date"].as<int>();
            int triggerCount = root["triggerCount"].as<int>();
            std::string cycle = root["cycle"].as<std::string>();
            int limit = root["limit"].as<int>();
            int currentDate = std::stoi(TransUtils::stringTransFileName("%Y%m%d"));
            if (cycle == "day") {
                if (currentDate != date) {
                    date = currentDate;
                    triggerCount = 0;
                }
            } else if (cycle == "month") {
                if ((currentDate < date) || ((currentDate - date) >= 100)) {
                    date = currentDate;
                    triggerCount = 0;
                }
            } else {
                DC_SERVER_LOG_ERROR << "wrong cycle config";
                return false;
            }
            triggerCount++;
            if (triggerCount > limit) {
                DC_SERVER_LOG_ERROR << "trigger count is more than limit";
                return false;
            }
            std::ofstream output_file(triggerLimitYamlFilePath, std::ios::out | std::ios::binary);
            output_file << "date: ";
            output_file << date;
            output_file << "\ntriggerCount: ";
            output_file << triggerCount;
            output_file << "\ncycle: ";
            output_file << cycle;
            output_file << "\nlimit: ";
            output_file << limit;
            output_file.close();
        } catch (const YAML::Exception& e) {
            DC_SERVER_LOG_ERROR << e.what();
            return false;
        }
        target.taskName = triggerId + "_" + TimeUtils::timestamp2ReadableStr(TimeUtils::getDataTimestamp());
        target.trigger_id = triggerId;
        target.priority = it->priority;
        return true;
    }

    bool getPresetPipeTask(const std::string& presetTaskName, PipeLineTask& target) {
        if (!currentConfig_[k_0_dcConfig][Type::PresetTask]) {
            return false;
        }
        if (!currentConfig_[k_0_dcConfig][Type::PresetTask][presetTaskName]) {
            return false;
        }
        target = currentConfig_[k_0_dcConfig][Type::PresetTask][presetTaskName].as<PipeLineTask>();
        return true;
    }

    bool getPresetPipeTaskNams(std::vector<std::string>& target) {
        if (!currentConfig_[k_0_dcConfig][Type::PresetTask]) {
            return false;
        }
        if (!currentConfig_[k_0_dcConfig][Type::PresetTask]["taskNames"]) {
            return false;
        }
        target = currentConfig_[k_0_dcConfig][Type::PresetTask]["taskNames"].as<std::vector<std::string>>();
        return true;
    }

    bool getTriggerPipeTask(const std::string& triggerTaskName, YAML::Node& target) { return getCommonConfigure(Type::TriggerTask, triggerTaskName, target); }

    bool getCollectionConfigure(const std::string& collectionName, YAML::Node& target) { return getCommonConfigure(Type::Collections, collectionName, target); }

    bool getProcessConfigure(const std::string& processName, YAML::Node& target) { return getCommonConfigure(Type::Processor, processName, target); }

    bool getDestinationConfigure(const std::string& uploadName, YAML::Node& target) { return getCommonConfigure(Type::Destination, uploadName, target); }

    static TaskPriority getPriority(const YAML::Node node) {
        if (!node[k_3_priority]) {
            return TaskPriority::LOW;
        }
        int enumValue = node[k_3_priority].as<int>();
        if (enumValue > TaskPriority::EXTRA_HIGH) {
            return TaskPriority::EXTRA_HIGH;
        }
        if (enumValue < TaskPriority::EXTRA_LOW) {
            return TaskPriority::EXTRA_LOW;
        }
        return (TaskPriority)enumValue;
    }

    bool loadStaticMapping(const std::string& yamlFilePath) {
        using namespace YAML;
        try {
            std::lock_guard<std::mutex> lg(mtxForDynamic);
            Node root = YAML::LoadFile(yamlFilePath);
            auto cfg = root.as<Mapping_>();
            triggerIdPriorityStatic_vec = cfg.mapping;
            DC_SERVER_LOG_DEBUG << "triggerIdPriorityStatic size " << triggerIdPriorityStatic_vec.size();
        } catch (const YAML::Exception& e) {
            DC_SERVER_LOG_ERROR << e.what();
            return false;
        }
        return true;
    }

    bool loadDynamicMapping(const std::string& yamlFilePath) {
        using namespace YAML;
        try {
            std::lock_guard<std::mutex> lg(mtxForDynamic);
            Node root = YAML::LoadFile(yamlFilePath);
            auto cfg = root.as<Mapping_>();
            triggerIdPriorityDynamic_vec = cfg.mapping;
            DC_SERVER_LOG_DEBUG << "triggerIdPriorityDynamic_vec size " << triggerIdPriorityDynamic_vec.size();
        } catch (const YAML::Exception& e) {
            DC_SERVER_LOG_ERROR << e.what();
            return false;
        }
        return true;
    }

    static void changeDynamicMapping(const std::vector<TriggerIdPriority_>& mapping) {
        triggerIdPriorityDynamic_vec = mapping;
        DC_SERVER_LOG_DEBUG << "triggerIdPriorityDynamic size " << triggerIdPriorityDynamic_vec.size();
    } 

    static bool changeLimitAndCycle (const std::string cycle, const int limit) {
        try {
            std::lock_guard<std::mutex> lg(mtx_);
            YAML::Node root = YAML::LoadFile(triggerLimitYamlFilePath);
            std::string date = root["date"].as<std::string>();
            int triggerCount = root["triggerCount"].as<int>();
            std::ofstream output_file(triggerLimitYamlFilePath, std::ios::out | std::ios::binary);
            output_file << "date: ";
            output_file << date;
            output_file << "\ntriggerCount: ";
            output_file << triggerCount;
            output_file << "\ncycle: ";
            output_file << cycle;
            output_file << "\nlimit: ";
            output_file << limit;
            output_file.close();
        } catch (const YAML::Exception& e) {
            DC_SERVER_LOG_ERROR << e.what();
            return false;
        }
        return true;
    }

   private:
    YAML::Node currentConfig_;
    STATIC_CONST_CHAR_P k_0_dcConfig = "dc_config";
    STATIC_CONST_CHAR_P k_3_priority = "priority";
    std::vector<TriggerIdPriority_> triggerIdPriorityStatic_vec;
    static std::vector<TriggerIdPriority_> triggerIdPriorityDynamic_vec;
    static std::mutex mtx_;
    static std::mutex mtxForDynamic;
    static std::string triggerLimitYamlFilePath;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

YCS_ADD_STRUCT(hozon::netaos::dc::TriggerIdPriority_, triggerId, task, priority);
YCS_ADD_STRUCT(hozon::netaos::dc::Mapping_, mapping);

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_MANAGER_CONFIG_MANAGER_H
