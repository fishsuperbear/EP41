/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: processor_manager.h
 * @Date: 2023/08/11
 * @Author: cheng
 * @Desc: --
 */

#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_PROCESS_PROCESSOR_MANAGER_H
#define MIDDLEWARE_TOOLS_DATA_COLLECT_PROCESS_PROCESSOR_MANAGER_H

#include <semaphore.h>
#include <mutex>
#include <vector>

#include "manager/include/manager.h"
#include "processor_factory.h"
#include "thread_pool/include/thread_pool_flex.h"
#include "timer/timer_manager.hpp"
#include "utils/include/dc_logger.hpp"
#include "yaml-cpp/yaml.h"

namespace hozon {
namespace netaos {
namespace dc {

class ProcessorManager : public Manager {
   public:
    ProcessorManager(ConfigManager* cfgm);
    ProcessorManager(std::string type, ConfigManager* cfgm);
    ~ProcessorManager();

    BasicTask* getTask(std::string taskType) override { return ProcessorFactory::getInstance()->createProcess(magic_enum::enum_cast<ProcessorEnum>(taskType).value()); };

    std::string getTaskName(std::string taskType) override { return taskType + TimeUtils::timestamp2ReadableStr(TimeUtils::getDataTimestamp()); };

    std::string runTaskb(YAML::Node& configureYaml, struct DataTrans& dataStruct) {
        proc_ = ProcessorFactory::getInstance()->createProcess(ProcessorEnum::COPIER);
        std::string configYamlStr = YAML::Dump(configureYaml);

        auto timestamp = TimeUtils::getDataTimestamp();
        proc_->configure("", dataStruct);
        std::string timeStr = TimeUtils::formatTimeStrForFileName(timestamp);
        auto nameNode = configureYaml["name"];
        auto taskName = nameNode.as<std::string>();

        tm_->addTimer(TaskPriority::EXTRA_HIGH, 0, 20, 1000, [this, taskName] {
            std::cout << "\ncall copier\n";
            proc_->active();
            std::cout << "\nafter call copier\n";
        });
        return "";
    }


    //    bool getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) override { return proc_->getTaskResult(taskName, dataStruct); }
    //
    //    TaskStatus getTaskStatus(const std::string& taskName) override { return proc_->getStatus(); }

   private:
    ThreadPoolFlex threadPoolFlex_;
    TimerManager* tm_;
    Processor* proc_;
    std::atomic_bool stopFlag_;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_PROCESS_PROCESSOR_MANAGER_H
