/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: manager.h
 * @Date: 2023/08/08
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_MANAGER_MANAGER_H_
#define MIDDLEWARE_TOOLS_DATA_COLLECT_MANAGER_MANAGER_H_

#include <atomic>
#include <functional>
#include <thread>
#include <pthread.h>

#include "basic/basic_task.h"
#include "basic/trans_struct.h"
#include "manager/include/config_manager.h"
#include "thread_pool/include/thread_pool_flex.h"
#include "timer/timer_manager.hpp"
#include "utils/include/dc_logger.hpp"
#include "utils/include/path_utils.h"
#include "yaml-cpp/yaml.h"

#define CHECK_NODE_EMPTY(node)                                  \
    if (!node) {                                                \
        DC_SERVER_LOG_ERROR << #node << " shouldn't be empty!"; \
        return "";                                              \
    }

namespace hozon {
namespace netaos {
namespace dc {

class Manager {
 public:
  Manager() : threadPoolFlex_(1), stopFlag_(false), cfgm_(nullptr) { DC_SERVER_LOG_INFO << "null manager init end"; };

  explicit Manager(ConfigManager *cfgm) : threadPoolFlex_(3, 6), stopFlag_(false), cfgm_(cfgm) {
      DC_SERVER_LOG_INFO << "manager init start";
      DC_NEW(tm_, TimerManager());
      tm_->start(threadPoolFlex_, "mng_timer");
      DC_SERVER_LOG_INFO << "manager init end";
  };

  explicit Manager(std::string type,ConfigManager *cfgm) : threadPoolFlex_(cfgm->getRunningThreadNum(type), cfgm->getMaxThreadNum(type)), stopFlag_(false), cfgm_(cfgm) {
      DC_SERVER_LOG_INFO <<type<< " manager init start";
      DC_NEW(tm_, TimerManager());
      type_ = type;
      tm_->start(threadPoolFlex_, type+"_time");
      DC_SERVER_LOG_INFO  <<type << " manager init end";
  };

  virtual bool stop(uint8_t timeOutSecond) {
      stopFlag_.store(true, std::memory_order::memory_order_release);
      uint8_t sleepCount = 0;
      tm_->stopAll();
      threadPoolFlex_.stop();
      std::cout << "\n";
      std::atomic_bool allIsStoped = true;
      {
          std::lock_guard<std::mutex> lg(name2TaskMutex_);
          for (auto &taskItem : name2Task_) {
              if (taskItem.second->getStatus() == TaskStatus::RUNNING) {
                  taskItem.second->terminate();
              }
          }
          for (auto taskItem = name2Task_.begin(); taskItem != name2Task_.end();) {
              while (taskItem->second->getStatus() == TaskStatus::RUNNING && sleepCount< timeOutSecond) {
                  TimeUtils::sleep(1000);
                  sleepCount++;
                  continue;
              }
              DC_OPER_LOG_INFO<<__FILE__<<":"<<__LINE__;
              if (taskItem->second->getStatus() == TaskStatus::RUNNING) {
                  allIsStoped = false;
                  DC_OPER_LOG_INFO<<taskItem->first<<" task can't be stoped.";
                  taskItem++;
              } else {
                  DC_DELETE(taskItem->second);
                  name2Task_.erase(taskItem++);
              }
          }
      }
//      if (allIsStoped) {
//          DC_DELETE(tm_);
//      }
      DC_DELETE(tm_);
      return allIsStoped;
  }

  virtual ~Manager() {
      std::cout << "after destruct manager\n";
  }

  virtual BasicTask *getTask(std::string taskType) = 0;
  virtual std::string getTaskName(std::string taskType) = 0;
  virtual BasicTask * tryCreateTask(std::string taskType, std::string taskName) {
        BasicTask *taskInstance = nullptr;
        if (taskName.empty()) {
            taskName = getTaskName(taskType);
        }
        {
            std::lock_guard<std::mutex> lg(name2TaskMutex_);
            if (name2Task_.find(taskName) != name2Task_.end()) {
                return name2Task_[taskName];
            }
        }
        taskInstance = getTask(taskType);
        if (taskInstance == nullptr) {
            DC_SERVER_LOG_ERROR << "create basic task failed for " << taskType << ", taskName:" << taskName;
            return nullptr;
        }
        DC_SERVER_LOG_DEBUG << "begin save data";
        {
            std::lock_guard<std::mutex> lg(name2TaskMutex_);
            name2Task_[taskName] = taskInstance;
        }
        return taskInstance;
  }
  virtual std::string runTask(YAML::Node &configureYaml, struct DataTrans &dataStruct, std::string &taskNameInput) {
      DC_SERVER_LOG_INFO << "begin run task";
      if (PathUtils::isFileExist(PathUtils::debugModeFilePath)) {
            auto result = YAML::Dump(configureYaml);
      }
      CHECK_NODE_EMPTY(configureYaml["type"]);
      auto taskType = configureYaml["type"].as<std::string>();
      std::string tempTaskName;
      if (configureYaml["name"])  {
          tempTaskName = configureYaml["name"].as<std::string>();
      }
      if (!taskNameInput.empty()) {
        tempTaskName = taskNameInput;
      }
      if (tempTaskName.empty()) {
          tempTaskName = getTaskName(taskType);
      }
      BasicTask * taskInstance= tryCreateTask(taskType, tempTaskName);
      DC_SERVER_LOG_DEBUG << "begin configure";

      taskNameInput = tempTaskName;
      for (auto node : configureYaml["configuration"]) {
          if (node["type"]) {
              taskInstance->configure(node["type"].as<std::string>(), node);
          } else {
              taskInstance->configure("default", node);
          }
      }
      DC_SERVER_LOG_INFO<<"===================all item input for :"<<tempTaskName<<"====================";
      if (dataStruct.dataType != DataTransType::null) {
          for (auto type2PathList : dataStruct.pathsList) {
              DC_SERVER_LOG_DEBUG<<"pathList type:"<<std::string(magic_enum::enum_name(type2PathList.first));
              for (auto &path : type2PathList.second) {
                  DC_SERVER_LOG_INFO<<path;
              }
          }
          taskInstance->configure("default", dataStruct);
      }
      DC_SERVER_LOG_INFO<<"===================all item input for :"<<tempTaskName<<"====================";
      if (taskInstance->getStatus() == TaskStatus::ERROR) {
          DC_SERVER_LOG_ERROR << " task is error after configure:"<<tempTaskName;
          return "task is Error after configure";
      }
      auto priority = ConfigManager::getPriority(configureYaml);
      TimingConfiguration lifecycleCfg;
      YAML::Node lifecycleYaml;
      if (configureYaml["lifecycle"]
          && cfgm_->getLifeCycle(configureYaml["lifecycle"].as<std::string>(), lifecycleYaml)) {
          lifecycleCfg = lifecycleYaml.as<TimingConfiguration>();
      } else {
          DC_SERVER_LOG_ERROR << " the lifecycle is not configured, will not run the task";
          return "lifecycle not found";
      }

      // 没有周期配置， 默认立即执行一次。

      Timer* startTimer = tm_->addTimer(priority,
                                        lifecycleCfg.timeOutMs,
                                        lifecycleCfg.executeTimes,
                                        lifecycleCfg.intervalMs,
                                        [taskInstance, tempTaskName] {
          DC_SERVER_LOG_DEBUG << "call active of " << tempTaskName;
          pthread_setname_np(pthread_self(), tempTaskName.substr(0, 15).c_str());
                                            std::cout << "\n" << __FILE__ << ":" << __LINE__ << " call active\n";
          taskInstance->active();
      });
      std::vector<Timer *> relatedTimer;
      relatedTimer.push_back(startTimer);
      if (!lifecycleCfg.autoStop) {
          TimingConfiguration stopControl;
          // 需要手动停止
          if (lifecycleYaml["stopControl"]) {
              stopControl = lifecycleYaml["stopControl"].as<TimingConfiguration>();
          }
          // 如果没有配置， 这里不会执行。
          auto endTimer = tm_->addTimer(priority,
                        stopControl.timeOutMs,
                        stopControl.executeTimes,
                        stopControl.intervalMs,
                        [taskInstance, tempTaskName] {
                          std::cout << "\n" << __FILE__ << ":" << __LINE__ << " call de-active\n";
                          pthread_setname_np(pthread_self(), tempTaskName.substr(0, 15).c_str());
                          taskInstance->deactive();
                          DC_SERVER_LOG_DEBUG << "call de-active of " << tempTaskName;
                        });
          relatedTimer.push_back(endTimer);
      }
      {
          std::lock_guard<std::mutex> lg(name2TaskMutex_);
          task2Timer_[taskInstance] =relatedTimer;
      }
      std::cout << "finished runTask\n";
      taskNameInput = tempTaskName;
      return "";
  }

  virtual void deleteTask(std::string taskName) {
      std::lock_guard<std::mutex> lg(name2TaskMutex_);
      if (name2Task_.find(taskName) == name2Task_.end()) {
          DC_SERVER_LOG_ERROR << taskName << " task instance not found for deleteTask";
          return;
      }
      if (name2Task_[taskName]->getStatus()!=TaskStatus::RUNNING) {
          for (auto *timer : task2Timer_[name2Task_[taskName]]) {
              tm_->disableTimer(timer);
          }
          DC_DELETE(name2Task_[taskName]);
          name2Task_.erase(taskName);
      }
  };

  virtual bool getTaskResult(const std::string &taskName, struct DataTrans &dataStruct) {
      std::lock_guard<std::mutex> lg(name2TaskMutex_);
      if (name2Task_.find(taskName) == name2Task_.end()) {
          DC_SERVER_LOG_ERROR << taskName << " task instance not found for getTaskResult";
          return false;
      }
      return name2Task_[taskName]->getTaskResult("default", dataStruct);
  };

  virtual TaskStatus getTaskStatus(const std::string &taskName) {
      std::lock_guard<std::mutex> lg(name2TaskMutex_);
      if (name2Task_.find(taskName) == name2Task_.end()) {
          DC_SERVER_LOG_ERROR << taskName << " task instance not found for getTaskStatus";
          return TaskStatus::DELETED;
      }
      return name2Task_[taskName]->getStatus();
  };

  virtual bool stopTask(const std::string& taskName) {
      std::lock_guard<std::mutex> lg(name2TaskMutex_);
      if (name2Task_.find(taskName) == name2Task_.end()) {
          DC_SERVER_LOG_ERROR << taskName << " task instance not found for stopTask";
          return false;
      }
      name2Task_[taskName]->terminate();
      return true;
  }

 protected:
  std::map<std::string, BasicTask *> name2Task_;
  std::map<BasicTask *,std::vector<Timer *>> task2Timer_;
  std::mutex name2TaskMutex_;
  TimerManager *tm_;
  ThreadPoolFlex threadPoolFlex_;
  std::atomic_bool stopFlag_;
  ConfigManager *cfgm_;
  std::string type_{"manager"};
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_MANAGER_MANAGER_H_
