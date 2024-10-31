/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: task.h
 * @Date: 2023/08/15
 * @Author: cheng
 * @Desc: --
 */

#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_BASIC_TASK_H
#define MIDDLEWARE_TOOLS_DATA_COLLECT_BASIC_TASK_H

#include <string>
#include <type_traits>
#include <utility>

#include "basic_task.h"
#include "include/yaml_cpp_struct.hpp"
#include "yaml-cpp/yaml.h"

namespace hozon {
namespace netaos {
namespace dc {
enum TaskStatus : int
{
    INITIAL = 0,
    CONFIGURED = 1,
    RUNNING = 2,
    FINISHED = 4,
    DELETED = 8,
    ERROR = 16,
};

struct strCmp{
    bool operator ()(std::string &a, std::string &b) {
        return a.compare(b)<0;
    }
};

struct TaskInfo {
    std::string type;
    std::string taskName;
    std::string policy;
    std::string lifecycle;
    bool createNew{true};
    bool waitReady{true};
    std::string endTime{"--"};
    std::string startTime{"--"};
    TaskStatus taskStatus{TaskStatus::INITIAL};
    std::vector<int> waitItems = {-1};
    int priority{-1};

    void setErrorStatus() { taskStatus = TaskStatus::ERROR; }
};

struct PipeLineTask {
    std::string taskName{};
    std::string lifecycle{};
    std::vector<TaskInfo> pipeLine;
    std::string trigger_id;
    int priority{-1};
};

enum TaskPriority { EXTRA_LOW = 0, LOW = 1, MID = 2, HIGH = 3, EXTRA_HIGH = 4 };

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

YCS_ADD_STRUCT(hozon::netaos::dc::TaskInfo, type, policy, lifecycle, createNew, waitReady, waitItems);
YCS_ADD_STRUCT(hozon::netaos::dc::PipeLineTask,lifecycle, taskName, pipeLine,priority);

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_BASIC_TASK_H
