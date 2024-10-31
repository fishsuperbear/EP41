/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: logic task monitor
 */

#ifndef PHM_LOGIC_H
#define PHM_LOGIC_H

#include <vector>
#include <unordered_map>
#include <unistd.h>
#include <mutex>
#include <memory>
#include <functional>
#include "phm/common/include/phm_config.h"


namespace hozon {
namespace netaos {
namespace phm {

struct LogicMessage {
    size_t cursor;
    uint32_t expect_point;
    std::vector<uint32_t> monitor_point_list;
};


class LogicMonitor {
public:
    static std::shared_ptr<LogicMonitor> MakeLogicMonitor();
    LogicMonitor(std::shared_ptr<ModuleConfig> cfg);
    ~LogicMonitor();

    void InitLogicMonitor(std::function<void(uint32_t, bool status)> fault_hook);

    void Run(uint32_t checkPointId);
    void Stop();

private:
    bool start_;
    std::mutex mtx_;
    std::function<void(uint32_t, bool)> fault_occure_hook_;
    // map<key, queue<point>>
    std::unordered_map<uint32_t, LogicMessage> logic_monitor_map_;
    std::shared_ptr<ModuleConfig> cfg_;
};

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_ALIVE_H
