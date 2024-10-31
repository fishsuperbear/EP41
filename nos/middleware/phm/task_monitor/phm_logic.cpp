/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: logic task monitor
*/

#include <functional>

#include "phm/common/include/phm_logger.h"
#include "phm/task_monitor/include/phm_logic.h"
#include "phm/common/include/timer_manager.h"
#include "phm/common/include/phm_config.h"

namespace hozon {
namespace netaos {
namespace phm {


std::shared_ptr<LogicMonitor>
LogicMonitor::MakeLogicMonitor()
{
    std::shared_ptr<LogicMonitor> LogicMonitor;

    return LogicMonitor;
}

LogicMonitor::LogicMonitor(std::shared_ptr<ModuleConfig> cfg)
: start_(false)
, cfg_(cfg)
{

}

LogicMonitor::~LogicMonitor()
{
    PHM_DEBUG << "LogicMonitor::~LogicMonitor";
    start_ = false;
}

void
LogicMonitor::InitLogicMonitor(std::function<void(uint32_t, bool)> fault_hook)
{
    start_ = true;
    logic_monitor_map_.clear();
    std::vector<phm_task_t> origin_tasks = cfg_->GetPhmTask();
    for (auto& item : origin_tasks) {
        if (item.monitorType == PHM_MONITOR_TYPE_LOGIC) {
            uint32_t key = item.faultId*100 + item.faultObj;
            LogicMessage message;
            message.cursor = 0;
            for (auto& point : item.checkPointId) {
                message.monitor_point_list.emplace_back(point);
            }
            if (message.monitor_point_list.empty()) {
                message.expect_point = 999;
            }
            else {
                message.expect_point = message.monitor_point_list.at(0);
            }

            logic_monitor_map_.insert(std::make_pair(key, message));
        }
    }

    fault_occure_hook_ = fault_hook;
}

void
LogicMonitor::Run(uint32_t checkPointId)
{
    if (!start_) {
        return;
    }

    bool bStatus = false;
    for (auto& item : logic_monitor_map_) {
        if (item.second.expect_point == checkPointId) {
            item.second.cursor++;
            if (item.second.cursor == item.second.monitor_point_list.size()) {
                PHM_DEBUG << "LogicMonitor::Run Fault " << item.first << " Logic Monitor Finish, Start Next Loop";
                item.second.cursor = 0;
                item.second.expect_point = item.second.monitor_point_list.at(0);

                if (fault_occure_hook_ != nullptr) {
                    fault_occure_hook_(item.first, bStatus);
                }
            }
            else {
                item.second.expect_point = item.second.monitor_point_list.at(item.second.cursor);
            }
        }
        else {
            if (item.second.cursor) {
                PHM_DEBUG << "LogicMonitor::Run Fault " << item.first << " Occur";
                bStatus = true;
                if (fault_occure_hook_ != nullptr) {
                    fault_occure_hook_(item.first, bStatus);
                }
            }
        }
    }
}

void
LogicMonitor::Stop()
{
    start_ = false;
}

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
