/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: system monitor manager
 */

#ifndef SYSTEM_MONITOR_MANAGER_H
#define SYSTEM_MONITOR_MANAGER_H

#include <mutex>
#include <unordered_map>
#include "system_monitor/include/common/system_monitor_def.h"
#include "system_monitor/include/monitor/system_monitor_base.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

class SystemMonitorManager {
public:
    static SystemMonitorManager* getInstance();

    void Init();
    void DeInit();

    void ControlEvent(const SystemMonitorControlEventInfo& info);
    void RefreshEvent(const std::string& reason);

private:
    void Start();
    void Stop();

private:
    SystemMonitorManager();
    SystemMonitorManager(const SystemMonitorManager &);
    SystemMonitorManager & operator = (const SystemMonitorManager &);

private:
    static std::mutex mtx_;
    static SystemMonitorManager* instance_;

    // map<sid, service>
    std::unordered_map<SystemMonitorSubFunctionId, SystemMonitorBase*> monitor_service_ptr_map_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_MANAGER_H